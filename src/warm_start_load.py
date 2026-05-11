"""Конвертация warm-start snapshot из AdaLoRA/L1RA в plain LoRA."""
from __future__ import annotations

import json
import os
import re
import torch
from torch.optim import AdamW

from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM


def load_snapshot(snapshot_dir: str) -> dict:
    trainable      = torch.load(os.path.join(snapshot_dir, 'model_trainable.pt'),
                                map_location='cpu', weights_only=False)
    training_state = torch.load(os.path.join(snapshot_dir, 'training_state.pt'),
                                map_location='cpu', weights_only=False)
    with open(os.path.join(snapshot_dir, 'metadata.json')) as f:
        meta = json.load(f)
    return {
        'trainable':      trainable,
        'training_state': training_state,
        'metadata':       meta,
    }


_LAYER_RE = re.compile(r'layers\.(\d+)\.(.+)$')


def _parse_lora_key(name: str):
    """Парсит имя LoRA-параметра."""
    if '.lora_' not in name:
        return None
    prefix, _, suffix = name.rpartition('.lora_')
    role = suffix[0]
    if role not in 'ABEc':
        return None
    m = _LAYER_RE.search(prefix)
    if m is None:
        return None
    layer = int(m.group(1))
    sub = m.group(2).split('.')[-1]
    return layer, sub, role


def collect_per_module(state_dict: dict) -> dict:
    """Группирует A, B и gate по модулю."""
    groups: dict = {}
    for name, t in state_dict.items():
        parsed = _parse_lora_key(name)
        if parsed is None:
            continue
        layer, sub, role = parsed
        key = (layer, sub)
        slot = groups.setdefault(key, {})
        if role == 'A':
            slot['A'] = t
        elif role == 'B':
            slot['B'] = t
        elif role in ('E', 'c'):
            slot['gate'] = t.flatten()
    return groups


def find_global_threshold(
    groups: dict,
    target_mean_k: float,
    max_iter: int = 80,
) -> float:
    """Подбирает общий порог tau так, чтобы средний активный ранг был target_mean_k."""
    gates = [g['gate'].abs() for g in groups.values() if 'gate' in g]
    if not gates:
        raise RuntimeError('no gates in groups')
    g_max = max(g.max().item() for g in gates)

    def mean_k(tau: float) -> float:
        ks = [int((g >= tau).sum()) for g in gates]
        return sum(ks) / len(ks)

    lo, hi = 0.0, g_max * 1.0001
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if mean_k(mid) > target_mean_k:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def compute_masks(groups: dict, tau: float) -> dict:
    """Маски выживших компонент для каждого модуля."""
    masks = {}
    for k, g in groups.items():
        if 'gate' not in g:
            continue
        masks[k] = (g['gate'].abs() >= tau)
    return masks


def survivor_counts(masks: dict) -> dict:
    return {k: int(m.sum()) for k, m in masks.items()}


def absorb_and_prune(
    slot: dict,
    mask: torch.Tensor,
    method: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Впитывает gate в A, применяет mask и возвращает PEFT-формат A/B."""
    A = slot['A']
    B = slot['B']
    gate = slot['gate']
    surv = mask.bool()

    if method == 'adalora':
        A_full = A * gate.view(-1, 1)
        B_full = B
        return (
            A_full[surv, :].contiguous(),
            B_full[:, surv].contiguous(),
        )

    elif method == 'l1ra':
        A_with_c = A * gate.view(1, -1)
        A_full = A_with_c.T
        B_full = B.T
        return (
            A_full[surv, :].contiguous(),
            B_full[:, surv].contiguous(),
        )

    else:
        raise ValueError(f'unknown method: {method}')


def _build_target_modules_regex(survivor_keys: list[tuple[int, str]]) -> str:
    """Регулярное выражение для выбора модулей с ненулевым рангом."""
    by_sub: dict = {}
    for (l, sub) in survivor_keys:
        by_sub.setdefault(sub, set()).add(int(l))
    parts = []
    for sub, layers in by_sub.items():
        layers_pat = '|'.join(str(l) for l in sorted(layers))
        parts.append(rf'.*\.layers\.({layers_pat})\..*\.{re.escape(sub)}')
    return '|'.join(parts)


def build_warm_started_lora(
    base_model_name: str,
    survivor_keys: list[tuple[int, str]],
    rank_pattern: dict,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    torch_dtype = None,
):
    """Создает plain LoRA с неоднородным rank_pattern."""
    if torch_dtype is None:
        torch_dtype = torch.bfloat16

    base = AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype=torch_dtype, trust_remote_code=True,
    )
    base.config.use_cache = False

    target_re = _build_target_modules_regex(survivor_keys)
    default_r = max(rank_pattern.values()) if rank_pattern else 16

    cfg = LoraConfig(
        r=default_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias='none',
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_re,
        rank_pattern=rank_pattern,
    )
    model = get_peft_model(base, cfg)
    return model


def assign_lora_weights(
    new_model,
    abs_pruned: dict,
):
    """Копирует подготовленные A/B в новую LoRA-модель."""
    matched, skipped = 0, []
    for name, param in new_model.named_parameters():
        if '.lora_' not in name:
            continue
        prefix, _, suffix = name.rpartition('.lora_')
        role = suffix[0]
        if role not in ('A', 'B'):
            continue
        m = _LAYER_RE.search(prefix)
        if m is None:
            skipped.append(name)
            continue
        layer = int(m.group(1))
        sub = m.group(2).split('.')[-1]
        key = (layer, sub)
        if key not in abs_pruned:
            skipped.append(name)
            continue
        A_p, B_p = abs_pruned[key]
        target = A_p if role == 'A' else B_p
        if param.shape != target.shape:
            raise RuntimeError(
                f'shape mismatch for {name}: param {tuple(param.shape)}, target {tuple(target.shape)}'
            )
        with torch.no_grad():
            param.data.copy_(target.to(param.device, dtype=param.dtype))
        matched += 1
    return matched, skipped


def _name_to_id_for_method(model, method: str) -> dict:
    """Восстанавливает порядок параметров в AdamW state."""
    name_to_id: dict = {}
    cnt = 0
    if method == 'adalora':
        for n, p in model.named_parameters():
            if p.requires_grad:
                name_to_id[n] = cnt; cnt += 1
    elif method == 'l1ra':
        for n, p in model.named_parameters():
            if 'lora_c' in n and p.requires_grad:
                name_to_id[n] = cnt; cnt += 1
        for n, p in model.named_parameters():
            if 'lora_c' not in n and p.requires_grad:
                name_to_id[n] = cnt; cnt += 1
    else:
        raise ValueError(method)
    return name_to_id


def _load_old_state_by_name(old_optim_state: dict, name_to_id: dict) -> tuple[dict, object]:
    """Перекладывает optimizer state из id-параметров в имена."""
    state = old_optim_state['state']
    by_name = {}
    g_step = None
    for n, pid in name_to_id.items():
        s = state.get(pid)
        if s is None:
            continue
        by_name[n] = {
            'exp_avg':    s.get('exp_avg'),
            'exp_avg_sq': s.get('exp_avg_sq'),
            'step':       s.get('step'),
        }
        if g_step is None:
            g_step = s.get('step')
    return by_name, g_step


def _find_old_name(old_names, layer: int, sub: str, role: str) -> str | None:
    """Ищет старое имя параметра по слою, модулю и роли."""
    needle_layer = f'.layers.{layer}.'
    needle_sub   = f'.{sub}.'
    target_role  = f'lora_{role}'
    for n in old_names:
        if needle_layer in n and needle_sub in n and target_role in n:
            return n
    return None


def transfer_optimizer_state(
    old_optimizer_state: dict,
    old_model,
    new_optimizer: AdamW,
    new_model,
    masks: dict,
    method: str,
) -> tuple[int, list, object]:
    """Переносит AdamW моменты для выживших LoRA-компонент."""
    name_to_id = _name_to_id_for_method(old_model, method)
    old_state, old_step = _load_old_state_by_name(old_optimizer_state, name_to_id)
    old_names = list(old_state.keys())

    transferred, missing = 0, []
    for name, p in new_model.named_parameters():
        if not p.requires_grad or '.lora_' not in name:
            continue
        prefix, _, suffix = name.rpartition('.lora_')
        role = suffix[0]
        if role not in ('A', 'B'):
            continue
        m = _LAYER_RE.search(prefix)
        if m is None:
            missing.append(name); continue
        layer = int(m.group(1))
        sub = m.group(2).split('.')[-1]
        key = (layer, sub)
        mask = masks.get(key)
        if mask is None:
            missing.append(name); continue
        surv = mask.bool()

        old_name = _find_old_name(old_names, layer, sub, role)
        if old_name is None:
            missing.append(name); continue
        old = old_state[old_name]
        m1, m2 = old['exp_avg'], old['exp_avg_sq']
        if m1 is None or m2 is None:
            missing.append(f'{name} (no moments yet)'); continue

        if method == 'adalora':
            if role == 'A':
                m1_s = m1[surv, :].clone()
                m2_s = m2[surv, :].clone()
            else:
                m1_s = m1[:, surv].clone()
                m2_s = m2[:, surv].clone()
        elif method == 'l1ra':
            if role == 'A':
                m1_s = m1.T[surv, :].contiguous().clone()
                m2_s = m2.T[surv, :].contiguous().clone()
            else:
                m1_s = m1.T[:, surv].contiguous().clone()
                m2_s = m2.T[:, surv].contiguous().clone()
        else:
            raise ValueError(method)

        if m1_s.shape != p.shape:
            missing.append(f'{name} (shape mismatch {tuple(m1_s.shape)} vs {tuple(p.shape)})')
            continue

        step_val = old_step
        if step_val is None:
            step_val = torch.tensor(0.0)
        elif not isinstance(step_val, torch.Tensor):
            step_val = torch.tensor(float(step_val))

        new_optimizer.state[p] = {
            'step':       step_val.clone(),
            'exp_avg':    m1_s.to(p.device, dtype=p.dtype),
            'exp_avg_sq': m2_s.to(p.device, dtype=p.dtype),
        }
        transferred += 1

    return transferred, missing, old_step


def warm_start_to_plain_lora(
    snapshot_dir: str,
    method: str,
    base_model_name: str,
    target_mean_k: int = 16,
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    rebuild_source_model_fn = None,
    scheduler_builder = None,
):
    """Готовит plain LoRA для продолжения обучения с warm-start snapshot.

    Args:
        snapshot_dir: папка `warm_start_snapshot`.
        method: исходный метод, `adalora` или `l1ra`.
        base_model_name: HuggingFace id базовой модели.
        target_mean_k: целевой средний ранг после pruning.
        rebuild_source_model_fn: функция, создающая исходную AdaLoRA/L1RA модель.
        scheduler_builder: функция создания scheduler для нового optimizer.

    Returns:
        Словарь с моделью, optimizer, scheduler, rank_pattern и метаданными.
    """
    snap = load_snapshot(snapshot_dir)
    trainable      = snap['trainable']
    training_state = snap['training_state']
    metadata       = snap['metadata']
    trigger_step   = int(metadata['trigger_step'])

    groups = collect_per_module(trainable)

    tau = find_global_threshold(groups, target_mean_k=target_mean_k)
    masks = compute_masks(groups, tau)
    K = survivor_counts(masks)

    survivor_keys = [k for k, kc in K.items() if kc > 0]

    abs_pruned = {
        k: absorb_and_prune(groups[k], masks[k], method)
        for k in survivor_keys
    }

    rank_pattern_by_full: dict = {}
    for (layer, sub) in survivor_keys:
        rp_key = rf'layers\.{layer}\..*\.{re.escape(sub)}'
        rank_pattern_by_full[rp_key] = K[(layer, sub)]

    new_model = build_warm_started_lora(
        base_model_name=base_model_name,
        survivor_keys=survivor_keys,
        rank_pattern=rank_pattern_by_full,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )

    matched, skipped = assign_lora_weights(new_model, abs_pruned)

    trainable_new = [p for p in new_model.parameters() if p.requires_grad]
    new_optimizer = AdamW(trainable_new, lr=lr, weight_decay=weight_decay)

    if rebuild_source_model_fn is None:
        raise ValueError('rebuild_source_model_fn is required to map snapshot params')
    src_model = rebuild_source_model_fn()
    missing_keys, unexpected = src_model.load_state_dict(trainable, strict=False)
    _ = missing_keys, unexpected

    transferred, miss_optim, old_step = transfer_optimizer_state(
        old_optimizer_state=training_state['optimizer_state_dict'],
        old_model=src_model,
        new_optimizer=new_optimizer,
        new_model=new_model,
        masks=masks,
        method=method,
    )

    scheduler = None
    if scheduler_builder is not None:
        scheduler = scheduler_builder(new_optimizer)
        try:
            scheduler.load_state_dict(training_state['scheduler_state_dict'])
        except Exception as e:
            print(
                f'[warm_start_load] scheduler не загрузился ({e}); '
                f'создаю новый и довожу до trigger_step={trigger_step}'
            )
            scheduler = scheduler_builder(new_optimizer)
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=UserWarning)
                for _ in range(trigger_step):
                    scheduler.step()

    try:
        if 'torch_rng_state' in training_state:
            rs = training_state['torch_rng_state']
            if isinstance(rs, torch.Tensor):
                torch.set_rng_state(rs.cpu().to(torch.uint8))
        if 'cuda_rng_state' in training_state and torch.cuda.is_available():
            cs = training_state['cuda_rng_state']
            if cs is not None:
                torch.cuda.set_rng_state_all([t.cpu().to(torch.uint8) for t in cs])
        if 'numpy_rng_state' in training_state:
            import numpy as np
            np.random.set_state(training_state['numpy_rng_state'])
    except Exception as e:
        print(f'[warm_start_load] RNG restore failed: {e}; continuing with current RNG')

    del src_model

    return {
        'model':         new_model,
        'optimizer':     new_optimizer,
        'scheduler':     scheduler,
        'training_state': training_state,
        'metadata':      metadata,
        'calibration':   {'tau': tau, 'target_mean_k': target_mean_k},
        'survivor_keys': survivor_keys,
        'rank_pattern':  {f'layer{l}.{s}': K[(l, s)] for (l, s) in survivor_keys},
        'K_per_module':  K,
        'trigger_step':  trigger_step,
        'matched_weights':   matched,
        'skipped_weights':   skipped,
        'transferred_optim': transferred,
        'missing_optim':     miss_optim,
    }
