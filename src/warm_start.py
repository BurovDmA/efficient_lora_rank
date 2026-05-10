"""Детектор стабилизации рангов для warm-start эксперимента."""
from __future__ import annotations

import json
import os
import re
import time
from typing import Optional

import numpy as np
import torch


def _svd_values_lowrank(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Сингулярные числа low-rank произведения left @ right."""
    _, R_l = torch.linalg.qr(left)
    _, R_r = torch.linalg.qr(right.T)
    inner = R_l @ R_r.T
    return torch.linalg.svdvals(inner)


def _energy_rank(sigma: torch.Tensor, threshold: float = 0.95) -> int:
    s2 = sigma.pow(2)
    cumsum = torch.cumsum(s2, dim=0)
    total = cumsum[-1]
    if total <= 0:
        return 0
    frac = cumsum / total
    idx = torch.searchsorted(frac, torch.tensor(threshold, device=frac.device))
    k = int(idx.item()) + 1
    return min(k, len(sigma))


_LAYER_RE = re.compile(r'layers\.(\d+)\.(.+)$')


def _parse_lora_key(name: str):
    """Выделяет номер слоя, имя модуля и тип LoRA-параметра."""
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
    module_path = m.group(2)
    module = module_path.split('.')[-1]
    return layer, module, role


def _collect_module_tensors(model) -> dict:
    """Группирует LoRA-тензоры по слою и модулю."""
    groups: dict = {}
    for name, param in model.named_parameters():
        parsed = _parse_lora_key(name)
        if parsed is None:
            continue
        layer, module, role = parsed
        key = (layer, module)
        slot = groups.setdefault(key, {})
        if role == 'A':
            slot['A'] = param.detach()
        elif role == 'B':
            slot['B'] = param.detach()
        elif role == 'c':
            slot['C'] = param.detach()
        elif role == 'E':
            slot['E'] = param.detach()
    return {k: v for k, v in groups.items() if 'A' in v and 'B' in v}


def effective_rank_vector(model, energy_threshold: float = 0.95) -> np.ndarray:
    """Вектор per-module energy rank в фиксированном порядке."""
    groups = _collect_module_tensors(model)
    if not groups:
        return np.array([])

    ranks = []
    for key in sorted(groups.keys()):
        d = groups[key]
        A = d['A']
        B = d['B']
        # PEFT: A=(r, d_in), B=(d_out, r); L1RA: A=(d_in, r), B=(r, d_out).
        gate = d.get('C', d.get('E', None))
        if gate is not None:
            gate = gate.flatten()
            r = gate.size(0)
            if B.shape[-1] == r and A.shape[0] == r:
                eff_B = B * gate.view(1, -1)
                eff_A = A
            elif B.shape[0] == r and A.shape[-1] == r:
                eff_B = A * gate.view(1, -1)
                eff_A = B
            else:
                continue
        else:
            if B.shape[-1] == A.shape[0]:
                eff_B = B
                eff_A = A
            elif A.shape[-1] == B.shape[0]:
                eff_B = A
                eff_A = B
            else:
                continue

        if eff_B.shape[1] != eff_A.shape[0]:
            continue
        try:
            with torch.no_grad():
                sigma = _svd_values_lowrank(eff_B.float(), eff_A.float())
            r = _energy_rank(sigma, energy_threshold)
        except Exception:
            continue
        ranks.append(r)
    return np.array(ranks, dtype=float)


def _spearmanr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3 or a.size != b.size:
        return float('nan')
    a_r = np.argsort(np.argsort(a)).astype(float)
    b_r = np.argsort(np.argsort(b)).astype(float)
    if a_r.std() == 0 or b_r.std() == 0:
        return float('nan')
    return float(np.corrcoef(a_r, b_r)[0, 1])


class WarmStartDetector:
    """Сохраняет snapshot, когда per-module rank pattern стабилизировался."""

    def __init__(
        self,
        save_steps: int = 50,
        window: int = 100,
        k_consecutive: int = 3,
        threshold: float = 0.90,
        energy_threshold: float = 0.95,
        verbose: bool = True,
    ):
        self.save_steps = save_steps
        self.window = window
        self.k_consecutive = k_consecutive
        self.threshold = threshold
        self.energy_threshold = energy_threshold
        self.verbose = verbose

        self.rank_vectors: dict[int, np.ndarray] = {}
        self.triggered: bool = False
        self.trigger_step: Optional[int] = None
        self.trigger_rhos: Optional[list] = None

    def on_checkpoint(self, model, optimizer, scheduler, step: int, output_dir: str):
        if self.triggered:
            return

        vec = effective_rank_vector(model, self.energy_threshold)
        if vec.size == 0:
            return
        self.rank_vectors[step] = vec

        required = set()
        for i in range(self.k_consecutive):
            now = step - i * self.save_steps
            required.add(now)
            required.add(now - self.window)
        if not required.issubset(self.rank_vectors.keys()):
            return

        rhos = []
        for i in range(self.k_consecutive):
            s1 = step - i * self.save_steps
            s2 = s1 - self.window
            rho = _spearmanr(self.rank_vectors[s1], self.rank_vectors[s2])
            rhos.append(rho)

        if self.verbose:
            print(f'[WSD] step={step}  ρ({self.window})×{self.k_consecutive} = '
                  + ', '.join(f'{r:.3f}' for r in rhos))

        if all((not np.isnan(r)) and r >= self.threshold for r in rhos):
            self._save_snapshot(model, optimizer, scheduler, step, output_dir, rhos)
            self.triggered = True
            self.trigger_step = step
            self.trigger_rhos = rhos

    def _save_snapshot(self, model, optimizer, scheduler, step: int,
                        output_dir: str, rhos: list):
        snap_dir = os.path.join(output_dir, 'warm_start_snapshot')
        os.makedirs(snap_dir, exist_ok=True)

        training_state = {
            'step': step,
            'rhos': rhos,
            'window': self.window,
            'k_consecutive': self.k_consecutive,
            'threshold': self.threshold,
            'energy_threshold': self.energy_threshold,
            'rank_vector': self.rank_vectors[step].tolist(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'torch_rng_state': torch.get_rng_state(),
            'cuda_rng_state': (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
            'numpy_rng_state': np.random.get_state(),
        }
        torch.save(training_state, os.path.join(snap_dir, 'training_state.pt'))

        trainable = {
            name: param.detach().cpu()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        torch.save(trainable, os.path.join(snap_dir, 'model_trainable.pt'))

        meta = {
            'trigger_step': step,
            'rhos': [float(r) for r in rhos],
            'window': self.window,
            'k_consecutive': self.k_consecutive,
            'threshold': self.threshold,
            'energy_threshold': self.energy_threshold,
            'rank_vector_len': len(self.rank_vectors[step]),
            'rank_vector_mean': float(np.mean(self.rank_vectors[step])),
            'rank_vector_min': float(np.min(self.rank_vectors[step])),
            'rank_vector_max': float(np.max(self.rank_vectors[step])),
            'timestamp': time.time(),
        }
        with open(os.path.join(snap_dir, 'metadata.json'), 'w') as f:
            json.dump(meta, f, indent=2)

        if self.verbose:
            print(f'[WSD] *** TRIGGERED at step={step} ***')
            print(f'[WSD]     ρ×{self.k_consecutive}: ' + ', '.join(f'{r:.3f}' for r in rhos))
            print(f'[WSD]     snapshot: {snap_dir}')
