import gc
import math
import os
import time

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm.auto import tqdm

from .utils import safe_ppl, get_gpu_mem_mb, save_params_csv, append_metrics_csv
from .l1ra.model import L1RAModel

try:
    from peft.tuners.adalora import AdaLoraModel as _AdaLoraModel
    def _is_adalora(model) -> bool:
        return hasattr(model, "base_model") and isinstance(model.base_model, _AdaLoraModel)
except ImportError:
    def _is_adalora(model) -> bool:
        return False


def _is_l1ra(model) -> bool:
    return isinstance(model, L1RAModel)


def build_optimizer(model, lr: float = 1e-4, weight_decay: float = 0.01) -> AdamW:
    trainable = [p for p in model.parameters() if p.requires_grad]
    return AdamW(trainable, lr=lr, weight_decay=weight_decay)


def build_l1ra_optimizer(model: L1RAModel, lr: float, weight_decay: float = 0.01) -> AdamW:
    """Оптимизатор L1RA с отдельным learning rate для gate-векторов."""
    eta_c = model.peft_config[model.trainable_adapter_name].eta_c
    gate_params  = [p for n, p in model.named_parameters() if "lora_c" in n and p.requires_grad]
    other_params = [p for n, p in model.named_parameters() if "lora_c" not in n and p.requires_grad]
    return AdamW(
        [
            {"params": gate_params,  "lr": eta_c,  "weight_decay": 0.0},
            {"params": other_params, "lr": lr,     "weight_decay": weight_decay},
        ]
    )


def build_scheduler(
    optimizer,
    train_loader_len: int,
    num_epochs: int = 5,
    grad_accum_steps: int = 32,
    warmup_ratio: float = 0.05,
):
    total_steps  = num_epochs * math.ceil(train_loader_len / grad_accum_steps)
    warmup_steps = int(warmup_ratio * total_steps)
    return get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)


def _l1_loss(model: L1RAModel) -> torch.Tensor:
    """Средняя L1-норма gate-векторов."""
    cfg   = model.peft_config[model.trainable_adapter_name]
    coef  = cfg.l1ra_lambda
    if coef <= 0:
        return torch.tensor(0.0)
    total, count = torch.tensor(0.0), 0
    for n, p in model.named_parameters():
        if "lora_c" in n and p.requires_grad:
            total = total + p.abs().sum()
            count += p.numel()
    return coef * total / max(count, 1)

@torch.no_grad()
def evaluate(model, val_loader: DataLoader, device: str, epoch: int, optimizer_step: int) -> dict:
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    loss_sum, n_steps = 0.0, 0
    start = time.time()

    for batch in tqdm(val_loader, desc=f"Val | epoch {epoch}", leave=False):
        batch = {k: v.to(device) for k, v in batch.items()}
        loss_sum += model(**batch).loss.item()
        n_steps  += 1

    avg_loss = loss_sum / max(n_steps, 1)
    return dict(
        epoch=epoch,
        step=optimizer_step,
        split="val",
        loss=avg_loss,
        ppl=safe_ppl(avg_loss),
        gpu_mem_mb=get_gpu_mem_mb(),
        time_sec=round(time.time() - start, 2),
    )


_L1RA_ADAPTER_KEYS = ("lora_A", "lora_B", "lora_c")


def _save_checkpoint(model, output_dir: str, step: int):
    ckpt_dir = os.path.join(output_dir, "checkpoints", f"step_{step}")
    os.makedirs(ckpt_dir, exist_ok=True)

    if isinstance(model, L1RAModel):
        adapter_state = {
            k: v.cpu()
            for k, v in model.state_dict().items()
            if any(key in k for key in _L1RA_ADAPTER_KEYS)
        }
        torch.save(adapter_state, os.path.join(ckpt_dir, "adapter_model.pt"))
    else:
        model.save_pretrained(ckpt_dir)


def train_model(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer,
    scheduler,
    output_dir: str = "./outputs",
    num_epochs: int = 5,
    grad_clip: float = 1.0,
    grad_accum_steps: int = 32,
    log_every: int = 100,
    device: str | None = None,
    save_steps: int | None = None,
    log_per_step: bool = False,
    detector=None,
    start_optimizer_step: int = 0,
    max_optimizer_steps: int | None = None,
    skip_initial_batches: int = 0,
) -> list[dict]:
    """Общий цикл обучения для LoRA, AdaLoRA и L1RA.

    Args:
        save_steps: частота сохранения адаптера в optimizer steps.
        log_per_step: логировать каждый optimizer step, а не среднее по окну.
        start_optimizer_step: начальный номер шага, нужен для продолжения с warm-start.
        max_optimizer_steps: остановка по числу optimizer steps.
        skip_initial_batches: сколько batch'ей пропустить перед продолжением обучения.
    """
    os.makedirs(output_dir, exist_ok=True)
    metrics_csv = os.path.join(output_dir, "metrics.csv")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else \
                 "mps"  if torch.backends.mps.is_available() else "cpu"

    model = model.to(device)

    for group in optimizer.param_groups:
        for p in group['params']:
            st = optimizer.state.get(p)
            if not st:
                continue
            for k, v in list(st.items()):
                if isinstance(v, torch.Tensor) and v.device != p.device:
                    st[k] = v.to(p.device)

    save_params_csv(model, output_dir)

    is_l1ra_model    = _is_l1ra(model)
    is_adalora_model = _is_adalora(model)

    metric_rows  = []
    optimizer_step = int(start_optimizer_step)
    skip_remaining = int(skip_initial_batches)

    def _log(row: dict):
        metric_rows.append(row)
        append_metrics_csv(row, metrics_csv)

    for epoch in range(1, num_epochs + 1):
        model.train()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        epoch_start = time.time()
        loss_sum, n_steps = 0.0, 0
        window_loss_sum, window_n = 0.0, 0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Train | epoch {epoch}", leave=False)
        for step, batch in enumerate(pbar, start=1):
            if skip_remaining > 0:
                skip_remaining -= 1
                if skip_remaining == 0:
                    pbar.set_postfix_str('пропуск завершен, обучение продолжено')
                continue

            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss

            if is_l1ra_model:
                loss = loss + _l1_loss(model).to(device)

            batch_loss = loss.item()
            loss_sum += batch_loss
            n_steps  += 1
            window_loss_sum += batch_loss
            window_n        += 1

            (loss / grad_accum_steps).backward()

            if step % grad_accum_steps == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer_step += 1

                if is_adalora_model:
                    model.base_model.update_and_allocate(optimizer_step)

                optimizer.zero_grad()

                avg_loss = loss_sum / n_steps
                pbar.set_postfix(
                    loss=f"{avg_loss:.4f}",
                    ppl=f"{safe_ppl(avg_loss):.2f}",
                    lr=f"{scheduler.get_last_lr()[0]:.2e}",
                )

                if log_per_step:
                    window_mean = window_loss_sum / max(window_n, 1)
                    _log(dict(
                        epoch=epoch,
                        step=optimizer_step,
                        split="train",
                        loss=window_mean,
                        ppl=safe_ppl(window_mean),
                        gpu_mem_mb=get_gpu_mem_mb(),
                        lr=scheduler.get_last_lr()[0],
                    ))
                    window_loss_sum, window_n = 0.0, 0
                elif optimizer_step % log_every == 0:
                    _log(dict(
                        epoch=epoch,
                        step=optimizer_step,
                        split="train",
                        loss=avg_loss,
                        ppl=safe_ppl(avg_loss),
                        gpu_mem_mb=get_gpu_mem_mb(),
                        lr=scheduler.get_last_lr()[0],
                    ))
                    torch.cuda.empty_cache()
                    gc.collect()

                if save_steps is not None and save_steps > 0 and optimizer_step % save_steps == 0:
                    _save_checkpoint(model, output_dir, optimizer_step)
                    if detector is not None:
                        detector.on_checkpoint(model, optimizer, scheduler,
                                                optimizer_step, output_dir)

                if max_optimizer_steps is not None and optimizer_step >= max_optimizer_steps:
                    break

        avg_train_loss = loss_sum / max(n_steps, 1)
        _log(dict(
            epoch=epoch,
            step=optimizer_step,
            split="train",
            loss=avg_train_loss,
            ppl=safe_ppl(avg_train_loss),
            gpu_mem_mb=get_gpu_mem_mb(),
            lr=scheduler.get_last_lr()[0],
            time_sec=round(time.time() - epoch_start, 2),
        ))

        val_row = evaluate(model, val_loader, device, epoch, optimizer_step)
        _log(val_row)

        if max_optimizer_steps is not None and optimizer_step >= max_optimizer_steps:
            break

    _save_checkpoint(model, output_dir, optimizer_step)

    return metric_rows
