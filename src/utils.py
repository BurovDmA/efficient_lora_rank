import csv
import math
import os

import torch


METRICS_FIELDS = ["epoch", "step", "split", "loss", "ppl", "gpu_mem_mb", "lr", "time_sec"]
PARAMS_FIELDS  = ["name", "numel"]


def safe_ppl(loss_value: float) -> float:
    try:
        return math.exp(min(loss_value, 20))
    except OverflowError:
        return float("inf")


def get_gpu_mem_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024**2
    return 0.0


def save_params_csv(model, output_dir: str) -> None:
    """Сохраняет список обучаемых параметров."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "params.csv")

    total_params     = sum(p.numel() for p in model.parameters())
    total_trainable  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_pct    = 100 * total_trainable / max(total_params, 1)

    rows = [{"name": name, "numel": param.numel()}
            for name, param in model.named_parameters() if param.requires_grad]

    rows += [
        {"name": "total_params",     "numel": total_params},
        {"name": "total_trainable",  "numel": total_trainable},
        {"name": "trainable_pct",    "numel": round(trainable_pct, 4)},
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PARAMS_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def append_metrics_csv(row: dict, csv_path: str) -> None:
    """Добавляет строку метрик в CSV."""
    is_new = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=METRICS_FIELDS, extrasaction="ignore")
        if is_new:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in METRICS_FIELDS})
