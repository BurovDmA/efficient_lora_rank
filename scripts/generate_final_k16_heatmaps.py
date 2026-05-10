from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from transformers import AutoModelForCausalLM


TARGET_K = 16
BASE_MODEL_ID = "Qwen/Qwen3-0.6B"
OUTPUT_ROOT = Path("analysis_v2/final_k16_heatmaps")
TARGET_MODULES = {
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
}

RUNS = [
    ("meta_math", "lora", "logs_v2_unseeded/meta_math_lora_r16/adapter"),
    ("meta_math", "adalora", "logs_v2_unseeded/meta_math_adalora_init24_target16/adapter"),
    ("meta_math", "l1ra", "logs_v2_unseeded/meta_math_l1ra_r24_lam0.001/adapter"),
    ("open_orca", "lora", "logs_v2_unseeded/open_orca_lora_r16/adapter"),
    ("open_orca", "adalora", "logs_v2_unseeded/open_orca_adalora_init24_target16/adapter"),
    ("open_orca", "l1ra", "logs_v2_unseeded/open_orca_l1ra_r24_lam0.001/adapter"),
]

MODULE_ORDER = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

KEY_PATTERNS = [
    re.compile(r"layers\.(?P<layer>\d+)\.(?P<module>\S+?)\.(?P<comp>lora_[ABc]|lora_E)\.weight$"),
    re.compile(r"layers\.(?P<layer>\d+)\.(?P<module>\S+?)\.(?P<comp>lora_[ABE])$"),
    re.compile(r"layers\.(?P<layer>\d+)\.(?P<module>\S+?)\.(?P<comp>lora_[ABc])\.default$"),
]


def load_state_dict(adapter_dir: Path) -> dict[str, torch.Tensor]:
    st_path = adapter_dir / "adapter_model.safetensors"
    pt_path = adapter_dir / "adapter_model.pt"
    if st_path.exists():
        from safetensors.torch import load_file

        return load_file(str(st_path), device="cpu")
    if pt_path.exists():
        return torch.load(pt_path, map_location="cpu", weights_only=False)
    raise FileNotFoundError(f"Веса адаптера не найдены: {adapter_dir}")


def parse_key(key: str):
    for pattern in KEY_PATTERNS:
        match = pattern.search(key)
        if match:
            return int(match.group("layer")), match.group("module"), match.group("comp")
    return None


def collect_components(state_dict: dict[str, torch.Tensor]) -> list[dict]:
    groups: dict[tuple[int, str], dict[str, torch.Tensor]] = {}
    for key, tensor in state_dict.items():
        parsed = parse_key(key)
        if parsed is None:
            continue
        layer, module, comp = parsed
        groups.setdefault((layer, module), {})[comp] = tensor.float()

    components = []
    for (layer, module), slot in sorted(groups.items()):
        if "lora_A" not in slot or "lora_B" not in slot:
            continue
        gate = None
        if "lora_E" in slot:
            gate = slot["lora_E"].flatten()
        elif "lora_c" in slot:
            gate = slot["lora_c"].flatten()
        components.append(
            {
                "layer_id": layer,
                "module_name": module,
                "module_short": module.split(".")[-1],
                "A": slot["lora_A"],
                "B": slot["lora_B"],
                "gate": gate,
            }
        )
    return components


def find_threshold_for_mean_k(components: list[dict], target_k: float, max_iter: int = 80):
    gates = [c["gate"].abs() for c in components if c["gate"] is not None and c["gate"].numel() > 0]
    if not gates:
        return None

    g_max = max(g.max().item() for g in gates)
    lo, hi = 0.0, g_max * 1.0001
    best = {"threshold": 0.0, "mean_k": None, "gap": float("inf")}
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        counts = [int((g >= mid).sum().item()) for g in gates]
        mean_k = float(np.mean(counts))
        gap = abs(mean_k - target_k)
        if gap < best["gap"]:
            best = {"threshold": mid, "mean_k": mean_k, "gap": gap}
        if mean_k > target_k:
            lo = mid
        else:
            hi = mid
    return best


def lowrank_singular_values(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Сингулярные числа произведения left @ right."""
    rank = min(left.shape[1], right.shape[0])
    if rank == 0:
        return torch.empty(0)

    _, r_l = torch.linalg.qr(left)
    _, r_r = torch.linalg.qr(right.T)
    core = r_l @ r_r.T
    return torch.linalg.svdvals(core)


def lowrank_fro_norm(left: torch.Tensor | None, right: torch.Tensor | None) -> float:
    """Норма Фробениуса для low-rank произведения."""
    if left is None or right is None:
        return 0.0
    left_gram = left.T @ left
    right_gram = right @ right.T
    fro_sq = torch.sum(left_gram * right_gram.T).clamp_min(0.0)
    return float(torch.sqrt(fro_sq).item())


def component_factors(comp: dict, method: str, threshold: float | None):
    A, B, gate = comp["A"], comp["B"], comp["gate"]

    if method == "lora":
        return B, A, int(A.shape[0])

    if gate is None:
        if method == "adalora":
            return B, A, int(A.shape[0])
        if method == "l1ra":
            return A, B, int(A.shape[1])

    if gate is not None and gate.numel() == 0:
        return None, None, 0

    mask = gate.abs() >= (threshold if threshold is not None else 0.0)
    active_rank = int(mask.sum().item())
    if active_rank == 0:
        return None, None, 0

    gate_masked = gate.clone()
    gate_masked[~mask] = 0.0

    if method == "adalora":
        return B * gate_masked.view(1, -1), A, active_rank

    if method == "l1ra":
        return A * gate_masked.view(1, -1), B, active_rank

    raise ValueError(f"Unknown method: {method}")


def energy_rank_from_factors(left: torch.Tensor | None, right: torch.Tensor | None, threshold: float):
    if left is None or right is None:
        return 0
    sigma = lowrank_singular_values(left, right)
    if sigma.numel() == 0:
        return 0
    energy = sigma.pow(2)
    total = energy.sum()
    if total <= 1e-12:
        return 0
    cumulative = torch.cumsum(energy, dim=0) / total
    return int((cumulative >= threshold).nonzero(as_tuple=True)[0][0].item()) + 1


def load_base_weight_norms() -> dict[tuple[int, str], float]:
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    norms: dict[tuple[int, str], float] = {}
    pattern = re.compile(r"model\.layers\.(?P<layer>\d+)\.(?P<module>\S+)\.weight$")
    for key, tensor in model.state_dict().items():
        match = pattern.match(key)
        if not match:
            continue
        module_name = match.group("module")
        if module_name not in TARGET_MODULES:
            continue
        norms[(int(match.group("layer")), module_name)] = float(tensor.float().norm().item())
    del model
    return norms


def save_heatmap(
    rows: list[dict],
    dataset: str,
    method: str,
    metric: str,
    title_metric: str,
    fmt: str = ".0f",
    cmap: str = "YlOrRd",
    vmax: float | None = None,
):
    df = pd.DataFrame(rows)
    pivot = df.pivot_table(index="module_short", columns="layer_id", values=metric, aggfunc="first")
    pivot = pivot.reindex([m for m in MODULE_ORDER if m in pivot.index])
    if vmax is None:
        vmax = float(pivot.max().max())

    fig, ax = plt.subplots(figsize=(14, 3.5))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        vmin=0,
        vmax=vmax,
        linewidths=0.5,
        ax=ax,
        annot_kws={"size": 7},
    )
    ax.set_title(f"{method.upper()} / {dataset} / final - {title_metric} (mean={pivot.to_numpy().mean():.2f})")
    ax.set_xlabel("Layer")
    ax.set_ylabel("")
    fig.tight_layout()

    out = OUTPUT_ROOT / metric / dataset / f"{method}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"Loading base weights for deltaW/W: {BASE_MODEL_ID}")
    base_weight_norms = load_base_weight_norms()
    print(f"Indexed base target weights: {len(base_weight_norms)}")

    all_rows = []
    summary_rows = []

    for dataset, method, adapter_path in RUNS:
        adapter_dir = Path(adapter_path)
        print(f"Processing {dataset}/{method}: {adapter_dir}")
        state = load_state_dict(adapter_dir)
        components = collect_components(state)
        if not components:
            raise RuntimeError(f"No adapter components found in {adapter_dir}")

        threshold_info = find_threshold_for_mean_k(components, TARGET_K) if method in {"adalora", "l1ra"} else None
        tau = threshold_info["threshold"] if threshold_info else None

        rows = []
        for comp in components:
            left, right, active_rank = component_factors(comp, method, tau)
            er95 = energy_rank_from_factors(left, right, 0.95)
            er99 = energy_rank_from_factors(left, right, 0.99)
            delta_norm = lowrank_fro_norm(left, right)
            base_norm = base_weight_norms.get((comp["layer_id"], comp["module_name"]), 0.0)
            delta_ratio = delta_norm / base_norm if base_norm > 1e-12 else 0.0
            row = {
                "dataset": dataset,
                "method": method,
                "layer_id": comp["layer_id"],
                "module_name": comp["module_name"],
                "module_short": comp["module_short"],
                "active_rank": active_rank,
                "energy_rank_95": er95,
                "energy_rank_99": er99,
                "deltaW_fro": delta_norm,
                "W_fro": base_norm,
                "deltaW_over_W_fro": delta_ratio,
                "tau": tau if tau is not None else np.nan,
                "mean_k_target": TARGET_K,
            }
            rows.append(row)
            all_rows.append(row)

        er95_path = save_heatmap(rows, dataset, method, "energy_rank_95", "energy_rank_95", vmax=TARGET_K)
        er99_path = save_heatmap(rows, dataset, method, "energy_rank_99", "energy_rank_99", vmax=TARGET_K)
        active_path = save_heatmap(rows, dataset, method, "active_rank", "active_rank", vmax=TARGET_K)
        delta_path = save_heatmap(
            rows,
            dataset,
            method,
            "deltaW_over_W_fro",
            "||ΔW||_F / ||W||_F",
            fmt=".4f",
            cmap="YlOrRd",
        )

        summary_rows.append(
            {
                "dataset": dataset,
                "method": method,
                "tau": tau if tau is not None else np.nan,
                "mean_active_rank": float(np.mean([r["active_rank"] for r in rows])),
                "mean_energy_rank_95": float(np.mean([r["energy_rank_95"] for r in rows])),
                "mean_energy_rank_99": float(np.mean([r["energy_rank_99"] for r in rows])),
                "mean_deltaW_over_W_fro": float(np.mean([r["deltaW_over_W_fro"] for r in rows])),
                "energy_rank_95_path": str(er95_path),
                "energy_rank_99_path": str(er99_path),
                "active_rank_path": str(active_path),
                "deltaW_over_W_fro_path": str(delta_path),
            }
        )

    pd.DataFrame(all_rows).to_csv(OUTPUT_ROOT / "final_k16_heatmap_values.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(OUTPUT_ROOT / "summary.csv", index=False)
    print(f"Saved values: {OUTPUT_ROOT / 'final_k16_heatmap_values.csv'}")
    print(f"Saved summary: {OUTPUT_ROOT / 'summary.csv'}")


if __name__ == "__main__":
    main()
