from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RUN_NAME = "open_orca_l1ra_r24_lam0.001"
DATASET = "open_orca"
METHOD = "l1ra"
METRIC = "active_rank_cal16"

CHECKPOINT_CSV = Path("analysis_v2/energy_rank_per_checkpoint.csv")
STAB_CSV = Path("analysis_v2/energy_rank_stab_spearman.csv")
METRICS_CSV = Path("logs_v2_unseeded/open_orca_l1ra_r24_lam0.001/metrics.csv")
OUT_DIR = Path("analysis_v2/open_orca_l1ra_stability")


def spearman_rho(a: np.ndarray, b: np.ndarray) -> float:
    if a.size != b.size or a.size < 3:
        return float("nan")
    if np.nanstd(a) == 0 or np.nanstd(b) == 0:
        return float("nan")
    ar = pd.Series(a).rank(method="average").to_numpy()
    br = pd.Series(b).rank(method="average").to_numpy()
    return float(np.corrcoef(ar, br)[0, 1])


def load_rank_vectors() -> dict[int, np.ndarray]:
    df = pd.read_csv(CHECKPOINT_CSV)
    df = df[
        (df["run_name"] == RUN_NAME)
        & (df["dataset"] == DATASET)
        & (df["method"] == METHOD)
    ].copy()
    df = df.sort_values(["step", "layer", "module"])

    vectors: dict[int, np.ndarray] = {}
    for step, group in df.groupby("step"):
        vectors[int(step)] = group[METRIC].to_numpy(dtype=float)
    return vectors


def load_train_loss() -> pd.DataFrame:
    metrics = pd.read_csv(METRICS_CSV)
    train = metrics[
        (metrics["split"] == "train")
        & metrics["time_sec"].isna()
        & metrics["step"].notna()
        & metrics["loss"].notna()
    ].copy()
    train["step"] = train["step"].astype(int)
    train = train.sort_values("step")
    train["loss_ma50"] = train["loss"].rolling(50, min_periods=1).mean()
    return train


def stable_from_threshold(curve: pd.DataFrame, value_col: str, threshold: float) -> int | None:
    ok_steps = curve[curve[value_col] >= threshold]["step"].astype(int).tolist()
    for step in ok_steps:
        tail = curve[curve["step"] >= step]
        if not tail.empty and bool((tail[value_col] >= threshold).all()):
            return step
    return None


def compute_window_curves(vectors: dict[int, np.ndarray], windows=(100, 200, 400)) -> pd.DataFrame:
    rows = []
    steps = sorted(vectors)
    for window in windows:
        for step in steps:
            prev = step - window
            if prev not in vectors:
                continue
            rows.append(
                {
                    "step": step,
                    "window": window,
                    "rho": spearman_rho(vectors[step], vectors[prev]),
                }
            )
    return pd.DataFrame(rows)


def load_final_curve() -> pd.DataFrame:
    stab = pd.read_csv(STAB_CSV)
    return stab[
        (stab["run_name"] == RUN_NAME)
        & (stab["dataset"] == DATASET)
        & (stab["method"] == METHOD)
        & (stab["metric"] == METRIC)
    ][["step", "rho_vs_final"]].copy()


def plot_base(ax, ax2, train: pd.DataFrame, title: str, right_ylabel: str):
    ax.plot(train["step"], train["loss"], color="0.7", lw=0.6, alpha=0.75, label="train_loss")
    ax.plot(train["step"], train["loss_ma50"], color="black", lw=1.6, label="train_loss (MA50)")
    ax.set_xlabel("step")
    ax.set_ylabel("train_loss")
    ax.grid(alpha=0.25)
    ax.set_title(title)
    ax2.set_ylabel(right_ylabel, color="tab:green")
    ax2.set_ylim(0, 1.05)
    ax2.axhline(0.95, ls="--", color="gray", lw=0.8, alpha=0.65)


def merged_legend(ax, ax2):
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=9, loc="center right")


def plot_window_stability(train: pd.DataFrame, window_df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax2 = ax.twinx()
    plot_base(
        ax,
        ax2,
        train,
        "open_orca - l1ra\n(metric: active_rank_cal16)",
        r"$\rho(t, t-W)$",
    )

    colors = {100: "tab:orange", 200: "tab:green", 400: "tab:red"}
    for window, group in window_df.groupby("window"):
        group = group.sort_values("step")
        ax2.plot(
            group["step"],
            group["rho"],
            marker="o",
            ms=3,
            lw=1.3,
            color=colors.get(int(window)),
            label=fr"$\rho(t, t-{int(window)})$",
        )

    conv = None
    w200 = window_df[window_df["window"] == 200].sort_values("step")
    if not w200.empty:
        conv = stable_from_threshold(w200.rename(columns={"rho": "rho_w200"}), "rho_w200", 0.95)
    if conv is not None:
        ax.axvline(conv, color="tab:green", ls=":", lw=1.2)
        ymax = ax.get_ylim()[1]
        ax.text(conv + 25, ymax * 0.78, f"conv@{conv}\n({conv / int(train['step'].max()):.0%})", color="tab:green")

    merged_legend(ax, ax2)
    fig.tight_layout()
    out = OUT_DIR / "open_orca_l1ra_window_stability_active_rank_cal16.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_final_stability(train: pd.DataFrame, final_df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax2 = ax.twinx()
    plot_base(
        ax,
        ax2,
        train,
        "open_orca - l1ra\n(metric: active_rank_cal16 vs final)",
        r"$\rho(t, final)$",
    )
    final_df = final_df.sort_values("step")
    ax2.plot(
        final_df["step"],
        final_df["rho_vs_final"],
        marker="o",
        ms=3,
        lw=1.5,
        color="tab:blue",
        label=r"$\rho(t, final)$",
    )

    conv = stable_from_threshold(final_df, "rho_vs_final", 0.9)
    if conv is not None:
        ax.axvline(conv, color="tab:blue", ls=":", lw=1.2)
        ymax = ax.get_ylim()[1]
        ax.text(conv + 25, ymax * 0.78, f"stable@{conv}\n({conv / int(train['step'].max()):.0%})", color="tab:blue")

    ax2.axhline(0.9, ls="--", color="tab:blue", lw=0.8, alpha=0.45)
    merged_legend(ax, ax2)
    fig.tight_layout()
    out = OUT_DIR / "open_orca_l1ra_vs_final_active_rank_cal16.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    vectors = load_rank_vectors()
    train = load_train_loss()
    window_df = compute_window_curves(vectors)
    final_df = load_final_curve()

    window_df.to_csv(OUT_DIR / "open_orca_l1ra_window_stability_active_rank_cal16.csv", index=False)
    final_df.to_csv(OUT_DIR / "open_orca_l1ra_vs_final_active_rank_cal16.csv", index=False)

    p1 = plot_window_stability(train, window_df)
    p2 = plot_final_stability(train, final_df)
    print(f"Saved: {p1}")
    print(f"Saved: {p2}")


if __name__ == "__main__":
    main()
