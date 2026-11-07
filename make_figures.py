"""Draw the figures in figures/ from results/rejection_rates.csv.

Usage:  python make_figures.py      (run run_simulation.py first)
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))
from dgp import simulate  # noqa: E402
from inference import slope_tstats  # noqa: E402

# Palette: two categorical slots plus a single-hue sequential ramp
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e6e5e0"
BLUE, ORANGE = "#2a78d6", "#eb6834"
SEQ = LinearSegmentedColormap.from_list(
    "blues", ["#f3f7fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
})

FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
df = pd.read_csv(ROOT / "results" / "rejection_rates.csv")


def heatmap(T=360):
    sub = df[df["T"] == T].pivot(index="rho", columns="R2_star", values="reject_ols")
    sub = sub.sort_index(ascending=False)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    im = ax.imshow(sub.values, cmap=SEQ, vmin=0, vmax=0.5, aspect="auto")
    for i in range(sub.shape[0]):
        for j in range(sub.shape[1]):
            v = sub.values[i, j]
            ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=9,
                    color="#ffffff" if v > 0.22 else INK)
    ax.set_xticks(range(sub.shape[1]), [f"{c:g}" for c in sub.columns])
    ax.set_yticks(range(sub.shape[0]), [f"{r:g}" for r in sub.index])
    ax.set_xlabel("R²* — share of return variance explained by the true expected return")
    ax.set_ylabel("Persistence ρ = ρ*")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title(f"How often an irrelevant predictor looks significant (OLS, 5% test, T = {T})")
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, format=lambda x, _: f"{x:.0%}")
    cb.outline.set_visible(False)
    cb.ax.tick_params(color=MUTED, labelcolor=INK2)
    fig.text(0.01, 0.01, "Nominal rejection rate is 5%. 10,000 simulated samples per cell.",
             color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIG / "rejection_heatmap.png", dpi=200)
    plt.close(fig)


def by_sample_size(rho=0.99, r2=0.10):
    sub = df[(df["rho"] == rho) & (df["R2_star"] == r2)].sort_values("T")
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.axhline(0.05, color=MUTED, lw=1, ls=(0, (4, 3)))
    ax.text(sub["T"].max(), 0.058, "nominal 5%", color=MUTED, fontsize=8, ha="right")
    for col, color, label in [("reject_ols", BLUE, "OLS standard errors"),
                              ("reject_nw", ORANGE, "Newey-West standard errors")]:
        ax.plot(sub["T"], sub[col], color=color, lw=2, marker="o", ms=7,
                mec=SURFACE, mew=2, label=label)
        last = sub.iloc[-1]
        ax.annotate(f"{last[col]:.0%}", (last["T"], last[col]), xytext=(8, 0),
                    textcoords="offset points", va="center", color=INK2, fontsize=9)
    ax.set_xticks(sub["T"], [f"{t}\n({t // 12} yrs monthly)" for t in sub["T"]])
    ax.set_ylim(0, max(0.6, sub[["reject_ols", "reject_nw"]].values.max() + 0.05))
    ax.yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("Sample length T")
    ax.set_ylabel("Rejection rate of H0: slope = 0")
    ax.set_title(f"More data makes it worse (ρ = {rho}, R²* = {r2})")
    ax.legend(frameon=False, loc="upper left", labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(FIG / "rejection_by_sample_size.png", dpi=200)
    plt.close(fig)


def tstat_distribution(rho=0.99, r2=0.10, T=360, n_paths=20_000, seed=7):
    rng = np.random.default_rng(seed)
    r_next, z = simulate(rng, n_paths, T, rho, rho, r2)
    _, t_ols, _ = slope_tstats(r_next, z)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    bins = np.linspace(-8, 8, 81)
    ax.hist(t_ols, bins=bins, density=True, color=BLUE, alpha=0.85,
            edgecolor=SURFACE, linewidth=0.6, label="Simulated t-statistics")
    grid = np.linspace(-8, 8, 400)
    ax.plot(grid, np.exp(-grid**2 / 2) / np.sqrt(2 * np.pi), color=ORANGE, lw=2,
            label="N(0, 1): what the textbook test assumes")
    for c in (-1.96, 1.96):
        ax.axvline(c, color=MUTED, lw=1, ls=(0, (4, 3)))
    share = np.mean(np.abs(t_ols) > 1.96)
    ax.text(0.98, 0.95, f"{share:.0%} of |t| > 1.96\n(should be 5%)", transform=ax.transAxes,
            ha="right", va="top", color=INK2, fontsize=9)
    ax.set_ylim(0, 0.5)
    ax.set_xlabel("t-statistic of the slope on an irrelevant predictor")
    ax.set_ylabel("Density")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_title(f"The t-statistic is far too dispersed (ρ = {rho}, R²* = {r2}, T = {T})")
    ax.legend(frameon=False, loc="upper left", labelcolor=INK2, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "tstat_distribution.png", dpi=200)
    plt.close(fig)


def data_mining():
    path = ROOT / "results" / "data_mining.csv"
    if not path.exists():
        print("Skipping data-mining figure: run run_data_mining.py first")
        return
    dm = pd.read_csv(path)
    colors = [BLUE, ORANGE, "#1baf7a"]
    markers = ["o", "s", "D"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    panels = [
        (axes[0], "reject_ols", "False-positive rate of the best of K", lambda x, _: f"{x:.0%}"),
        (axes[1], "crit95_ols", "|t| needed for a true 5% test", lambda x, _: f"{x:.1f}"),
    ]
    for ax, col, title, fmt in panels:
        ref = 0.05 if col == "reject_ols" else 1.96
        ax.axhline(ref, color=MUTED, lw=1, ls=(0, (4, 3)))
        ax.text(140, ref, "nominal 5%" if col == "reject_ols" else "1.96", color=MUTED,
                fontsize=8, va="bottom", ha="right")
        for (label, sub), color, mk in zip(dm.groupby("scenario", sort=False), colors, markers):
            ax.plot(sub["K"], sub[col], color=color, lw=2, marker=mk, ms=7, mec=SURFACE,
                    mew=2, label=label)
            if col == "crit95_ols":  # left-panel lines all end near 100%, labels would collide
                last = sub.iloc[-1]
                ax.annotate(f"{last[col]:.1f}", (last["K"], last[col]), xytext=(8, 0),
                            textcoords="offset points", va="center", color=INK2, fontsize=9)
        ax.set_xscale("log")
        ax.set_xticks(dm["K"].unique(), [str(k) for k in dm["K"].unique()])
        ax.minorticks_off()
        ax.set_xlim(0.8, 150)
        ax.yaxis.set_major_formatter(fmt)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.set_xlabel("K = number of irrelevant predictors tried")
        ax.set_title(title)
    axes[0].set_ylim(0, 1.05)
    axes[1].set_ylim(0, 8)
    axes[0].set_ylabel("Share of samples with best |t| > 1.96")
    axes[1].set_ylabel("95th percentile of best |t|")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="lower center", ncol=3, labelcolor=INK2,
               fontsize=9.5)
    fig.suptitle("Data mining amplifies spurious regression (OLS, T = 360)", x=0.01, ha="left",
                 fontweight="bold", fontsize=13)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(FIG / "data_mining.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    heatmap()
    by_sample_size()
    tstat_distribution()
    data_mining()
    print("Saved figures to", FIG)
