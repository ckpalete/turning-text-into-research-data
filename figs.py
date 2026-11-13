"""Figures for the workshop. Writes PNG + PDF into ./figures/."""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sim

OUT = "figures"
os.makedirs(OUT, exist_ok=True)

# palette (validated: dataviz reference instance, slots 1-3, light surface)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE = "#fcfcfb"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK2,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.titlecolor": INK,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "savefig.facecolor": SURFACE,
    "savefig.bbox": "tight",
    "savefig.dpi": 200,
})

EST_LABEL = {
    "naive": "Naive\n(LLM label, n=4,000)",
    "labeled_only": "Hand-labeled only\n(n=200)",
    "rectified": "Rectified\n(both)",
}
EST_COLOR = {"naive": ORANGE, "labeled_only": MUTED, "rectified": BLUE}


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/{name}.{ext}")
    plt.close(fig)
    print(f"  wrote {OUT}/{name}.png")


# ------------------------------------------------------------------ figure 1
def fig_headline():
    _, _, tab = sim.headline()
    order = ["naive", "labeled_only", "rectified"]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    fig.subplots_adjust(left=0.34, top=0.80, bottom=0.16, right=0.98)

    ypos, ylabels, seen = [], [], set()
    row = 0
    group_mid = {}
    for prompt in ["A (broad)", "B (strict)"]:
        start = row
        for est in order:
            r = tab[(tab.prompt == prompt) & (tab.estimator == est)].iloc[0]
            c = EST_COLOR[est]
            ax.plot([r.lo, r.hi], [row, row], color=c, lw=2.0,
                    solid_capstyle="round", zorder=2)
            ax.plot([r.beta], [row], "o", ms=9, color=c, mec=SURFACE, mew=1.5,
                    zorder=3,
                    label=EST_LABEL[est].replace("\n", " ") if est not in seen else None)
            seen.add(est)
            ax.annotate(f"{r.beta:.2f}", (r.beta, row), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=9.5,
                        color=INK, fontweight="bold")
            ypos.append(row); ylabels.append(EST_LABEL[est])
            row += 1
        group_mid[prompt] = (start + row - 1) / 2
        row += 0.6

    lo, hi = tab.lo.min(), tab.hi.max()
    pad = 0.12 * (hi - lo)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(row - 0.9, -0.9)

    ax.axvline(sim.BETA_TRUE, color=INK, lw=1.4, ls=(0, (4, 3)), zorder=1)
    ax.annotate("true $\\beta$ = 2.0", (sim.BETA_TRUE, -0.85),
                textcoords="offset points", xytext=(7, 0), fontsize=10,
                color=INK, fontweight="bold", va="top", ha="left")

    ax.set_yticks(ypos); ax.set_yticklabels(ylabels, fontsize=9.5, color=INK2)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("estimated effect of the concept on the outcome (bps)")
    ax.xaxis.grid(True); ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)

    for prompt, lab in (("A (broad)", "Prompt A\n“broad”"),
                        ("B (strict)", "Prompt B\n“strict”")):
        ax.annotate(lab, (-0.33, group_mid[prompt]),
                    xycoords=("axes fraction", "data"),
                    fontsize=11, fontweight="bold", color=INK,
                    ha="center", va="center")

    ax.set_title("Same corpus, same concept — two prompts", pad=34, loc="left",
                 x=-0.33)
    ax.legend(loc="lower left", bbox_to_anchor=(-0.33, 1.015), ncol=3,
              frameon=False, fontsize=9.5, labelcolor=INK2,
              handletextpad=0.4, columnspacing=1.6)
    save(fig, "01_headline")


# ------------------------------------------------------------------ figure 2
def fig_quality():
    q = sim.quality_sweep(reps=400)
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    x = q.label_accuracy * 100

    ax.plot(x, q.width_labeled_only, "-o", color=MUTED, lw=2.0, ms=8,
            mec=SURFACE, mew=1.5, label="Hand-labeled only (n=200)")
    ax.plot(x, q.width_rectified, "-o", color=BLUE, lw=2.0, ms=8,
            mec=SURFACE, mew=1.5, label="Rectified (LLM + 200 hand labels)")

    # crossover
    d = q.width_rectified.values - q.width_labeled_only.values
    i = np.where(np.sign(d[:-1]) != np.sign(d[1:]))[0]
    if len(i):
        i = i[0]
        t = d[i] / (d[i] - d[i + 1])
        xc = x.values[i] + t * (x.values[i + 1] - x.values[i])
        ax.axvline(xc, color=INK, lw=1.2, ls=(0, (4, 3)), zorder=1)
        ax.annotate(f"break-even\n≈ {xc:.0f}% accurate", (xc, ax.get_ylim()[1] * 0.93),
                    textcoords="offset points", xytext=(8, 0), fontsize=9.5,
                    color=INK, fontweight="bold", va="top")

    ax.set_xlabel("LLM label accuracy (sensitivity = specificity, %)")
    ax.set_ylabel("width of the 95% confidence interval")
    ax.set_title("A bad label is worse than no label")
    ax.yaxis.grid(True); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=10, labelcolor=INK2)
    save(fig, "02_label_quality")
    return q


# ------------------------------------------------------------------ figure 3
def fig_coverage():
    a = sim.monte_carlo(reps=800)
    b = sim.monte_carlo(reps=800, differential=True, seed=99)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), sharey=True)
    order = ["naive", "labeled_only", "rectified"]

    for ax, tab, title in ((axes[0], a, "LLM errors unrelated to the outcome"),
                           (axes[1], b, "LLM errors correlated with the outcome")):
        g = tab.groupby("estimator").coverage.mean().reindex(order)
        xs = np.arange(len(order))
        ax.bar(xs, g.values, width=0.6, color=[EST_COLOR[e] for e in order],
               zorder=2)
        for xi, v in zip(xs, g.values):
            ax.annotate(f"{v*100:.0f}%", (xi, v), textcoords="offset points",
                        xytext=(0, 5), ha="center", fontsize=10.5,
                        fontweight="bold", color=INK)
        ax.axhline(0.95, color=INK, lw=1.2, ls=(0, (4, 3)), zorder=3)
        ax.set_xticks(xs)
        ax.set_xticklabels([EST_LABEL[e].split("\n")[0] for e in order],
                           fontsize=10, color=INK2)
        ax.set_title(title, fontsize=11.5)
        ax.yaxis.grid(True); ax.set_axisbelow(True)
        ax.set_ylim(0, 1.12)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0", "25%", "50%", "75%", "100%"])
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    axes[0].set_ylabel("share of 95% CIs that contain the truth")
    axes[0].annotate("nominal 95%", (0.02, 0.95), xycoords=("axes fraction", "data"),
                     textcoords="offset points", xytext=(0, 5), fontsize=9,
                     color=INK2)
    fig.suptitle("Does the confidence interval mean what it says?",
                 fontsize=13, fontweight="bold", color=INK, y=1.02)
    save(fig, "03_coverage")
    return a, b


if __name__ == "__main__":
    print("building figures...")
    fig_headline()
    q = fig_quality()
    a, b = fig_coverage()
    pd.set_option("display.float_format", lambda v: f"{v:9.3f}")
    print("\nlabel-quality sweep:\n", q.to_string(index=False))
    print("\ncoverage (nondifferential):\n", a.to_string(index=False))
    print("\ncoverage (differential):\n", b.to_string(index=False))
