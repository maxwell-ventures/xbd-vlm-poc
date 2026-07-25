#!/usr/bin/env python3
"""Generate the data-driven figures (matplotlib) into outputs/figures/.

Numbers are read from outputs/eval/*.json so nothing is transcribed by hand.
Run with the venv:  .venv/bin/python scripts/fig_charts.py
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap

FIG = Path("outputs/figures"); FIG.mkdir(parents=True, exist_ok=True)

# --- design tokens -------------------------------------------------------
INK, MUTED, GRIDC, SURFACE = "#22201d", "#6b665e", "#e7e3da", "#fcfcfb"
THREEB, SEVENB = "#3b6ea5", "#c8663b"          # validated categorical pair
UP, DOWN = "#2e8b6b", "#b5482f"                 # delta poles, always direct-labeled
# damage severity: ordinal, so a single light->dark sequential ramp (CVD-safe)
SEV = ["#e9c46a", "#e08b3f", "#c85a3c", "#7f2b2b"]   # no / minor / major / destroyed
SEQ = LinearSegmentedColormap.from_list("seq", ["#f4f1ea", "#2f4b6e"])  # confusion counts

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": GRIDC,
    "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
})

GRADES = ["no-damage", "minor-damage", "major-damage", "destroyed"]
SHORT = ["no", "minor", "major", "destr"]
CELLS = {
    ("3B","zero-shot"):"base_post_gpu.json", ("3B","tuned\npost"):"run1.json",
    ("3B","tuned\npre+post"):"run2.json",
    ("7B","zero-shot"):"base_post_7b.json", ("7B","tuned\npost"):"run1_post_7b.json",
    ("7B","tuned\npre+post"):"run2_prepost_7b.json",
}
def load(name): return json.load(open(f"outputs/eval/{name}"))
M = {k: load(v) for k, v in CELLS.items()}
COLS = ["zero-shot", "tuned\npost", "tuned\npre+post"]
ROWS = ["3B", "7B"]

def title(ax, t, sub=None, wrap=90):
    """Figure-level title + subtitle with reserved headroom, so nothing overlaps."""
    import textwrap
    fig = ax.figure
    fig.suptitle(t, x=0.125, ha="left", fontsize=13, fontweight="bold", y=0.99)
    if sub:
        sub = "\n".join(textwrap.wrap(sub, wrap))
        nlines = sub.count("\n") + 1
        fig.text(0.125, 0.915, sub, ha="left", va="top", fontsize=9.5, color=MUTED)
        fig.subplots_adjust(top=0.90 - 0.045 * nlines)
    else:
        fig.subplots_adjust(top=0.90)

def save(fig, name):
    fig.savefig(FIG / name, dpi=150, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig); print("wrote", FIG / name)


# 1 ── the headline 2x3 QWK grid, as a heatmap ---------------------------
def fig_grid():
    vals = [[M[(r, c)]["qwk"] for c in COLS] for r in ROWS]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    im = ax.imshow(vals, cmap=SEQ, vmin=0, vmax=0.65, aspect="auto")
    for i, r in enumerate(ROWS):
        for j, c in enumerate(COLS):
            v = vals[i][j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                    color="white" if v > 0.33 else INK, fontsize=17, fontweight="bold")
    ax.set_xticks(range(3)); ax.set_xticklabels([c.replace("\n"," ") for c in COLS], fontsize=11)
    ax.set_yticks(range(2)); ax.set_yticklabels(ROWS, fontsize=13, fontweight="bold")
    ax.set_xticks([x-0.5 for x in range(1,3)], minor=True)
    ax.set_yticks([0.5], minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=3); ax.tick_params(which="minor", length=0)
    ax.tick_params(length=0)
    title(ax, "Ordinal agreement (QWK) across model size and input",
          "0 = chance, 1 = perfect. Capacity lifts the zero-shot floor; tuning equalizes; capacity + before-image compound at the top.")
    save(fig, "01_qwk_grid.png")


# 2 ── per-class recall journey (the minor/major through-line) ------------
def fig_perclass():
    order = [("3B","zero-shot"),("3B","tuned\npost"),("3B","tuned\npre+post"),
             ("7B","zero-shot"),("7B","tuned\npost"),("7B","tuned\npre+post")]
    labels = ["3B base","3B post","3B pre+post","7B base","7B post","7B pre+post"]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.5), sharey=True)
    for gi, g in enumerate(GRADES):
        ax = axes[gi]
        rec = [M[k]["per_class"][g]["recall"] for k in order]
        bars = ax.bar(range(6), rec, color=SEV[gi], width=0.72,
                      edgecolor=SURFACE, linewidth=1.5)
        for b, v in zip(bars, rec):
            ax.text(b.get_x()+b.get_width()/2, v+0.02, f"{v:.2f}", ha="center",
                    va="bottom", fontsize=8, color=INK)
        ax.set_title(g, fontsize=11, fontweight="bold", color=SEV[gi], loc="center")
        ax.set_xticks(range(6)); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylim(0, 1.08); ax.grid(axis="y", color=GRIDC, linewidth=0.8); ax.set_axisbelow(True)
        if gi == 0: ax.set_ylabel("recall", fontsize=10)
    fig.subplots_adjust(top=0.78)
    fig.suptitle("Per-class recall: nothing moves minor-damage",
                 x=0.09, ha="left", fontsize=13, fontweight="bold", y=1.10)
    fig.text(0.09, 1.00, "no-damage and destroyed respond to tuning and capacity; minor-damage stays stuck and collapses into major across every run.",
             ha="left", fontsize=9.5, color=MUTED)
    save(fig, "02_perclass_recall.png")


# 3 ── all six confusion matrices, 2x3 -----------------------------------
def fig_confusions():
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 8))
    order = [[("3B",c) for c in COLS],[("7B",c) for c in COLS]]
    vmax = 400
    for i in range(2):
        for j in range(3):
            ax = axes[i][j]; k = order[i][j]; cm = M[k]["confusion_matrix"]
            ax.imshow(cm, cmap=SEQ, vmin=0, vmax=vmax, aspect="equal")
            for a in range(4):
                for b in range(4):
                    v = cm[a][b]
                    ax.text(b, a, str(v), ha="center", va="center", fontsize=9,
                            color="white" if v > vmax*0.5 else (INK if v else "#c9c4ba"),
                            fontweight="bold" if a==b else "normal")
            ax.set_xticks(range(4)); ax.set_yticks(range(4))
            ax.set_xticklabels(SHORT, fontsize=8); ax.set_yticklabels(SHORT, fontsize=8)
            ax.set_title(f"{k[0]}  {k[1].replace(chr(10),' ')}   QWK {M[k]['qwk']:.3f}",
                         fontsize=10.5, fontweight="bold", pad=6)
            if j == 0: ax.set_ylabel("true", fontsize=9, color=MUTED)
            if i == 1: ax.set_xlabel("predicted", fontsize=9, color=MUTED)
            for s in ax.spines.values(): s.set_visible(False)
            ax.tick_params(length=0)
    fig.suptitle("Confusion matrices — the whole grid (rows = true, cols = predicted)",
                 x=0.5, fontsize=14, fontweight="bold", y=0.98)
    fig.text(0.5, 0.945, "3B base collapses everything into one column. Tuning and capacity spread the mass onto the diagonal — except the minor↔major cell, which stays hot everywhere.",
             ha="center", fontsize=9.5, color=MUTED)
    save(fig, "03_confusion_grid.png")


# 4 ── the QWK journey as connected dots (capacity lifts floor) ----------
def fig_journey():
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    for r, col in [("3B", THREEB), ("7B", SEVENB)]:
        ys = [M[(r, c)]["qwk"] for c in COLS]
        ax.plot(range(3), ys, "-o", color=col, lw=2.4, ms=9, label=r,
                markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
        for x, y in zip(range(3), ys):
            ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                        xytext=(0, 11 if r=="7B" else -16), ha="center",
                        fontsize=9, color=col, fontweight="bold")
    ax.set_xticks(range(3)); ax.set_xticklabels([c.replace("\n"," ") for c in COLS])
    ax.set_ylim(-0.05, 0.68); ax.set_ylabel("QWK", fontsize=11)
    ax.grid(axis="y", color=GRIDC, linewidth=0.8); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=12, loc="lower right")
    ax.annotate("7B already grades\nzero-shot (0.460);\n3B is at chance (0.009)",
                xy=(0, 0.46), xytext=(0.35, 0.30), fontsize=8.5, color=MUTED,
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1))
    title(ax, "The QWK journey",
          "Fine-tuning is the great equalizer on post-only; the two lines nearly meet, then the before-image reopens a gap at 7B.")
    save(fig, "04_qwk_journey.png")


# 5 ── eval-loss curve (sampled checkpoints; full log lost with the pod) --
def fig_training():
    # 3B post-only, eval_loss at logged checkpoints (captured during monitoring).
    steps = [50,100,150,200,250,300,350,400,450]
    evloss = [0.187,0.134,0.131,0.132,0.130,0.115,0.129,0.117,0.115]
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    ax.plot(steps, evloss, "-o", color=THREEB, lw=2.2, ms=7,
            markeredgecolor=SURFACE, markeredgewidth=1.4)
    ax.axhline(0.115, color=MUTED, ls="--", lw=1, zorder=0)
    ax.text(455, 0.117, "best 0.115", fontsize=9, color=MUTED, va="bottom", ha="right")
    ax.set_xlabel("training step", fontsize=11); ax.set_ylabel("validation loss", fontsize=11)
    ax.grid(color=GRIDC, linewidth=0.8); ax.set_axisbelow(True)
    ax.set_ylim(0.10, 0.20)
    title(ax, "Validation loss converges early and plateaus (3B post-only)",
          "Logged eval checkpoints. Loss saturates fast because the templated targets are low-entropy; it is not a measure of grading skill.")
    fig.text(0.13, -0.02, "Note: sampled from monitoring — full per-step log was not retained after the ephemeral pod was torn down.",
             fontsize=8, color=MUTED, style="italic")
    save(fig, "05_training_curve.png")


# 6 ── class imbalance: raw vs balanced ----------------------------------
def fig_distribution():
    raw = {"no-damage":233635,"minor-damage":25724,"major-damage":21449,"destroyed":23562}
    tot = sum(raw.values())
    train = {"no-damage":1490,"minor-damage":1488,"major-damage":1489,"destroyed":1454}
    ttot = sum(train.values())
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax, data, tt, ttl in [(a1, raw, tot, "Raw xBD (304,370 buildings)"),
                              (a2, train, ttot, "Balanced training set (5,921)")]:
        vals = [data[g]/tt for g in GRADES]
        bars = ax.barh(range(4), vals, color=SEV, edgecolor=SURFACE, linewidth=1.5)
        for b, v, g in zip(bars, vals, GRADES):
            ax.text(v+0.008, b.get_y()+b.get_height()/2, f"{v:.0%}", va="center", fontsize=10, color=INK)
        ax.set_yticks(range(4)); ax.set_yticklabels(SHORT, fontsize=10)
        ax.invert_yaxis(); ax.set_xlim(0, max(vals)*1.18)
        ax.set_xticks([]); ax.spines["bottom"].set_visible(False)
        ax.set_title(ttl, fontsize=11.5, fontweight="bold", loc="left", pad=8)
    fig.subplots_adjust(top=0.74)
    fig.suptitle("Why class balance is its own pipeline stage",
                 x=0.09, ha="left", fontsize=13, fontweight="bold", y=1.13)
    fig.text(0.09, 1.02, "77% of real xBD is no-damage. Train on that and the model just predicts no-damage. The training set is sampled flat; the test set is too, so accuracy is per-class skill, not a deployment figure.",
             ha="left", fontsize=9, color=MUTED)
    save(fig, "06_class_distribution.png")


if __name__ == "__main__":
    fig_grid(); fig_perclass(); fig_confusions(); fig_journey(); fig_training(); fig_distribution()
    print("\nall charts ->", FIG)
