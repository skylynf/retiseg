"""Fig. 3 | Performance of the original models (E1 author weights, E1r author recipe).

a  three numbers per released checkpoint: paper, author function, this protocol
b  where the gap comes from, step by step (M2MRF, measured)
c  retraining with the authors' recipe against the reported number
d  author weights on every test set, per class
"""

import numpy as np

from retifig import data as D
from retifig import panels as P
from retifig import style as S

C_PAPER, C_AUTH, C_OURS = "#898781", "#b3261e", "#1d3f5e"
CKPT = [("M2MRF", "IDRiD"), ("M2MRF", "DDR"), ("WSRFNet", "IDRiD"), ("WSRFNet", "DDR")]


def get(exp, model, variant, train, test, metric, cls="mean"):
    r = D.sel(experiment=exp, model=model, variant=variant, train_set=train, test_set=test, metric=metric, cls=cls)
    if len(r) == 0:
        return np.nan, True
    return float(r["value"].mean()), D.simulated(r)


def panel_a(fig, top):
    rows, a, b, c, sims = [], [], [], [], []
    for model, train in CKPT:
        v1, s1 = get("E1", model, "paper", train, train, "aupr_author_impl")
        v2, s2 = get("E1", model, "author_function", train, train, "aupr_author_impl")
        v3, s3 = get("E1", model, "this_protocol", train, train, "aupr")
        rows.append(f"{model} · {train}")
        a.append(v1), b.append(v2), c.append(v3), sims.append(s1 or s2 or s3)
    ax = S.axes_mm(fig, 27, top, 40, 26)
    P.dumbbell(ax, rows, [a, b, c], [C_PAPER, C_AUTH, C_OURS],
               ["reported in the paper", "author checkpoint, author metric", "author checkpoint, this protocol"],
               markers=["D", "^", "o"], xlim=(0.3, 0.75), size=20, fs=5.6)
    for i in range(len(rows)):
        ax.text(0.758, i, f"−{b[i] - c[i]:.3f}", ha="left", va="center", fontsize=5.2, color=S.INK, clip_on=False)
        if sims[i] and not S.FINAL:
            ax.text(0.305, i - 0.33, "simulated", fontsize=4.2, color="#9a5b00", style="italic", va="center")
    ax.text(0.758, -0.85, "metric\ngap", ha="left", va="center", fontsize=4.9, color=S.INK2, clip_on=False, linespacing=1.1)
    ax.axhline(1.5, color=S.INK2, lw=0.5, ls=(0, (2, 2)))
    ax.set_xlabel("mAUPR on the official test set")
    ax.legend(loc="upper left", bbox_to_anchor=(-0.66, -0.3), ncol=1, handletextpad=0.2, labelspacing=0.25, fontsize=5.4)


def panel_b(fig, top):
    ax = S.axes_mm(fig, 86, top, 38, 30)
    steps = ["author function,\nfloat output", "same function,\n16-bit maps", "exact AUPR,\nFOV, multilabel"]
    for train, color, mk in (("IDRiD", C_OURS, "o"), ("DDR", "#6a94ba", "s")):
        v = [get("E1", "M2MRF", "author_function", train, train, "aupr_author_impl")[0],
             get("E1", "M2MRF", "this_protocol", train, train, "aupr_11pt")[0],
             get("E1", "M2MRF", "this_protocol", train, train, "aupr")[0]]
        ax.plot(range(3), v, color=color, lw=1.1, marker=mk, ms=4, mec=S.PAPER, mew=0.5, zorder=3)
        for i, val in enumerate(v):
            ax.text(i, val + 0.02, f"{val + 1e-9:.3f}", ha="center", va="bottom", fontsize=5.2, color=S.INK)
        ax.text(2.12, v[-1], train, ha="left", va="center", fontsize=5.6, color=S.INK, clip_on=False)
    ax.set_xticks(range(3))
    ax.set_xticklabels(steps, fontsize=5.0)
    ax.set_xlim(-0.35, 2.3)
    ax.set_ylim(0.3, 0.76)
    ax.set_yticks([0.3, 0.4, 0.5, 0.6, 0.7])
    ax.set_ylabel("mAUPR, M2MRF author weights")
    S.light_grid(ax)
    ax.text(0.5, 0.325, "threshold 0 counts\nevery p > 0", ha="center", va="bottom", fontsize=4.7, color=S.INK2, linespacing=1.1)
    ax.text(1.5, 0.325, "FOV, pooling,\nlabel overlap", ha="center", va="bottom", fontsize=4.7, color=S.INK2, linespacing=1.1)
    ax.text(0.98, 1.02, "measured", transform=ax.transAxes, ha="right", va="bottom", fontsize=4.8, color=S.INK2, style="italic")


def panel_c(fig, top):
    ax = S.axes_mm(fig, 142, top, 34, 30)
    groups = [("M2MRF", "IDRiD"), ("M2MRF", "DDR"), ("HACDR-Net", "IDRiD"), ("HACDR-Net", "DDR")]
    rng = np.random.default_rng(1)
    any_sim = False
    for i, (model, train) in enumerate(groups):
        paper = D.sel(experiment="E1r", model=model, variant="paper", train_set=train, metric="aupr_author_impl")
        runs = D.sel(experiment="E1r", model=model, variant="author_recipe", train_set=train, metric="aupr_author_impl")
        any_sim |= D.simulated(runs)
        pv = float(paper["value"].iloc[0])
        ax.plot([i - 0.32, i + 0.32], [pv, pv], color=C_PAPER, lw=1.4, zorder=2, solid_capstyle="round")
        col = S.FAMILY["fundus"]
        x = i + rng.uniform(-0.14, 0.14, len(runs))
        ax.scatter(x, runs["value"], s=13, color=col, edgecolors=S.PAPER, linewidths=0.4, zorder=3)
        ax.plot([i, i], [runs["value"].mean(), pv], color=S.AXIS, lw=0.6, zorder=1)
        ax.text(i, pv + 0.012, f"{runs['value'].mean() - pv:+.3f}", ha="center", va="bottom", fontsize=5.2)
    first = D.sel(experiment="E1r", model="M2MRF", variant="bs1_first_run", metric="aupr_author_impl")
    ax.scatter([0.3], first["value"], s=16, marker="X", color=S.INK, zorder=4, linewidths=0)
    ax.annotate("first run, batch 1\n(measured)", (0.3, float(first["value"].iloc[0])), xytext=(1.25, 0.665), fontsize=4.7, color=S.INK2,
                arrowprops=dict(arrowstyle="-", lw=0.4, color=S.MUTED), ha="center", va="center", linespacing=1.1)
    ax.set_xticks(range(4))
    ax.set_xticklabels([t for _, t in groups], fontsize=5.2)
    for xc, name in ((0.5, "M2MRF"), (2.5, "HACDR-Net")):
        ax.text(xc, -0.2, name, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=5.6)
        ax.plot([xc - 0.8, xc + 0.8], [-0.17, -0.17], transform=ax.get_xaxis_transform(), color=S.AXIS, lw=0.6, clip_on=False)
    ax.set_xlim(-0.6, 3.6)
    ax.set_ylim(0.3, 0.76)
    ax.set_yticks([0.3, 0.4, 0.5, 0.6, 0.7])
    ax.set_ylabel("mAUPR, author metric")
    S.light_grid(ax)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=C_PAPER, lw=1.4, label="reported"), Line2D([], [], marker="o", ls="", ms=3.4, color=S.FAMILY["fundus"], label="retrained, one seed")]
    ax.legend(handles=h, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, columnspacing=1.0, handletextpad=0.3, fontsize=5.4)
    if any_sim:
        S.tag_simulated(ax, loc="br")


def panel_d(fig, top):
    x0, row_h = 27.0, 3.6
    rows = [f"{m} · {t}" for m, t in CKPT]
    means = []
    for j, test in enumerate(D.TESTS):
        M = np.full((len(CKPT), 4), np.nan)
        for i, (model, train) in enumerate(CKPT):
            for k, cls in enumerate(D.CLASSES):
                M[i, k] = get("E1", model, "this_protocol", train, test, "aupr", cls)[0]
        means.append(np.nanmean(M, axis=1))
        ax = S.axes_mm(fig, x0 + j * 17.6, top, 16.4, row_h * len(rows))
        P.heat(ax, M, rows=rows if j == 0 else None, cols=D.CLASSES, vmin=0, vmax=0.9, fs=4.8)
        ax.tick_params(axis="x", labelsize=5.0)
        tier = S.DATASET_TIER[test]
        P.header_band(ax, [(0, 4, test, S.TIER[tier])], y=1.06, h=0.24, fs=5.4)
        # mark the cells where the checkpoint was trained on this source
        for i, (_, train) in enumerate(CKPT):
            if train == test:
                ax.add_patch(__import__("matplotlib").patches.Rectangle((0.03, i + 0.04), 3.94, 0.92, fill=False, ec=S.INK, lw=0.8, zorder=6))
    ax = S.axes_mm(fig, x0 + 7 * 17.6 + 1.5, top, 22, row_h * len(rows))
    A = np.array(means).T
    internal = np.array([A[i, D.TESTS.index(train)] for i, (_, train) in enumerate(CKPT)])
    indep = np.array([A[i, [D.TESTS.index(t) for t in D.FINE_INDEPENDENT]].mean() for i in range(len(CKPT))])
    P.heat(ax, np.c_[internal, indep], cols=["Int.", "Ind.\nfine"], vmin=0, vmax=0.9, fs=4.8)
    ax.tick_params(axis="x", labelsize=5.0)
    S.text_mm(fig, "mAUPR", x0 + 7 * 17.6 + 1.5, top - 5.0, fontsize=5.4)
    S.tag_simulated(ax, loc="tr", text="simulated except M2MRF internal", dy=0.78)
    P.colorbar(fig, S.rect_mm(fig, 27, top + row_h * len(rows) + 9, 26, 1.6), S.CMAP_SEQ, 0, 0.9, "AUPR, this protocol", ticks=[0, 0.3, 0.6, 0.9])
    S.text_mm(fig, "Outlined cells: test set of the source the checkpoint was trained on (internal test). Header colour: annotation granularity (fine → region consensus).",
              60, top + row_h * len(rows) + 7.6, fontsize=5.0, color=S.INK2)


def main():
    fig = S.figure(100)
    S.label_mm(fig, "a", 2, 2)
    S.text_mm(fig, "Three numbers per checkpoint", 8, 2.6, fontsize=6.8, fontweight="bold")
    panel_a(fig, 11)
    S.label_mm(fig, "b", 72, 2)
    S.text_mm(fig, "Where the gap comes from", 78, 2.6, fontsize=6.8, fontweight="bold")
    panel_b(fig, 11)
    S.label_mm(fig, "c", 128, 2)
    S.text_mm(fig, "Author recipe, retrained", 134, 2.6, fontsize=6.8, fontweight="bold")
    panel_c(fig, 11)
    S.label_mm(fig, "d", 2, 57)
    S.text_mm(fig, "Author weights on internal and independent tests", 8, 57.6, fontsize=6.8, fontweight="bold")
    panel_d(fig, 70)
    S.save(fig, "fig3_original")


if __name__ == "__main__":
    main()
