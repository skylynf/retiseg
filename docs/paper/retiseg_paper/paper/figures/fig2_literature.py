"""Fig. 2 | The numbers in the literature (L1, L2, and the metric-implementation audit of E1).

a  reproducibility funnel for fundus-specific methods (levels from models.yaml)
b  what method papers report: split, AUPR implementation, provenance of baseline numbers (literature/meta, measured)
c  spread of the mAUPR reported for one method across papers, against the retrained value
d  the same probability maps scored with the exact AUPR and with the 11-threshold implementation
e  rank change caused by the implementation alone
"""

import numpy as np
from scipy import stats as sps

from retifig import data as D
from retifig import panels as P
from retifig import style as S

DR_ADAPTED = ["TP-DRSeg", "SAM2-DR-Chen2025", "Ophthalmology-SAM"]
RAMP = ["#c9d9e8", "#8fb0cc", "#4f7ea8", "#1d3f5e", "#eb6834"]
PART = ["#1d3f5e", "#6a94ba", "#b9cde0", "#e4e3dc"]


def dr_specific():
    reg = D.registry()
    keep = reg.family.isin(["fundus_multiscale", "relational_transformer", "semi_supervised"]) | reg.name.isin(DR_ADAPTED)
    return reg[keep]


def panel_a(fig, top):
    reg = dr_specific()
    level = reg.repro_level.str[1].astype(int)
    rep = D.sel(experiment="L2", variant="reproduced within 0.02 mAUPR")
    labels = ["Fundus-specific methods registered", "Author code found", "Training code", "Code, weights, splits\nand environment", "Reported number reproduced"]
    notes = ["multi-scale CNN, relation, semi-supervised, adapted FM", "level R1 or higher", "level R3 or higher", "level R4", "within 0.02 mAUPR, author recipe"]
    values = [len(reg), int((level >= 1).sum()), int((level >= 3).sum()), int((level >= 4).sum()), int(rep["value"].iloc[0])]
    ax = S.axes_mm(fig, 2, top, 76, 40)
    P.funnel(ax, labels, values, RAMP, notes=notes, fs=5.6, width_scale="linear")
    ax.set_xlim(-1.75, 0.6)
    for t in ax.texts:
        if t.get_ha() == "right":
            t.set_x(-0.6)
    # the last stage is derived from the author-recipe reruns (collect_runs.derive_funnel); tagged only while placeholder
    if D.simulated(rep) and not S.FINAL:
        ax.text(0.14, 4.5, "simulated", fontsize=4.6, color="#9a5b00", style="italic", va="center",
                bbox=dict(boxstyle="round,pad=0.22", fc="#fff3d6", ec="#e8c36a", lw=0.3))


def stacked_row(ax, y, parts, colors, total_label):
    total = sum(v for _, v in parts)
    left = 0.0
    prev_end, prev_above = -1.0, False
    for (name, v), c in zip(parts, colors):
        w = v / total
        if w == 0:
            continue
        above = not (left < prev_end and prev_above)
        prev_end, prev_above = left + 0.017 * len(name) + 0.02, above
        ax.barh(y, w, left=left, height=0.5, color=c, lw=0, zorder=2)
        ax.plot([left + w] * 2, [y - 0.25, y + 0.25], color=S.PAPER, lw=0.9, zorder=3)
        rgba = __import__("matplotlib.colors", fromlist=["to_rgba"]).to_rgba(c)
        if w > 0.07:
            ax.text(left + w / 2, y, f"{100 * w:.0f}%", ha="center", va="center", fontsize=5.2, color=S.text_on(rgba), zorder=4)
        ax.text(left + 0.004, y - 0.31 if above else y + 0.31, name, ha="left", va="bottom" if above else "top", fontsize=4.9, color=S.INK2)
        left += w
    ax.text(1.012, y, total_label.format(n=int(total)), ha="left", va="center", fontsize=5.0, color=S.INK2)


def panel_b(fig, top):
    """Paper audit (L1): what the method papers in the ledger report. Measured."""
    ax = S.axes_mm(fig, 108, top + 1, 56, 40)
    L = D.sel(experiment="L1", model="papers")

    def parts(prefix, names, **kw):
        sub = L[L.variant.str.startswith(prefix)]
        for k, v in kw.items():
            sub = sub[sub[k] == v]
        return [(n, sub[sub.variant == f"{prefix}{n}"]["value"].sum()) for n in names]

    rows = [
        ("Split on IDRiD", parts("split:", ["official", "own split", "not stated"], train_set="IDRiD"), "{n} papers"),
        ("Split on DDR", parts("split:", ["official", "own split", "not stated"], train_set="DDR"), "{n} papers"),
        ("AUPR implementation", parts("impl:", ["exact AP", "11-threshold", "per-image mean", "not stated"]), "{n} papers"),
        ("Baseline numbers", parts("provenance:", ["re-run", "copied once", "copied again", "unverifiable"]), "{n} numbers"),
    ]
    short = {"11-threshold": "fixed thresholds"}
    for i, (title, p, tot) in enumerate(rows):
        p = [(short.get(n, n), v) for n, v in p]
        stacked_row(ax, i, p, PART, tot)
        ax.text(-0.02, i, title, ha="right", va="center", fontsize=5.8)
    ax.set_xlim(0, 1)
    ax.set_ylim(len(rows) - 0.45, -0.75)
    S.bare(ax)


def panel_c(fig, top):
    L = D.sel(experiment="L1", metric="reported_maupr")
    L = L[L.variant.str.startswith("paper:")]
    methods = list(L.groupby("model")["value"].median().sort_values(ascending=False).index)
    ax = S.axes_mm(fig, 22, top + 5, 44, 46)
    retr = D.score_matrix("IDRiD")["IDRiD"]
    alias = {"HRNet": "HRNet-W48", "DeepLabv3+": "DeepLabv3"}
    for i, m in enumerate(methods):
        v = L[L.model == m]["value"].to_numpy()
        ax.plot([v.min(), v.max()], [i, i], color=S.AXIS, lw=2.4, solid_capstyle="round", zorder=1)
        ax.scatter(v, np.full(len(v), i), s=9, color="#4f7ea8", edgecolors=S.PAPER, linewidths=0.3, zorder=3)
        key = alias.get(m, m)
        if key in retr.index:
            ax.scatter([retr[key]], [i], s=20, marker="D", color=S.FAMILY["fundus"], edgecolors=S.PAPER, linewidths=0.4, zorder=4)
        ax.text(0.925, i, f"n = {len(v)}", fontsize=4.8, color=S.MUTED, va="center", ha="right")
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods)
    ax.set_ylim(len(methods) - 0.4, -0.6)
    ax.set_xlim(0.3, 0.93)
    ax.set_xlabel("mAUPR on IDRiD")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    S.light_grid(ax, "x")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ls="", ms=3.2, color="#4f7ea8", label="reported in one paper"),
         Line2D([], [], marker="D", ls="", ms=3.4, color=S.FAMILY["fundus"], label="retrained here (B-std)")]
    ax.legend(handles=h, loc="lower left", bbox_to_anchor=(-0.4, 1.01), ncol=1, handletextpad=0.2, labelspacing=0.2, fontsize=5.4)
    if D.simulated(D.main_runs()):
        S.tag_simulated(ax, loc="tr", text="retrained values simulated")


def impl_table():
    exact = D.score_matrix("IDRiD")["IDRiD"]
    eleven = D.score_matrix("IDRiD", metric="aupr_11pt")["IDRiD"]
    return exact, eleven


def panel_d(fig, top):
    exact, eleven = impl_table()
    order = exact.sort_values(ascending=False).index
    e1 = D.sel(experiment="E1", model="M2MRF", cls="mean")
    def val(train, variant, metric):
        r = e1[(e1.train_set == train) & (e1.test_set == train) & (e1.variant == variant) & (e1.metric == metric)]
        return float(r["value"].iloc[0])
    rows = ["M2MRF, IDRiD checkpoint", "M2MRF, DDR checkpoint"] + list(order)
    a = [val("IDRiD", "this_protocol", "aupr"), val("DDR", "this_protocol", "aupr")] + list(exact[order])
    b = [val("IDRiD", "this_protocol", "aupr_11pt"), val("DDR", "this_protocol", "aupr_11pt")] + list(eleven[order])
    c = [val("IDRiD", "author_function", "aupr_author_impl"), val("DDR", "author_function", "aupr_author_impl")] + [np.nan] * len(order)
    ax = S.axes_mm(fig, 94, top + 5, 40, 46)
    P.dumbbell(ax, rows, [a, b, c], ["#1d3f5e", "#eda100", "#b3261e"],
               ["exact AUPR (this protocol)", "11 thresholds, stored 16-bit maps", "11 thresholds, author function on float output"],
               markers=["o", "s", "^"], xlim=(0.3, 0.75), size=15, fs=5.3)
    ax.axhline(1.5, color=S.INK2, lw=0.5, ls=(0, (2, 2)))
    ax.axhspan(-0.5, 1.5, color=S.SOFT, zorder=0, lw=0)
    ax.text(0.305, 0.5, "author weights,\nmeasured", fontsize=4.7, color=S.INK2, ha="left", va="center", linespacing=1.1)
    ax.text(0.305, 2.5, "retrained,\nIDRiD test", fontsize=4.7, color=S.INK2, ha="left", va="center", linespacing=1.1)
    ax.set_xlabel("mAUPR")
    ax.legend(loc="lower left", bbox_to_anchor=(-0.6, 1.01), ncol=1, handletextpad=0.2, labelspacing=0.2, fontsize=5.4)
    if D.simulated(D.main_runs()) or D.simulated(D.main_runs("aupr_11pt")):
        S.tag_simulated(ax, loc="br", text="retrained rows simulated")


def panel_e(fig, top):
    exact, eleven = impl_table()
    ra, rb = sps.rankdata(-exact.to_numpy()), sps.rankdata(-eleven.to_numpy())
    tau = D.bstats.kendall_tau(exact.to_numpy(), eleven.to_numpy())
    ax = S.axes_mm(fig, 157, top + 8, 9, 43)
    names = [m.replace(" (full FT)", "").replace("RETFound + UPerNet", "RETFound+UPer.").replace("nnU-Net ResEnc-M", "nnU-Net ResEnc") for m in exact.index]
    cols = [S.FAMILY[D.FAMILY_OF[m]] for m in exact.index]
    P.bump(ax, names, ra, rb, cols, labels=("exact", "11-thr."), fs=4.9, lw=0.9)
    moved = int((ra != rb).sum())
    S.text_mm(fig, f"{moved} of {len(ra)} models change rank · Kendall τ = {tau['tau']:.2f}", 161.5, top + 54.5, ha="center", fontsize=5.0, color=S.INK2)
    if D.simulated(D.main_runs()) or D.simulated(D.main_runs("aupr_11pt")):
        S.tag_simulated(ax, loc="tr", dx=1.2, dy=0.03)


def main():
    fig = S.figure(126)
    S.label_mm(fig, "a", 2, 2)
    S.text_mm(fig, "From a published method to a reproduced number", 8, 2.6, fontsize=6.8, fontweight="bold")
    panel_a(fig, 9)
    S.label_mm(fig, "b", 84, 2)
    S.text_mm(fig, "What method papers report", 90, 2.6, fontsize=6.8, fontweight="bold")
    panel_b(fig, 9)
    S.label_mm(fig, "c", 2, 55)
    S.text_mm(fig, "One method, many reported values", 8, 55.6, fontsize=6.8, fontweight="bold")
    panel_c(fig, 66)
    S.label_mm(fig, "d", 70, 55)
    S.text_mm(fig, "Same maps, two AUPR implementations", 76, 55.6, fontsize=6.8, fontweight="bold")
    panel_d(fig, 66)
    S.label_mm(fig, "e", 138, 55)
    S.text_mm(fig, "Rank by implementation", 144, 55.6, fontsize=6.8, fontweight="bold")
    panel_e(fig, 66)
    S.save(fig, "fig2_literature")


if __name__ == "__main__":
    main()
