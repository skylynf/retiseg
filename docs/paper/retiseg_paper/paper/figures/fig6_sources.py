"""Fig. 6 | Annotation granularity and training source (label audit, E2 + E3b, E4).

a  annotation style of every source: lesions per image against lesion size
b  training source x test set, mean of the representative models
c  loss against training on the test set's own source
d  the same matrix for each representative model
e  adding mixed or coarse labels to fine-label training (E4)
"""

import numpy as np
from matplotlib.lines import Line2D

from retifig import data as D
from retifig import panels as P
from retifig import style as S

TRAINS = ["IDRiD", "DDR", "TJDR-std", "FGADR"]
CLS_MARKER = {"MA": "o", "HE": "s", "EX": "^", "SE": "D"}


def matrix(model=None):
    """train x test seed-mean mAUPR for one representative model, or their mean."""
    a = D.sel(experiment=["E2", "E5"], variant=D.MAIN_VARIANTS, metric="aupr", model=D.REPRESENTATIVE)
    b = D.sel(experiment="E3b", metric="aupr", model=D.REPRESENTATIVE)
    df = __import__("pandas").concat([a, b])
    if model:
        df = df[df.model == model]
    # measured cells only: a cell still in training (or one the protocol does not score, such as the ultra-widefield set
    # for sources whose field is unknown) stays empty instead of carrying a placeholder into the mean
    sim = D.simulated(df)
    df = df[df.source == "measured"]
    g = df.groupby(["model", "train_set", "test_set"])["value"].mean().groupby(["train_set", "test_set"]).mean().unstack("test_set")
    return g.reindex(index=TRAINS, columns=D.TESTS), sim


CODE = {"IDRiD": "I", "DDR": "D", "TJDR-std": "Ts", "TJDR-uwf": "Tu", "MAPLES-DR": "M", "FGADR": "F", "Retinal-Lesions": "R", "DiaRetDB1": "Dr"}


def panel_a(fig, top):
    r = D.sel(experiment="audit")
    n = r[r.metric == "lesions_per_image"].set_index(["train_set", "cls"])["value"]
    d = r[r.metric == "median_rel_diameter"].set_index(["train_set", "cls"])["value"]
    src = r[r.metric == "lesions_per_image"].set_index(["train_set", "cls"])["source"]
    for k, cls in enumerate(D.CLASSES):
        col, row = k % 2, k // 2
        ax = S.axes_mm(fig, 15 + col * 27, top + row * 23, 24.5, 19.5)
        for ds in D.TESTS:
            color = S.TIER[S.DATASET_TIER[ds]]
            measured = src[(ds, cls)] == "measured"
            x, y = n[(ds, cls)], d[(ds, cls)]
            ax.scatter([x], [y], s=15, color=color, edgecolors=S.INK if measured else S.PAPER, linewidths=0.7 if measured else 0.3, zorder=3)
            ax.annotate(CODE[ds], (x, y), xytext=(2.6, 1.6), textcoords="offset points", fontsize=4.9, color=S.INK, zorder=4)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(0.7, 400)
        ax.set_ylim(0.0018, 0.2)
        ax.set_xticks([1, 10, 100])
        ax.set_yticks([0.003, 0.01, 0.03, 0.1])
        ax.set_xticklabels(["1", "10", "100"] if row else [])
        ax.set_yticklabels(["0.003", "0.01", "0.03", "0.1"] if col == 0 else [])
        ax.minorticks_off()
        ax.grid(color=S.GRID, lw=0.4, zorder=0)
        ax.set_title(f"{cls} · {S.LESION_NAME[cls].lower()}", fontsize=5.4, fontweight="bold", loc="left", pad=1.5)
        if k == 3 and (src != "measured").any():
            # the label-style audit (bench/scripts/label_style_audit.py, 2026-10-24) measured every source; the tag
            # names the measured ones only while a placeholder row is left
            done = sorted({CODE[d] for d in D.TESTS if src[(d, cls)] == "measured"})
            S.tag_simulated(ax, loc="tr", text=f"simulated except {', '.join(done)} (outlined)", dx=0.02, dy=-0.1)
    S.text_mm(fig, "Lesions per image", 15 + 26, top + 48.5, ha="center", fontsize=6.2)
    S.text_mm(fig, "Median lesion diameter / FOV diameter", 3.5, top + 21, rotation=90, va="center", ha="center", fontsize=6.2)
    h = [Line2D([], [], marker="o", ls="", ms=3.6, color=S.TIER[t], label=S.TIER_LABEL[t]) for t in S.TIER]
    fig.legend(handles=h, loc="upper left", bbox_to_anchor=tuple(S.rect_mm(fig, 13, top - 8.5, 1, 1)[:2]), ncol=4, columnspacing=0.8, handletextpad=0.05, fontsize=5.2)
    S.text_mm(fig, "Codes as in d.", 15, top + 52.2, fontsize=4.9, color=S.INK2)


def tier_ticks(ax, tests):
    trans = ax.get_xaxis_transform()
    for j, t in enumerate(tests):
        ax.plot([j + 0.12, j + 0.88], [1.03, 1.03], transform=trans, color=S.TIER[S.DATASET_TIER[t]], lw=2.2, clip_on=False, solid_capstyle="butt")


def panel_b(fig, top):
    M, sim = matrix()
    ax = S.axes_mm(fig, 92, top, 50, 22)   # eight test columns since MAPLES-DR joined (2026-10-14)
    P.heat(ax, M.to_numpy(), rows=TRAINS, cols=[S.DATASET_SHORT[t] for t in D.TESTS], vmin=0, vmax=0.7, fs=5.2)
    ax.tick_params(axis="x", labelsize=4.4)
    for i, tr in enumerate(TRAINS):
        j = D.TESTS.index(tr)
        ax.add_patch(__import__("matplotlib").patches.Rectangle((j + 0.04, i + 0.05), 0.92, 0.9, fill=False, ec=S.INK, lw=0.9, zorder=6))
    tier_ticks(ax, D.TESTS)
    ax.set_ylabel("Training source")
    S.text_mm(fig, "Test set (mAUPR, mean of the measured representative models)", 92, top - 5.2, fontsize=5.6, color=S.INK2)
    if sim:
        S.tag_simulated(ax, loc="tr", dy=0.3, text="one cell pending")
    P.colorbar(fig, S.rect_mm(fig, 92, top + 33, 22, 1.5), S.CMAP_SEQ, 0, 0.7, "mAUPR", ticks=[0, 0.35, 0.7])
    return M, sim


def panel_c(fig, top, M):
    tests = TRAINS
    A = M[tests].to_numpy()
    delta = A - np.diag(A)[None, :]
    ax = S.axes_mm(fig, 149, top, 26, 22)
    P.heat(ax, delta, rows=None, cols=[S.DATASET_SHORT[t] for t in tests], cmap=S.CMAP_DIV, vmin=-0.5, vmax=0.5, fmt="{:+.2f}", fs=5.2)
    ax.tick_params(axis="x", labelsize=5.0)
    tier_ticks(ax, tests)
    S.text_mm(fig, "Δ against in-source training", 149, top - 5.2, fontsize=5.6, color=S.INK2)
    P.colorbar(fig, S.rect_mm(fig, 149, top + 33, 22, 1.5), S.CMAP_DIV, -0.5, 0.5, "Δ mAUPR", ticks=[-0.5, 0, 0.5])


def panel_d(fig, top):
    x = 22.0
    for k, model in enumerate(D.REPRESENTATIVE):
        M, _ = matrix(model)
        ax = S.axes_mm(fig, x, top, 28, 12.5)
        P.heat(ax, M.to_numpy(), rows=TRAINS if k == 0 else None, cols=[CODE[t] for t in D.TESTS], vmin=0, vmax=0.7, fs=4.3)
        ax.tick_params(axis="both", labelsize=4.9)
        tier_ticks(ax, D.TESTS)
        st = S.REP_STYLE[model]
        ax.text(0.07, 1.14, model, transform=ax.transAxes, fontsize=5.8, va="bottom")
        ax.scatter([0.02], [1.2], transform=ax.transAxes, s=10, marker=st["marker"], color=st["color"], clip_on=False)
        x += 31.3
    S.text_mm(fig, "I, IDRiD · D, DDR · Ts, TJDR-std · Tu, TJDR-uwf · M, MAPLES-DR · F, FGADR · R, Retinal-Lesions · Dr, DiaRetDB1", 22, top + 17.5, fontsize=4.9, color=S.INK2)


def panel_e(fig, top):
    configs = ["DDR", "DDR + FGADR (383)", "DDR + FGADR", "DDR + Retinal-Lesions", "FGADR", "Retinal-Lesions"]
    labels = ["DDR\n(fine)", "DDR + FGADR\n383 images", "DDR +\nFGADR", "DDR +\nRet.-Lesions", "FGADR\n(mixed)", "Ret.-Lesions\n(coarse)"]
    r = D.sel(experiment="E4", metric="aupr")
    g = r.groupby(["model", "variant", "test_set"])["value"].mean().unstack("test_set")
    sd = r.groupby(["model", "variant", "test_set", "seed"])["value"].mean().groupby(["model", "variant", "test_set"]).std().unstack("test_set")
    for k, test in enumerate(("IDRiD", "TJDR-std")):
        ax = S.axes_mm(fig, 16 + k * 86, top, 74, 28)
        for model in [m for m in S.REP_STYLE if m in g.index.get_level_values(0)]:
            st = S.REP_STYLE[model]
            y = [g.loc[(model, c), test] for c in configs]
            e = [sd.loc[(model, c), test] for c in configs]
            e = None if all(np.isnan(v) for v in e) else e
            ax.errorbar(range(len(configs)), y, yerr=e, color=st["color"], marker=st["marker"], ls=st["ls"], lw=1.0, ms=3.6, mec=S.PAPER, mew=0.4,
                        elinewidth=0.5, capsize=0, zorder=3)
        ax.axvspan(-0.4, 0.4, color=S.SOFT, zorder=0, lw=0)
        ax.axvline(3.5, color=S.INK2, lw=0.5, ls=(0, (2, 2)))
        ax.text(1.75, 0.985, "fine labels plus mixed or coarse labels", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=4.9, color=S.INK2)
        ax.text(4.5, 0.985, "no fine labels", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=4.9, color=S.INK2)
        ax.set_xticks(range(len(configs)))
        ax.set_xticklabels(labels, fontsize=5.0, linespacing=1.1)
        ax.set_xlim(-0.4, len(configs) - 0.6)
        ax.set_ylim(0, 0.75)
        ax.set_yticks([0, 0.2, 0.4, 0.6])
        ax.set_ylabel(f"mAUPR on {test} test")
        ax.set_title(f"Tested on {test} (fine labels, never used for training)", fontsize=6.0, loc="left")
        S.light_grid(ax)
    if D.simulated(r):
        S.tag_simulated(ax, loc="br")
    S.rep_legend(fig, loc="upper left", bbox=tuple(S.rect_mm(fig, 60, top - 11.5, 1, 1)[:2]))


def main():
    fig = S.figure(158)
    S.label_mm(fig, "a", 2, 2)
    S.text_mm(fig, "Annotation style of each source", 8, 2.6, fontsize=6.8, fontweight="bold")
    panel_a(fig, 14)
    S.label_mm(fig, "b", 72, 2)
    S.text_mm(fig, "Training source × test set", 78, 2.6, fontsize=6.8, fontweight="bold")
    M, _ = panel_b(fig, 18)
    S.label_mm(fig, "c", 143, 2)
    S.text_mm(fig, "Cost of a foreign source", 149, 2.6, fontsize=6.8, fontweight="bold")
    panel_c(fig, 18, M)
    S.label_mm(fig, "d", 2, 70)
    S.text_mm(fig, "The same matrix for each representative model", 8, 70.6, fontsize=6.8, fontweight="bold")
    panel_d(fig, 82)
    S.label_mm(fig, "e", 2, 105)
    S.text_mm(fig, "Adding mixed or coarse labels at an equal number of image passes", 8, 105.6, fontsize=6.8, fontweight="bold")
    panel_e(fig, 119)
    S.save(fig, "fig6_sources")


if __name__ == "__main__":
    main()
