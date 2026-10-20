"""Fig. 7 | Lesion size, input resolution and training amount (frozen audits, E2, E7, E8).

a  lesion size by class, with the frozen small / medium / large boundaries   (measured)
b  lesions that vanish from the training target when the input is shrunk     (measured)
c  lesion-level recall by lesion size, per model
d  input resolution: mAUPR and recall of small lesions
e  training amount and merged sources
"""

import numpy as np

from retifig import data as D
from retifig import panels as P
from retifig import style as S

ROW = 3.05
EDGES = (0.005, 0.011)


def panel_a(fig, top):
    fr = D.lesion_size_freeze()["per_dataset"]
    ax = S.axes_mm(fig, 16, top, 50, 33)
    ax.axvspan(0.001, EDGES[0], color="#eef4f9", zorder=0, lw=0)
    ax.axvspan(EDGES[1], 0.1, color="#f6efe6", zorder=0, lw=0)
    for e in EDGES:
        ax.axvline(e, color=S.INK2, lw=0.6, ls=(0, (2, 2)), zorder=1)
        ax.text(e, -0.62, f"{e:g}", ha="center", va="bottom", fontsize=5.0, color=S.INK2)
    for x, t in ((0.0026, "small"), (0.0074, "medium"), (0.035, "large")):
        ax.text(x, 7.72, t, ha="center", va="center", fontsize=5.2, color=S.INK2, style="italic")
    y, ticks, labels = 0, [], []
    for cls in D.CLASSES:
        for ds, shade in (("IDRiD", 1.0), ("DDR", 0.5)):
            q = fr[ds][cls]
            ax.plot([q["p10"], q["p90"]], [y, y], color=S.LESION[cls], lw=0.8, alpha=shade, zorder=2, solid_capstyle="round")
            ax.plot([q["p25"], q["p75"]], [y, y], color=S.LESION[cls], lw=3.2, alpha=shade, zorder=3, solid_capstyle="butt")
            ax.scatter([q["p50"]], [y], s=11, color=S.PAPER, edgecolors=S.INK, linewidths=0.6, zorder=4)
            ax.text(0.095, y, f"n = {q['n']:,}", ha="right", va="center", fontsize=4.7, color=S.MUTED)
            ticks.append(y)
            labels.append(f"{cls} · {ds}")
            y += 0.85
        y += 0.3
    ax.set_xscale("log")
    ax.set_xlim(0.0012, 0.1)
    ax.set_xticks([0.002, 0.005, 0.01, 0.02, 0.05])
    ax.set_xticklabels(["0.002", "0.005", "0.01", "0.02", "0.05"])
    ax.minorticks_off()
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=5.3)
    ax.set_ylim(y - 0.5 + 0.9, -0.65)
    ax.set_xlabel("Lesion equivalent diameter / FOV diameter")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    S.text_mm(fig, "Training images only (IDRiD 44, DDR 383). Bar, quartiles; line, 10th–90th percentile; dot, median.", 4, top + 42.5, fontsize=4.9, color=S.INK2)


def panel_b(fig, top):
    audit = D.input_label_audit()["datasets"]
    sizes = [("long_side_1440", "1,440\n(canvas)"), ("h2former_960", "960"), ("fct_384", "384"), ("swin_224", "224")]
    for k, ds in enumerate(("IDRiD", "DDR")):
        ax = S.axes_mm(fig, 86 + k * 46, top, 40, 33)
        tab = audit[ds]["train"]["lesion_disappearance"]
        x = np.arange(len(sizes))
        w = 0.19
        for j, cls in enumerate(D.CLASSES):
            v = [100 * tab[key][cls]["ratio"] for key, _ in sizes]
            ax.bar(x + (j - 1.5) * (w + 0.015), v, w, color=S.LESION[cls], lw=0, zorder=2, label=cls)
            for xi, vi in zip(x + (j - 1.5) * (w + 0.015), v):
                if vi >= 5:
                    ax.text(xi, vi + 0.8, f"{vi:.0f}", ha="center", va="bottom", fontsize=4.9)
        ax.set_xticks(x)
        ax.set_xticklabels([lab for _, lab in sizes], fontsize=5.3)
        ax.set_ylim(0, 42)
        ax.set_yticks([0, 10, 20, 30, 40])
        ax.set_title(f"{ds} training set", fontsize=6.0, loc="left")
        S.light_grid(ax)
        if k == 0:
            ax.set_ylabel("Lesions lost from the\ntraining target (%)")
            ax.legend(loc="upper left", ncol=4, columnspacing=0.8, handlelength=0.9, handletextpad=0.3, fontsize=5.4)
        else:
            ax.set_yticklabels([])
        ax.set_xlabel("Whole image resized to (pixels)")


def panel_c(fig, top, models, fams):
    x = 30.0
    for k, train in enumerate(D.TRAIN_MAIN):
        cols = []
        for size in ("small", "medium", "large"):
            m = D.score_matrix(train, metric=f"lesion_recall_{size}", models=models)[train]
            cols.append(m.to_numpy())
        prec = D.score_matrix(train, metric="lesion_precision", models=models)[train].to_numpy()
        ax = S.axes_mm(fig, x, top, 19.5, ROW * len(models))
        P.heat(ax, np.array(cols).T, rows=models if k == 0 else None, cols=["small", "med.", "large"], vmin=0, vmax=1, fs=5.0, cmap=S.CMAP_SEQ)
        ax.xaxis.tick_top()
        ax.tick_params(axis="x", pad=1, labelsize=5.2)
        if k == 0:
            S.family_chips(ax, fams, x=-0.045)
        ax2 = S.axes_mm(fig, x + 20.7, top, 6.5, ROW * len(models))
        P.heat(ax2, prec[:, None], cols=["prec."], vmin=0, vmax=1, fs=5.0, cmap=S.CMAP_WARM)
        ax2.xaxis.tick_top()
        ax2.tick_params(axis="x", pad=1, labelsize=5.2)
        S.text_mm(fig, f"Trained on {train}", x, top - 8.3, fontsize=6.2, fontweight="bold")
        S.text_mm(fig, "recall by lesion size", x, top - 5.4, fontsize=5.0, color=S.INK2)
        x += 31
    if D.simulated(D.main_runs("lesion_recall_small")):
        S.tag_simulated(ax2, loc="br", dy=-0.085, text="MedSAM row simulated")


def lines(ax, exp, levels, test, metric="aupr", prefix=""):
    """One line per representative model over ``levels``; a level whose rows are still placeholders is left out
    (the line is drawn through the measured levels only), and the panel is tagged when that happens."""
    r = D.sel(experiment=exp, metric=metric, test_set=test)
    sim = D.simulated(r)
    r = r[r.source == "measured"]
    per_seed = r.groupby(["model", "variant", "seed"])["value"].mean()
    mean = per_seed.groupby(["model", "variant"]).mean()
    sd = per_seed.groupby(["model", "variant"]).std()
    present = [m for m in S.REP_STYLE if m in mean.index.get_level_values(0)]
    for model in present:
        st = S.REP_STYLE[model]
        xs = [i for i, lv in enumerate(levels) if (model, prefix + lv) in mean.index]
        y = [mean[(model, prefix + levels[i])] for i in xs]
        e = [sd[(model, prefix + levels[i])] for i in xs]
        e = None if all(np.isnan(v) for v in e) else e  # one seed: no error bar
        ax.errorbar(xs, y, yerr=e, color=st["color"], marker=st["marker"], ls=st["ls"], lw=1.0, ms=3.6, mec=S.PAPER, mew=0.4, elinewidth=0.5, capsize=0, zorder=3)
    if sim:
        S.tag_simulated(ax, loc="br", text="pending")
    return present


def panel_d(fig, top):
    levels = ["512", "1440", "native patches"]
    labels = ["512", "1,440", "native\npatches"]
    specs = [("aupr", "mAUPR on DDR test", (0, 0.5)), ("lesion_recall_small", "Recall, small lesions", (0, 0.6))]
    for k, (metric, ylabel, ylim) in enumerate(specs):
        ax = S.axes_mm(fig, 104 + k * 39, top, 30, 30)
        lines(ax, "E7", levels, "DDR", metric=metric, prefix="res:")
        ax.set_xticks(range(len(levels)))
        ax.set_xticklabels(labels, fontsize=5.3)
        ax.set_xlim(-0.3, len(levels) - 0.7)
        ax.set_ylim(*ylim)
        ax.set_ylabel(ylabel)
        ax.axvspan(0.6, 1.4, color=S.SOFT, zorder=0, lw=0)
        S.light_grid(ax)
    ax.text(1, 0.585, "B-std", ha="center", va="top", fontsize=4.9, color=S.INK2)
    S.text_mm(fig, "Long side of the training and inference canvas (pixels)", 139, top + 37, ha="center", fontsize=6.2)


def panel_e(fig, top):
    levels = ["DDR 10%", "DDR 50%", "DDR 100%", "fine merged"]
    labels = ["10%", "50%", "100%", "+IDRiD\n+TJDR"]
    n_img = [38, 192, 383, 599]
    for k, (test, role) in enumerate((("DDR", "internal"), ("TJDR-std", "independent until “+TJDR”"), ("FGADR", "independent"))):
        ax = S.axes_mm(fig, 16 + k * 56.5, top, 47, 28)
        lines(ax, "E8", levels, test, prefix="size:")
        ax.set_xticks(range(len(levels)))
        ax.set_xticklabels([f"{a}\n{b:,}" for a, b in zip(labels, n_img)], fontsize=4.7, linespacing=1.15)
        ax.set_xlim(-0.35, len(levels) - 0.65)
        ax.set_ylim(0, 0.7)
        ax.set_yticks([0, 0.2, 0.4, 0.6])
        ax.axvline(2.5, color=S.INK2, lw=0.5, ls=(0, (2, 2)))
        ax.set_title(f"Tested on {test} ({role})", fontsize=5.9, loc="left")
        ax.set_ylabel("mAUPR")
        if k == 0:
            ax.text(1.0, 0.67, "fraction of the DDR training set", ha="center", va="top", fontsize=4.8, color=S.INK2)
            ax.text(3.0, 0.67, "fine sources\nmerged", ha="center", va="top", fontsize=4.8, color=S.INK2, linespacing=1.1)
        S.light_grid(ax)
    S.text_mm(fig, "Second line: number of training images. Every setting uses the same number of image passes; U-Net and M2MRF, seed 0.", 16, top + 37, fontsize=4.9, color=S.INK2)


def main():
    models, _blocks, _fr = D.ranked_models()
    fams = [D.FAMILY_OF[m] for m in models]
    c_bottom = 78 + ROW * len(models)          # panel c grows with the number of main-table rows (19 since 2026-10)
    e_label, e_top = c_bottom + 6, c_bottom + 16
    fig = S.figure(e_top + 43)
    S.label_mm(fig, "a", 2, 2)
    S.text_mm(fig, "Lesion size by class", 8, 2.6, fontsize=6.8, fontweight="bold")
    panel_a(fig, 11)
    S.label_mm(fig, "b", 72, 2)
    S.text_mm(fig, "Lesions that vanish when the image is shrunk", 78, 2.6, fontsize=6.8, fontweight="bold")
    panel_b(fig, 12)
    S.label_mm(fig, "c", 2, 61)
    S.text_mm(fig, "Lesion-level recall by size", 8, 61.6, fontsize=6.8, fontweight="bold")
    panel_c(fig, 78, models, fams)
    S.label_mm(fig, "d", 94, 61)
    S.text_mm(fig, "Input resolution", 100, 61.6, fontsize=6.8, fontweight="bold")
    panel_d(fig, 78)
    S.rep_legend(fig, loc="upper left", bbox=tuple(S.rect_mm(fig, 98, 67.5, 1, 1)[:2]), ncol=3)
    S.label_mm(fig, "e", 2, e_label)
    S.text_mm(fig, "Training amount and merged sources", 8, e_label + 0.6, fontsize=6.8, fontweight="bold")
    panel_e(fig, e_top)
    S.save(fig, "fig7_scale")


if __name__ == "__main__":
    main()
