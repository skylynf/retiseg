"""Fig. 4 | Performance of retrained models under the unified protocol (E2, E3a).

a  internal tests, per-class AUPR and mAUPR with seed s.d.
b  every model on the eight test sets, by training source
c  internal against independent tests, one line per model
d  model families
e  critical-difference diagram over the eight test sets

The panel tops follow the number of main-table models (19 since the
fundus-specific expansion, docs/paper/18), so the row blocks never overlap.
f  published rank against retrained rank on IDRiD
"""

import numpy as np
from matplotlib.lines import Line2D
from scipy import stats as sps

from retifig import data as D
from retifig import panels as P
from retifig import style as S

ROW = 3.05  # mm per model row


def order_models():
    return D.ranked_models()   # measured models by average rank, then the unmeasured ones (placeholder rows)


def panel_a(fig, models, fams, top):
    internal = D.internal()
    ms = D.mean_sd()
    sim = D.simulated(D.main_runs())
    x = 30.0
    for k, train in enumerate(D.TRAIN_MAIN):
        M = internal.xs(train, level="train_set").loc[models].to_numpy()
        ax = S.axes_mm(fig, x, top, 24, ROW * len(models))
        P.heat(ax, M, rows=models if k == 0 else None, cols=D.CLASSES, vmin=0, vmax=0.9, bold_max="col")
        ax.xaxis.tick_top()
        ax.tick_params(axis="x", pad=1)
        if k == 0:
            S.family_chips(ax, fams)
        n_img = {"IDRiD": 27, "DDR": 225}[train]
        S.text_mm(fig, f"Trained on {train} · internal test (n = {n_img} images)", x, top - 9.6, fontsize=6.5, fontweight="bold")
        S.text_mm(fig, "AUPR per lesion class", x, top - 6.4, fontsize=5.6, color=S.INK2)
        sub = ms[(ms.train_set == train) & (ms.test_set == train)].set_index("model").loc[models]
        axb = S.axes_mm(fig, x + 25.5, top, 30, ROW * len(models))
        P.score_bars(axb, sub["mean"].to_numpy(), [S.FAMILY[f] for f in fams], xmax=0.9, err=sub["sd"].to_numpy())
        axb.set_xticks([0, 0.3, 0.6, 0.9])
        axb.xaxis.tick_top()
        axb.spines["bottom"].set_visible(False)
        axb.spines["top"].set_visible(True)
        axb.tick_params(axis="x", pad=1)
        S.text_mm(fig, "mAUPR (mean over seeds; s.d. where 3 seeds)", x + 25.5, top - 6.4, fontsize=5.6, color=S.INK2)
        if sim:
            S.tag_simulated(axb, loc="br")
        x += 72
    P.colorbar(fig, S.rect_mm(fig, 168, top + 4, 2.2, 14), S.CMAP_SEQ, 0, 0.9, "AUPR", ticks=[0, 0.3, 0.6, 0.9], orientation="vertical")


def panel_b(fig, models, fams, top):
    sim = D.simulated(D.main_runs())
    x = 30.0
    for k, train in enumerate(D.TRAIN_MAIN):
        tests = [train] + [t for t in D.TESTS if t != train]
        M = D.score_matrix(train, models=models)[tests].to_numpy()
        ax = S.axes_mm(fig, x, top, 45, ROW * len(models))
        P.dot_matrix(ax, M, rows=models if k == 0 else None, cols=[S.DATASET_SHORT[t] for t in tests], vmin=0, vmax=0.7, size=19, col_fs=4.2)
        if k == 0:
            S.family_chips(ax, fams, x=-0.03)
        ax.axvline(1, color=S.INK, lw=0.6)
        ax.axvline(2, color=S.INK, lw=0.6, ls=(0, (2, 2)))
        spans = [(0, 1, "Int.", "#3d3d3a"), (1, 2, "Cross", "#8f8d86"), (2, len(tests), "Independent", "#d9d7cf")]
        P.header_band(ax, spans, y=1.015, h=0.055)
        trans = ax.get_xaxis_transform()
        for j, t in enumerate(tests):
            ax.plot([j + 0.18, j + 0.82], [-0.135, -0.135], transform=trans, color=S.TIER[S.DATASET_TIER[t]], lw=2.0, clip_on=False, solid_capstyle="butt")
        S.text_mm(fig, f"Trained on {train}", x, top - 7.4, fontsize=6.5, fontweight="bold")
        int_s, ind_fine = D.internal_vs_independent(train, models=models, tests=D.FINE_INDEPENDENT)
        A = np.c_[int_s.to_numpy(), ind_fine.to_numpy(), (ind_fine - int_s).to_numpy()]
        ax2 = S.axes_mm(fig, x + 46.5, top, 14, ROW * len(models))
        P.heat(ax2, A[:, :2], cols=["Int.", "Ind.\nfine"], vmin=0, vmax=0.7, fs=5.0)
        ax2.tick_params(axis="x", labelsize=4.9)
        ax3 = S.axes_mm(fig, x + 61.2, top, 7, ROW * len(models))
        P.heat(ax3, A[:, 2:], cols=["Δ"], cmap=S.CMAP_DIV, vmin=-0.4, vmax=0.4, fmt="{:+.2f}", fs=5.0)
        ax3.tick_params(axis="x", labelsize=5.4)
        if sim:
            S.tag_simulated(ax3, loc="tr", dy=0.02)
        x += 72
    P.colorbar(fig, S.rect_mm(fig, 171.5, top + 4, 2.2, 14), S.CMAP_SEQ, 0, 0.7, "mAUPR", ticks=[0, 0.35, 0.7], orientation="vertical")
    P.colorbar(fig, S.rect_mm(fig, 171.5, top + 27, 2.2, 14), S.CMAP_DIV, -0.4, 0.4, "Δ", ticks=[-0.4, 0, 0.4], orientation="vertical")
    # granularity key
    W, H = fig.get_size_inches() / S.MM
    y = top + ROW * len(models) + 9.8
    S.text_mm(fig, "Annotation granularity of the test set", 30, y, fontsize=5.4, color=S.INK2)
    xx = 74.0
    for tier in ("fine", "mixed", "coarse", "region"):
        fig.add_artist(Line2D([xx / W, (xx + 4) / W], [1 - (y + 1.15) / H] * 2, color=S.TIER[tier], lw=2.0))
        S.text_mm(fig, S.TIER_LABEL[tier], xx + 5, y, fontsize=5.4, color=S.INK2)
        xx += 8 + 1.05 * len(S.TIER_LABEL[tier]) + 3


def panel_c(fig, models, fams, top):
    cols = [S.FAMILY[f] for f in fams]
    x = 12.0
    pvals, axes = [], []
    for train in D.TRAIN_MAIN:
        m = D.score_matrix(train, models=models)
        keys = [train, "TJDR-std", "TJDR-uwf", "MAPLES-DR", "FGADR", "Retinal-Lesions"]
        ax = S.axes_mm(fig, x, top, 47, 36)
        P.paired_slope(ax, [m[k].to_numpy() for k in keys], cols, ["Internal", "TJDR\nstd", "TJDR\nUWF", "Maples", "FGADR", "Ret.-\nLesions"])
        ax.set_ylim(0, 0.9)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
        ax.set_ylabel("mAUPR")
        ax.axvspan(-0.45, 0.5, color=S.SOFT, zorder=0, lw=0)
        ax.set_title(f"Trained on {train}", fontsize=6.5, fontweight="bold", loc="left", pad=6)
        ax.tick_params(axis="x", labelsize=5.2)
        S.light_grid(ax)
        for j in (1, 4):
            pvals.append(sps.wilcoxon(m[train].to_numpy(), m[keys[j]].to_numpy()).pvalue)
        axes.append(ax)
        x += 57
    adj = D.bstats.holm(pvals)
    i = 0
    for ax in axes:
        for j, yb in ((1, 0.78), (4, 0.86)):
            P.bracket(ax, 0, j, yb, P.p_text(adj[i]), h=0.012)
            i += 1
        if D.simulated(D.main_runs()):
            S.tag_simulated(ax, loc="br")


def panel_d(fig, top):
    left = 133.0
    ms = D.mean_sd()
    ms["family"] = ms["model"].map(D.FAMILY_OF)
    internal = ms[ms.train_set == ms.test_set]
    indep = ms[ms.test_set.isin(D.FINE_INDEPENDENT)].groupby(["model", "train_set", "family"], sort=False)["mean"].mean().reset_index()
    colors = [S.FAMILY[f] for f in S.FAMILY_ORDER]
    for k, (name, tab) in enumerate((("Internal", internal), ("Independent, fine", indep))):
        ax = S.axes_mm(fig, left + k * 22.5, top, 20, 36)
        groups = [tab[tab.family == f]["mean"].to_numpy() for f in S.FAMILY_ORDER]
        P.strip_mean_sd(ax, groups, colors, labels=[S.FAMILY_SHORT[f] for f in S.FAMILY_ORDER], size=8)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")
        ax.set_ylim(0, 0.9)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
        ax.tick_params(axis="x", labelsize=5.2)
        S.light_grid(ax)
        ax.set_title(name, fontsize=6.2, loc="left", pad=6)
        if k == 0:
            ax.set_ylabel("mAUPR")
        else:
            ax.set_yticklabels([])
            if D.simulated(D.main_runs()):
                S.tag_simulated(ax, loc="br")
        h = sps.kruskal(*groups)
        ax.text(0.5, 0.965, f"K–W {P.p_text(h.pvalue)}", transform=ax.transAxes, fontsize=4.9, color=S.INK2, va="top", ha="center")


def panel_e(fig, blocks, fr, top):
    ax = S.axes_mm(fig, 30, top, 66, 44)
    short = [m.replace(" (full FT)", "").replace("RETFound + UPerNet", "RETFound+UPerNet") for m in blocks.index]
    cols = [S.FAMILY[D.FAMILY_OF[m]] for m in blocks.index]
    P.cd_diagram(ax, short, fr["avg_rank"], fr["critical_difference"], cols)
    S.text_mm(fig, f"Friedman χ² = {fr['statistic']:.1f}, {P.p_text(fr['p'])}; Nemenyi critical difference at α = 0.05; n = {len(D.TESTS)} test sets",
              61, top + 45.5, ha="center", fontsize=5.2, color=S.INK2)
    if D.simulated(D.main_runs()):
        S.tag_simulated(ax, loc="tr", dy=-0.1)


def panel_f(fig, top):
    pub_rows = D.sel(experiment="L1", variant="published_idrid", metric="reported_maupr")
    pub_rows = pub_rows[pub_rows.model.isin(D.measured_models())]   # a published number pairs with a measured retraining only
    pub_sim = D.simulated(pub_rows)
    pub = pub_rows.set_index("model")["value"]
    ret = D.score_matrix("IDRiD", models=list(pub.index))["IDRiD"]
    ra = sps.rankdata(-pub.to_numpy())
    rb = sps.rankdata(-ret.to_numpy())
    tau = D.bstats.kendall_tau(pub.to_numpy(), ret.to_numpy())
    ax = S.axes_mm(fig, 137, top + 6, 22, 33)
    cols = [S.FAMILY[D.FAMILY_OF[m]] for m in pub.index]
    P.bump(ax, list(pub.index), ra, rb, cols, labels=("Published", "Retrained"))
    S.text_mm(fig, f"Rank on IDRiD (1 = best) · Kendall τ = {tau['tau']:.2f}, {P.p_text(tau['p'])}, n = {len(pub)}",
              148, top + 45.5, ha="center", fontsize=5.2, color=S.INK2)
    if pub_sim or D.simulated(D.main_runs()):
        S.tag_simulated(ax, loc="tr", dx=0.75, dy=0.1)


def main():
    models, blocks, fr = order_models()
    fams = [D.FAMILY_OF[m] for m in models]
    block = ROW * len(models)                 # height of one model-row block (42.7 mm for 14 models, 58 for 19)
    ta = 15
    tb = ta + block + 17                      # headers of b sit 9.5 mm above its block
    tc = tb + block + 24                      # x labels, the granularity key, then c's title
    te = tc + 36 + 14
    fig = S.figure(te + 48)
    S.label_mm(fig, "a", 2, ta - 11.5)
    panel_a(fig, models, fams, ta)
    S.label_mm(fig, "b", 2, tb - 9.5)
    panel_b(fig, models, fams, tb)
    S.label_mm(fig, "c", 2, tc - 7.5)
    panel_c(fig, models, fams, tc)
    S.label_mm(fig, "d", 124, tc - 7.5)
    panel_d(fig, tc)
    S.label_mm(fig, "e", 2, te - 2)
    panel_e(fig, blocks, fr, te)
    S.label_mm(fig, "f", 116, te - 2)
    panel_f(fig, te)
    P.family_legend(fig, S.FAMILY_ORDER, loc="upper right", ncol=4, bbox=(0.99, 0.999))
    S.save(fig, "fig4_retrained")


if __name__ == "__main__":
    main()
