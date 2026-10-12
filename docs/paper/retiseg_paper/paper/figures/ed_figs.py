"""Extended Data figures that need no images.

ED6   source recipes (B1) against the unified protocol (B-std)
ED7   learning-rate sweep (E2c): does a tuned baseline change the ranking?
ED8   loss (E9), U-Net only; the vessel-input panel (E6) was removed with docs/paper/16
ED10  computational cost (E10)
"""

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from retifig import data as D
from retifig import panels as P
from retifig import style as S


def ed6():
    fig = S.figure(62, width=S.FULL_W)
    b1 = D.sel(experiment="E2b", metric="aupr").groupby(["model", "train_set"])["value"].mean().unstack("train_set")
    models = list(b1.index)
    for k, train in enumerate(D.TRAIN_MAIN):
        bs = D.score_matrix(train, models=models)[train]
        order = bs.sort_values(ascending=False).index
        ax = S.axes_mm(fig, 30 + k * 76, 14, 46, 36)
        P.dumbbell(ax, list(order), [bs[order].to_numpy(), b1.loc[order, train].to_numpy()], ["#1d3f5e", "#eda100"],
                   ["B-std: shared budget, augmentation and selection (mean of 3 seeds)", "B1: source recipe with minimal adaptation (seed 0)"],
                   markers=["o", "s"], xlim=(0, 0.8), size=20, fs=5.8)
        for i, m in enumerate(order):
            ax.text(0.795, i, f"{b1.loc[m, train] - bs[m]:+.2f}", ha="right", va="center", fontsize=5.2, color=S.INK2)
        ax.text(0.795, -0.85, "B1 − B-std", ha="right", va="center", fontsize=4.9, color=S.INK2)
        ax.set_xlabel(f"mAUPR on {train} test (internal)")
        ax.set_title(f"Trained on {train}", fontsize=6.5, fontweight="bold", loc="left", pad=8)
        S.family_chips(ax, [D.FAMILY_OF[m] for m in order], x=-0.03)
        ax.set_ylim(len(order) - 0.5, -0.5)
        # family_chips assumes rows at i + 0.5; shift to dumbbell rows at i
        ax.collections[-1].set_offsets(np.c_[np.full(len(order), -0.03), np.arange(len(order))])
        if k == 0:
            ax.legend(loc="upper left", bbox_to_anchor=S.rect_mm(fig, 28, 1.5, 1, 1)[:2], bbox_transform=fig.transFigure, ncol=2, columnspacing=1.5, handletextpad=0.2, fontsize=5.6)
    S.tag_simulated(ax, loc="tr", dy=0.06)
    S.save(fig, "ed06_b1_vs_bstd")


def ed7():
    fig = S.figure(56, width=S.FULL_W)
    r = D.sel(experiment="E2c", metric="aupr")
    r = r.assign(level=r["variant"].str.replace("lr_x", ""))
    levels = ["1/3", "1", "3"]
    for k, train in enumerate(D.TRAIN_MAIN):
        sub = r[r.train_set == train].set_index(["model", "level"])["value"]
        ax = S.axes_mm(fig, 16 + k * 62, 12, 46, 34)
        for model, st in S.REP_STYLE.items():
            y = [sub[(model, lv)] for lv in levels]
            ax.plot(range(3), y, color=st["color"], marker=st["marker"], ls=st["ls"], lw=1.0, ms=3.8, mec=S.PAPER, mew=0.4, zorder=3)
        ax.set_xticks(range(3))
        ax.set_xticklabels(["× 1/3", "× 1\n(recipe)", "× 3"])
        ax.set_xlim(-0.3, 2.3)
        ax.axvspan(0.75, 1.25, color=S.SOFT, zorder=0, lw=0)
        ax.set_ylim(0, 0.8)
        ax.set_ylabel(f"Validation mAUPR, {train}")
        ax.set_xlabel("Learning rate relative to the author recipe")
        ax.set_title(f"Trained on {train}", fontsize=6.5, fontweight="bold", loc="left")
        S.light_grid(ax)
        base = np.argsort(np.argsort([-sub[(m, "1")] for m in D.REPRESENTATIVE]))
        best = np.argsort(np.argsort([-max(sub[(m, lv)] for lv in levels) for m in D.REPRESENTATIVE]))
        tau = D.bstats.kendall_tau(base, best)["tau"]
        ax.text(0.98, 0.04, f"rank at recipe vs best of three: Kendall τ = {tau:.2f}", transform=ax.transAxes, ha="right", fontsize=5.0, color=S.INK2)
    S.tag_simulated(ax, loc="tr")
    S.rep_legend(fig, loc="center left", bbox=tuple(S.rect_mm(fig, 136, 30, 1, 1)[:2]), ncol=1)
    S.save(fig, "ed07_lr_sweep")


def ed8():
    """E9: U-Net with four losses, architecture fixed (docs/paper/16: one model, seed 0)."""
    fig = S.figure(64, width=S.FULL_W)
    S.label_mm(fig, "a", 2, 2)
    S.text_mm(fig, "Loss, architecture fixed (E9, U-Net)", 8, 2.6, fontsize=6.8, fontweight="bold")
    losses = ["CE", "Dice", "CE + Dice", "CE + per-image Dice"]
    labels = ["CE", "batch\nDice", "CE +\nbatch Dice", "CE +\nper-image Dice"]
    r = D.sel(experiment="E9", metric="aupr")
    per_seed = r.groupby(["model", "variant", "seed"])["value"].mean()
    mean, sd = per_seed.groupby(["model", "variant"]).mean(), per_seed.groupby(["model", "variant"]).std()
    models = [m for m in S.REP_STYLE if m in mean.index.get_level_values(0)]
    ax = S.axes_mm(fig, 16, 13, 50, 36)
    for model in models:
        st = S.REP_STYLE[model]
        y = [mean[(model, f"loss:{lv}")] for lv in losses]
        e = [sd[(model, f"loss:{lv}")] for lv in losses]
        e = None if all(np.isnan(v) for v in e) else e
        ax.errorbar(range(4), y, yerr=e, color=st["color"], marker=st["marker"], ls=st["ls"], lw=1.0, ms=3.8, mec=S.PAPER, mew=0.4, elinewidth=0.5, zorder=3)
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels, fontsize=5.3, linespacing=1.1)
    ax.set_xlim(-0.3, 3.3)
    ax.set_ylim(0, 0.55)
    ax.axvspan(1.6, 2.4, color=S.SOFT, zorder=0, lw=0)
    ax.text(2, 0.54, "B-std", ha="center", va="top", fontsize=4.9, color=S.INK2)
    ax.set_ylabel("mAUPR on DDR test")
    S.light_grid(ax)
    S.tag_simulated(ax, loc="br")
    # per-class effect relative to CE
    S.label_mm(fig, "b", 76, 2)
    S.text_mm(fig, "Per-class change against cross-entropy", 82, 2.6, fontsize=6.8, fontweight="bold")
    cls_mean = r.groupby(["model", "variant", "cls"])["value"].mean()
    x0 = 98.0
    for j, lv in enumerate(losses[1:]):
        M = np.array([[cls_mean[(m, f"loss:{lv}", c)] - cls_mean[(m, "loss:CE", c)] for c in D.CLASSES] for m in models])
        axh = S.axes_mm(fig, x0 + j * 24, 16, 17, 6 * len(models))
        P.heat(axh, M, rows=models if j == 0 else None, cols=D.CLASSES, cmap=S.CMAP_DIV, vmin=-0.3, vmax=0.3, fmt="{:+.2f}", fs=4.6)
        axh.tick_params(labelsize=5.0)
        axh.set_title(labels[j + 1].replace("\n", " "), fontsize=5.8, loc="left", pad=3)
    S.tag_simulated(axh, loc="tr", dy=0.2)
    P.colorbar(fig, S.rect_mm(fig, 98, 46, 22, 1.5), S.CMAP_DIV, -0.3, 0.3, "Δ AUPR against CE", ticks=[-0.3, 0, 0.3])
    S.text_mm(fig, "The per-image Dice term (smooth 1) of the registered U-Net recipe drives the rare-class\nchannels to zero under patch training; the main table uses batch Dice (Methods).", 98, 54, fontsize=4.9, color=S.INK2, linespacing=1.2)
    S.save(fig, "ed08_loss")


def ed10():
    """E10. Training time and parameter counts come from environment.json / data/cost_measured.tsv
    where measured; FLOPs, memory and inference time are placeholders until bench/scripts/cost_audit.py runs."""
    fig = S.figure(74, width=S.FULL_W)
    c = D.sel(experiment="E10")
    get = lambda metric, train="": c[(c.metric == metric) & (c.train_set == train)].set_index("model")["value"]
    params, hours, mem, sec, flops = get("params_m"), get("train_gpu_h", "IDRiD"), get("peak_mem_gb"), get("infer_s_per_image"), get("flops_g")
    trainable = get("params_trainable_m")
    score = D.score_matrix("IDRiD")["IDRiD"]
    models = [m for m in score.sort_values(ascending=False).index if m in hours.index and m in params.index]
    size = lambda p: 6 + 1.6 * np.sqrt(p)
    S.label_mm(fig, "a", 2, 2)
    S.text_mm(fig, "Accuracy against training cost", 8, 2.6, fontsize=6.8, fontweight="bold")
    ax = S.axes_mm(fig, 16, 13, 62, 48)
    for m in models:
        ax.scatter([hours[m]], [score[m]], s=size(params[m]), color=S.FAMILY[D.FAMILY_OF[m]], alpha=0.85, edgecolors=S.PAPER, linewidths=0.5, zorder=3)
    nudge = {"nnU-Net": (-4, 6, "right"), "nnU-Net ResEnc-M": (5, 4, "left"), "M2MRF": (5, 6, "left"), "HRNet-W48": (-4, -8, "right"), "SAM2-UNet": (7, 0, "left"),
             "U-Net": (5, -2, "left"), "FCT": (-5, -7, "right"), "Swin-Unet": (3, -8, "center"), "DeepLabv3": (-5, 4, "right"), "RETFound + UPerNet": (7, 1, "left"),
             "HACDR-Net": (0, 8, "center"), "SAMed": (-6, 4, "right"), "MedSAM (full FT)": (6, 5, "left"), "SAM-Adapter": (-7, -3, "right")}
    for m in models:
        dx, dy, ha = nudge.get(m, (5, 0, "left"))
        ax.annotate(m, (hours[m], score[m]), xytext=(dx, dy), textcoords="offset points", fontsize=5.0, ha=ha, va="center")
    ax.set_xscale("log")
    ax.set_xlim(0.8, 40)
    ax.set_xticks([1, 2, 5, 10, 20])
    ax.set_xticklabels(["1", "2", "5", "10", "20"])
    ax.minorticks_off()
    ax.set_ylim(0.2, 0.8)
    ax.set_xlabel("Training time on IDRiD, one seed (A100 GPU hours)")
    ax.set_ylabel("mAUPR on IDRiD test")
    ax.grid(color=S.GRID, lw=0.4, zorder=0)
    h = [Line2D([], [], marker="o", ls="", ms=np.sqrt(size(p)), color=S.MUTED, label=f"{p} M") for p in (10, 100, 600)]
    ax.legend(handles=h, title="Parameters", title_fontsize=5.2, loc="lower right", fontsize=5.2, labelspacing=1.1, borderpad=0.6, handletextpad=0.8)
    if D.simulated(c[c.metric.isin(["train_gpu_h", "params_m"]) & c.model.isin(models)]) or D.simulated(D.main_runs()):
        S.tag_simulated(ax, loc="tl", text="mAUPR and unmarked hours simulated")
    P.family_legend(fig, S.FAMILY_ORDER, loc="upper left", ncol=4, bbox=tuple(S.rect_mm(fig, 14, 68, 1, 1)[:2]), marker="o")
    # aligned bars
    S.label_mm(fig, "b", 88, 2)
    S.text_mm(fig, "Resources per model", 94, 2.6, fontsize=6.8, fontweight="bold")
    tr = pd.Series({m: trainable.get(m, params[m]) for m in models})
    cols = [(params, "Parameters (M)", "{:.0f}", 720), (tr, "Trainable (M)", "{:.0f}", 720), (flops, "FLOPs (G)", "{:.0f}", 700),
            (mem, "Peak training\nmemory (GB)", "{:.0f}", 46), (sec, "Inference\n(s per image)", "{:.1f}", 5.6)]
    row = 3.05
    for j, (series, title, fmt, xmax) in enumerate(cols):
        axb = S.axes_mm(fig, 112 + j * 13.5, 17, 10, row * len(models))
        P.score_bars(axb, series.reindex(models).to_numpy(), [S.FAMILY[D.FAMILY_OF[m]] for m in models], xmax=xmax, fmt=fmt, fs=4.9)
        axb.set_xticks([])
        axb.spines["bottom"].set_visible(False)
        axb.set_title(title, fontsize=5.2, loc="left", pad=3, linespacing=1.1)
        if j == 0:
            axb.set_yticks(np.arange(len(models)) + 0.5)
            axb.set_yticklabels(models, fontsize=5.4)
            axb.tick_params(axis="y", length=0)
        if j == 0:
            # mark measured parameter counts
            meas = c[(c.metric == "params_m") & (c.source == "measured")].model
            for i, m in enumerate(models):
                if m in set(meas):
                    axb.text(-0.02, i + 0.5, "*", transform=axb.get_yaxis_transform(), ha="right", va="center", fontsize=4.6, color=S.INK2)
    S.text_mm(fig, "Models ordered by mAUPR on the IDRiD test set. 40 GB A100, one task per card. * parameter count measured;\nFLOPs, memory and inference time are placeholders until the cost audit runs.", 94, 63, fontsize=4.9, color=S.INK2, linespacing=1.2)
    S.tag_simulated(axb, loc="br", dy=-0.1, text="FLOPs, memory, inference simulated")
    S.save(fig, "ed10_cost")


if __name__ == "__main__":
    # ed6 (B1 against B-std) is withdrawn as a figure (docs/paper/16); the function stays for the supplementary table.
    ed7()
    ed8()
    ed10()
