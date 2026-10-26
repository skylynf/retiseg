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
    """E2c. The sweep that ran: HRNet-W48 and M2MRF on DDR at 1/3, 1 and 3 times the recipe learning rate, judged on
    validation mAUPR (docs/ops/00 2026-10-12: DDR only). Only measured (model, source) pairs are drawn; the registered
    IDRiD half and the other representative models were cut and are not shown as placeholders."""
    r = D.sel(experiment="E2c", metric="aupr")
    r = r[r.source == "measured"].assign(level=r["variant"].str.replace("lr_x", ""))
    levels = ["1/3", "1", "3"]
    sources = [t for t in D.TRAIN_MAIN if (r.train_set == t).any()]
    fig = S.figure(56, width=S.FULL_W)
    ax = None
    for k, train in enumerate(sources):
        sub = r[r.train_set == train].set_index(["model", "level"])["value"]
        models = [m for m in dict.fromkeys(r[r.train_set == train].model) if all((m, lv) in sub.index for lv in levels)]
        ax = S.axes_mm(fig, 16 + k * 62, 12, 46, 34)
        for model in models:
            st = S.REP_STYLE.get(model, {"color": S.FAMILY[D.FAMILY_OF[model]], "marker": "o", "ls": "-"})
            y = [sub[(model, lv)] for lv in levels]
            ax.plot(range(3), y, color=st["color"], marker=st["marker"], ls=st["ls"], lw=1.0, ms=3.8, mec=S.PAPER, mew=0.4, zorder=3, label=model)
            ax.text(2.08, y[-1], model, fontsize=5.2, color=S.INK, va="center", ha="left", clip_on=False)
        ax.set_xticks(range(3))
        ax.set_xticklabels(["× 1/3", "× 1\n(recipe)", "× 3"])
        ax.set_xlim(-0.3, 2.3)
        ax.axvspan(0.75, 1.25, color=S.SOFT, zorder=0, lw=0)
        ax.set_ylim(0.4, 0.7)
        ax.set_ylabel(f"Validation mAUPR, {train}")
        ax.set_xlabel("Learning rate relative to the author recipe")
        ax.set_title(f"Trained on {train} (validation split; the test split is not read)", fontsize=6.5, fontweight="bold", loc="left")
        S.light_grid(ax)
        base = np.argsort(np.argsort([-sub[(m, "1")] for m in models]))
        best = np.argsort(np.argsort([-max(sub[(m, lv)] for lv in levels) for m in models]))
        tau = D.bstats.kendall_tau(base, best)["tau"] if len(models) > 1 else float("nan")
        ax.text(0.98, 0.04, f"rank at recipe vs best of three: Kendall τ = {tau:.2f}", transform=ax.transAxes, ha="right", fontsize=5.0, color=S.INK2)
    if ax is not None and D.simulated(D.sel(experiment="E2c", metric="aupr", train_set=sources)):
        S.tag_simulated(ax, loc="tr")
    S.text_mm(fig, "The registered sweep also had the IDRiD half and three further representative models; they were cut when the scope was fixed (docs/paper/16) and are not shown.",
              16, 52, fontsize=4.9, color=S.INK2)
    S.save(fig, "ed07_lr_sweep")


def ed8():
    """E9: U-Net with four losses, architecture fixed (docs/paper/16: one model, seed 0)."""
    fig = S.figure(64, width=S.FULL_W)
    S.label_mm(fig, "a", 2, 2)
    S.text_mm(fig, "Loss, architecture fixed (U-Net)", 8, 2.6, fontsize=6.8, fontweight="bold")
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
    ax.text(2, 0.54, "unified\nprotocol", ha="center", va="top", fontsize=4.6, color=S.INK2, linespacing=1.1)
    ax.set_ylabel("mAUPR on DDR test")
    S.light_grid(ax)
    if D.simulated(r):
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
    if D.simulated(r):
        S.tag_simulated(axh, loc="tr", dy=0.2)
    P.colorbar(fig, S.rect_mm(fig, 98, 46, 22, 1.5), S.CMAP_DIV, -0.3, 0.3, "Δ AUPR against CE", ticks=[-0.3, 0, 0.3])
    S.text_mm(fig, "The per-image Dice term (smooth 1) of the registered U-Net recipe drives the rare-class\nchannels to zero under patch training; the main table uses batch Dice (Methods).", 98, 54, fontsize=4.9, color=S.INK2, linespacing=1.2)
    S.save(fig, "ed08_loss")


def ed9():
    """E3a. a: DiaRetDB1 under its official image-level protocol (ROC AUC of the maximum probability inside the field
    of view, per class), every main-table model from both sources. b: the ultra-widefield TJDR images predicted at
    native resolution with sliding windows (the main variant) against the same images resized to the 1,440 canvas."""
    fig = S.figure(96, width=S.FULL_W)
    models = list(D.ranked_models()[0])
    # --- a: image-level AUC per class on DiaRetDB1
    S.label_mm(fig, "a", 2, 2)
    S.text_mm(fig, "DiaRetDB1, official image-level protocol", 8, 2.6, fontsize=6.8, fontweight="bold")
    row_h = 3.3
    for k, train in enumerate(D.TRAIN_MAIN):
        M = np.full((len(models), 4), np.nan)
        sim = False
        for i, m in enumerate(models):
            for j, cls in enumerate(D.CLASSES):
                r = D.sel(experiment=["E2", "E5"], variant=D.MAIN_VARIANTS, model=m, train_set=train, test_set="DiaRetDB1", metric="img_auc", cls=cls, seed=0)
                if len(r):
                    M[i, j] = float(r["value"].mean())
                    sim |= D.simulated(r)
        ax = S.axes_mm(fig, 34 + k * 24, 13, 20, row_h * len(models))
        P.heat(ax, M, rows=models if k == 0 else None, cols=D.CLASSES, vmin=0.5, vmax=1.0, fs=4.6)
        ax.tick_params(axis="x", labelsize=5.0)
        P.header_band(ax, [(0, 4, f"trained on {train}", S.TIER["fine"])], y=1.03, h=0.03, fs=5.2)
        if sim:
            S.tag_simulated(ax, loc="br")
    P.colorbar(fig, S.rect_mm(fig, 34, 13 + row_h * len(models) + 6, 40, 1.5), S.CMAP_SEQ, 0.5, 1.0, "image-level ROC AUC (maximum probability inside the FOV)", ticks=[0.5, 0.75, 1.0])
    # --- b: TJDR-uwf native vs 1,440 canvas
    S.label_mm(fig, "b", 92, 2)
    S.text_mm(fig, "Ultra-widefield TJDR at two inference scales", 98, 2.6, fontsize=6.8, fontweight="bold")
    for k, train in enumerate(D.TRAIN_MAIN):
        native, fov, rows_ok = [], [], []
        for m in models:
            a = D.sel(experiment=["E2", "E5"], variant=D.MAIN_VARIANTS, model=m, train_set=train, test_set="TJDR-uwf", metric="aupr", seed=0)
            b = D.sel(experiment=["E2", "E5"], variant=[v + ":fov" for v in D.MAIN_VARIANTS], model=m, train_set=train, test_set="TJDR-uwf", metric="aupr", seed=0)
            a = a[a.cls.isin(D.CLASSES)]; b = b[b.cls.isin(D.CLASSES)]
            native.append(float(a["value"].mean()) if len(a) else np.nan)
            fov.append(float(b["value"].mean()) if len(b) else np.nan)
        ax = S.axes_mm(fig, 108 + k * 38, 13, 30, row_h * len(models))
        P.dumbbell(ax, models if k == 0 else [""] * len(models), [native, fov], ["#1d3f5e", "#eda100"],
                   ["native resolution, sliding windows (main)", "resized to the 1,440-pixel canvas"], markers=["o", "s"], xlim=(0, 0.45), size=14, fs=5.0)
        ax.set_ylim(len(models) - 0.5, -0.5)
        ax.set_xlabel("mAUPR, TJDR ultra-widefield")
        ax.set_title(f"trained on {train}", fontsize=5.8, loc="left", pad=3)
        if k == 0:
            ax.legend(loc="upper left", bbox_to_anchor=S.rect_mm(fig, 106, 13 + row_h * len(models) + 6, 1, 1)[:2], bbox_transform=fig.transFigure, ncol=1, handletextpad=0.2, labelspacing=0.3, fontsize=5.2)
        elif ax.get_legend():
            ax.get_legend().remove()
    S.text_mm(fig, "Models in the order of their average rank (Fig. 4e). Native resolution is the main-table setting; the resized version is the comparison variant registered for the independent tests.",
              34, 13 + row_h * len(models) + 14, fontsize=4.9, color=S.INK2)
    S.save(fig, "ed09_diaretdb1_uwf")


def ed10():
    """E10. Training time and parameter counts come from environment.json / data/cost_measured.tsv
    where measured; FLOPs, memory and inference time are placeholders until bench/scripts/cost_audit.py runs."""
    fig = S.figure(92, width=S.FULL_W)
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
    texts = [ax.text(hours[m], score[m], m, fontsize=4.8, ha="center", va="center") for m in models]
    try:
        from adjustText import adjust_text
        adjust_text(texts, ax=ax, expand=(1.15, 1.3), arrowprops=dict(arrowstyle="-", color=S.MUTED, lw=0.35), only_move={"points": "", "text": "xy"})
    except ImportError:   # fall back to the hand-placed offsets
        for t, m in zip(texts, models):
            dx, dy, ha = nudge.get(m, (5, 0, "left"))
            t.set_position((hours[m], score[m])); t.set_ha(ha)
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
    sim_hours = D.simulated(c[(c.metric == "train_gpu_h") & c.model.isin(models)])
    sim_params = D.simulated(c[(c.metric == "params_m") & c.model.isin(models)])
    if sim_hours or D.simulated(D.main_runs()):
        S.tag_simulated(ax, loc="tl", text="mAUPR or hours simulated")
    elif sim_params:
        S.tag_simulated(ax, loc="tl", text="unmarked parameter counts from the model's plan")
    P.family_legend(fig, S.FAMILY_ORDER, loc="upper left", ncol=4, bbox=tuple(S.rect_mm(fig, 14, 68, 1, 1)[:2]), marker="o")
    # aligned bars
    S.label_mm(fig, "b", 88, 2)
    S.text_mm(fig, "Resources per model", 94, 2.6, fontsize=6.8, fontweight="bold")
    tr = pd.Series({m: trainable.get(m, params[m]) for m in models})
    hours_ddr = get("train_gpu_h", "DDR")
    measured_metric = lambda metric: bool((c[(c.metric == metric) & (c.source == "measured")]).shape[0])
    cols = [(params, "Parameters (M)", "{:.0f}", 720), (tr, "Trainable (M)", "{:.0f}", 720),
            (hours, "GPU h, IDRiD\n(one seed)", "{:.1f}", 40), (hours_ddr, "GPU h, DDR\n(one seed)", "{:.0f}", 60)]
    # FLOPs, peak memory and inference time join once bench/scripts/cost_audit.py has run (17 section 4 item 65)
    for series, title, fmt, xmax, metric in ((flops, "FLOPs (G)", "{:.0f}", 700, "flops_g"), (mem, "Peak training\nmemory (GB)", "{:.0f}", 46, "peak_mem_gb"),
                                             (sec, "Inference\n(s per image)", "{:.1f}", 5.6, "infer_s_per_image")):
        if measured_metric(metric):
            cols.append((series, title, fmt, xmax))
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
    S.text_mm(fig, "Models ordered by mAUPR on the IDRiD test set. 40 GB A100, one task per card; nnU-Net follows its own\n1,000-epoch schedule. * parameter count read from the run (the others are the plan's); FLOPs, peak memory\nand inference time join when the cost audit has run.", 94, 17 + row * len(models) + 4, fontsize=4.9, color=S.INK2, linespacing=1.2)
    if D.simulated(c[(c.metric == "train_gpu_h") & c.model.isin(models)]):
        S.tag_simulated(axb, loc="br", dy=-0.1, text="some hours simulated")
    S.save(fig, "ed10_cost")


if __name__ == "__main__":
    # ed6 (B1 against B-std) is withdrawn as a figure (docs/paper/16); the function stays for the supplementary table.
    ed7()
    ed8()
    ed9()
    ed10()
