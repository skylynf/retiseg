"""Fig. 5 | Foundation models under four kinds of use (E5).

a  per-class AUPR by kind of use, on the IDRiD and DDR test sets
b  oracle prompts: performance against the number of prompts per image and class
c  SAM 3 text prompts: sensitivity to wording
d  encoder pretraining under one UPerNet head: internal against independent
e  best of each kind of use against the specialist and self-configuring references
"""

import numpy as np
from matplotlib.lines import Line2D

from retifig import data as D
from retifig import panels as P
from retifig import style as S

MODE_COLOR = {"Zero-shot": "#c8c4ee", "Upper bound": "#9085e9", "PEFT": "#6a5bc9", "Encoder + decoder": "#4a3aa7"}
PROMPT_COLOR = {"SAM": "#2a78d6", "SAM 2": "#eb6834", "MedSAM": "#1baf7a"}
ROW = 3.05


def e5_simulated(**kw):
    """True when the selection holds a placeholder row (or nothing): the panel keeps its tag until every row it draws is measured."""
    r = D.sel(experiment="E5", metric="aupr", **kw)
    return (not len(r)) or D.simulated(r)


def e5(model, variant, test, train=None):
    kw = dict(experiment="E5", model=model, variant=variant, test_set=test, metric="aupr")
    if train:
        kw["train_set"] = train
    r = D.sel(**kw)
    inference = r[r["seed"] == -1]  # the inference tiers have one run per (model, variant, test set): never average two
    if len(inference) and int(inference.groupby("cls").size().max()) > 1:
        raise ValueError(f"E5 {model} {variant} {test}: more than one inference-tier row per class; a variant must name one run directory")
    return r.groupby("cls")["value"].mean().reindex(D.CLASSES).to_numpy()


def any_lesion(model, variant, test):
    """The class-agnostic any-lesion AUPR (collect_runs metric aupr_any_lesion), the registered zero-shot
    result of the automatic masks (e5_frozen.yaml zero_shot.outputs.main). Missing rows are an error, never
    a substitute number (review N2): results.csv predates the key until build_placeholder.py --force and
    the two collects are run (docs/ops/00, the Luming list, item 8)."""
    r = D.sel(experiment="E5", model=model, variant=variant, test_set=test, metric="aupr_any_lesion", cls="any")
    if not len(r):
        raise ValueError(f"E5 {model} {variant} {test}: no aupr_any_lesion row in results.csv; regenerate the placeholder table "
                         "(python scripts/build_placeholder.py --force, then collect_runs.py and collect_literature.py) or collect the runs")
    return float(r["value"].mean())


def rows_for(test):
    """(kind of use, label, per-class AUPR, summary) per row; summary None means the mean of the four classes.

    Blocks as decided on 2026-10-14 (docs/ops/00, Luming list item 5): the zero-shot block holds only what sees
    no test label and is summarised by the any-lesion AUPR of the class-agnostic map (automatic masks carry no
    class, so their per-class cells are empty; SAM 3's per-class cells are its text prompts, its summary the
    any-lesion AUPR of the same maps); everything that reads the test labels (masks classed by overlap, oracle
    prompts from every component) is the upper-bound block. Nothing here is chosen on a test set.
    """
    blank = np.full(4, np.nan)
    out = [
        ("Zero-shot", "SAM · auto masks, class-agnostic", blank, any_lesion("SAM", "zero_shot:auto_any_lesion", test)),
        ("Zero-shot", "SAM 2 · auto masks, class-agnostic", blank, any_lesion("SAM 2", "zero_shot:auto_any_lesion", test)),
        ("Zero-shot", "SAM 3 · text, wording chosen on val", e5("SAM 3", "text:selected", test), any_lesion("SAM 3", "text:selected", test)),
        ("Upper bound", "SAM · auto masks, class by overlap†", e5("SAM", "zero_shot:auto_oracle_class", test), None),
        ("Upper bound", "SAM 2 · auto masks, class by overlap†", e5("SAM 2", "zero_shot:auto_oracle_class", test), None),
        ("Upper bound", "SAM · all boxes", e5("SAM", "oracle:box:all", test), None),
        ("Upper bound", "SAM 2 · all points", e5("SAM 2", "oracle:point:all", test), None),
        ("Upper bound", "SAM 2 · all boxes", e5("SAM 2", "oracle:box:all", test), None),
        ("Upper bound", "MedSAM · all boxes", e5("MedSAM", "oracle:box:all", test), None),
        ("PEFT", "SAMed (LoRA)", e5("SAMed", "peft", test, test), None),
        ("PEFT", "SAM-Adapter", e5("SAM-Adapter", "peft", test, test), None),
        ("PEFT", "SAM2-UNet", e5("SAM2-UNet", "peft", test, test), None),
        ("Encoder + decoder", "MedSAM · full fine-tuning", e5("MedSAM (full FT)", "encoder_decoder", test, test), None),
        ("Encoder + decoder", "RETFound + UPerNet", e5("RETFound + UPerNet", "encoder_decoder", test, test), None),
    ]
    return out


def summary_of(row):
    """The row's single number: its registered summary, otherwise the mean of its four classes."""
    return row[3] if row[3] is not None else float(np.nanmean(row[2]))


def panel_a(fig, top):
    x = 58.0
    for k, test in enumerate(("IDRiD", "DDR")):
        rows = rows_for(test)
        M = np.array([r[2] for r in rows])
        ax = S.axes_mm(fig, x, top, 22, ROW * len(rows))
        P.heat(ax, M, rows=[r[1] for r in rows] if k == 0 else None, cols=D.CLASSES, vmin=0, vmax=0.9, fs=5.0)
        ax.xaxis.tick_top()
        ax.tick_params(axis="x", pad=1)
        S.text_mm(fig, f"{test} test", x, top - 7.6, fontsize=6.4, fontweight="bold")
        axb = S.axes_mm(fig, x + 23.2, top, 15, ROW * len(rows))
        P.score_bars(axb, np.array([summary_of(r) for r in rows]), [MODE_COLOR[r[0]] for r in rows], xmax=0.9, fs=5.0)
        # the zero-shot rows are read against the any-lesion AUPR of a trained reference on the same test set
        # (decided 2026-10-14); drawn only where a measured run carries the metric
        n_zero = sum(1 for r in rows if r[0] == "Zero-shot")
        ref_any = D.reference_any_lesion("nnU-Net", test)
        if ref_any is not None:
            axb.plot([ref_any, ref_any], [0, n_zero], color=S.INK, lw=0.7, ls=(0, (1, 1.5)), zorder=4, transform=axb.transData)
            axb.text(ref_any + 0.025, 1.0, "nnU-Net\nany-lesion", fontsize=4.2, color=S.INK2, ha="left", va="center", linespacing=1.0, clip_on=False)
        axb.set_xticks([0, 0.4, 0.8])
        axb.xaxis.tick_top()
        axb.spines["bottom"].set_visible(False)
        axb.spines["top"].set_visible(True)
        axb.tick_params(axis="x", pad=1)
        S.text_mm(fig, "mAUPR", x + 23.2, top - 7.2, fontsize=5.4, color=S.INK2)
        if k == 0:
            # group brackets on the far left
            trans = ax.get_yaxis_transform()
            start = 0
            modes = [r[0] for r in rows]
            for mode in dict.fromkeys(modes):
                n = modes.count(mode)
                ax.plot([-1.84, -1.84], [start + 0.12, start + n - 0.12], transform=trans, color=MODE_COLOR[mode], lw=2.2, clip_on=False, solid_capstyle="butt")
                ax.text(-1.9, start + n / 2, mode.replace(" + ", " +\n").replace("Upper bound", "Upper\nbound"), transform=trans, ha="right", va="center", fontsize=5.4, linespacing=1.1)
                if start:
                    ax.plot([-1.86, 4.0], [start, start], transform=trans, color=S.PAPER, lw=1.6, clip_on=False)
                start += n
        elif e5_simulated(test_set=test):
            S.tag_simulated(axb, loc="tr", dy=0.1)
        x += 43
    P.colorbar(fig, S.rect_mm(fig, 146, top + 2, 2.2, 14), S.CMAP_SEQ, 0, 0.9, "AUPR", ticks=[0, 0.3, 0.6, 0.9], orientation="vertical")
    S.text_mm(fig, "Zero-shot rows see no test label; the\nbar is the any-lesion AUPR of the\nclass-agnostic map (automatic masks\ncarry no class). Density, area cap and\nSAM 3 wording were chosen on the\nvalidation pool. Upper-bound rows\nread the test labels († masks given the\nclass they overlap most; oracle\nprompts from every component) and\nare not usable settings. PEFT and\nencoder + decoder rows are trained\non the test set's source.", 146, top + 19, fontsize=4.7, color=S.INK2, linespacing=1.2)


def panel_b(fig, top):
    ks = [1, 5, 20, "all"]  # docs/paper/16 row E5b (2026-10-12): four levels
    panels = [("mean", "mAUPR"), ("MA", "MA · microaneurysm"), ("SE", "SE · soft exudate")]
    ref = D.score_matrix("IDRiD", models=["nnU-Net"])["IDRiD"].iloc[0]
    refs = D.internal().xs("IDRiD", level="train_set").loc["nnU-Net"]
    for j, (cls, title) in enumerate(panels):
        ax = S.axes_mm(fig, 13 + j * 31.5, top, 26, 27)
        for model, kind in (("SAM", "box"), ("SAM 2", "box"), ("SAM 2", "point"), ("MedSAM", "box")):  # SAM ViT-B boxes only; points for SAM 2.1 only (16)
            y = []
            for k in ks:
                v = e5(model, f"oracle:{kind}:{k}", "IDRiD")
                y.append(v.mean() if cls == "mean" else v[D.CLASSES.index(cls)])
            ax.plot(range(len(ks)), y, color=PROMPT_COLOR[model], ls="-" if kind == "box" else (0, (2.5, 1.5)), lw=1.0,
                    marker="s" if kind == "box" else "o", ms=3.0, mec=S.PAPER, mew=0.4, zorder=3)
        r = ref if cls == "mean" else refs[cls]
        ax.axhline(r, color=S.INK, lw=0.6, ls=(0, (1, 1.5)), zorder=2)
        if j == 0:
            ax.text(0.0, r + 0.02, "nnU-Net, trained", fontsize=4.8, color=S.INK2, va="bottom")
        ax.set_xticks(range(len(ks)))
        ax.set_xticklabels([str(k) for k in ks])
        ax.set_ylim(0, 1.0)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        if j:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel("AUPR on IDRiD test")
        ax.set_title(title, fontsize=6.0, loc="left")
        S.light_grid(ax)
        if j == 1:
            ax.set_xlabel("Oracle prompts per image and class")
    if any(e5_simulated(model=m, variant=f"oracle:{k}:all", test_set="IDRiD") for m, k in (("SAM", "box"), ("SAM 2", "box"), ("SAM 2", "point"), ("MedSAM", "box"))):
        S.tag_simulated(ax, loc="br")
    h = [Line2D([], [], color=PROMPT_COLOR[m], lw=1.2, label=m) for m in PROMPT_COLOR]
    h += [Line2D([], [], color=S.INK2, lw=1.0, marker="s", ms=3, label="box"), Line2D([], [], color=S.INK2, lw=1.0, ls=(0, (2.5, 1.5)), marker="o", ms=3, label="point")]
    fig.legend(handles=h, loc="upper left", bbox_to_anchor=S.rect_mm(fig, 40, top - 9.5, 1, 1)[:2], ncol=5, fontsize=5.5, columnspacing=1.2, handlelength=2.2)


def panel_c(fig, top):
    r = D.sel(experiment="E5", model="SAM 3", test_set="IDRiD", metric="aupr")
    r = r[r["variant"] != "text:selected"]  # the three registered wordings; the chosen one is panel a's row
    ax = S.axes_mm(fig, 135, top, 40, 27)
    y, labels = 0, []
    for cls in D.CLASSES:
        sub = r[r.cls == cls]
        vals = sub["value"].to_numpy()
        ax.plot([vals.min(), vals.max()], [y + 1, y + 1], color=S.AXIS, lw=0)
        for _, row in sub.iterrows():
            ax.scatter([row["value"]], [y], s=13, color=S.LESION[cls], edgecolors=S.PAPER, linewidths=0.4, zorder=3)
            ax.plot([0, row["value"]], [y, y], color=S.LESION[cls], lw=0.7, alpha=0.5, zorder=2)
            labels.append((y, row["variant"].split(":", 1)[1]))
            y += 1
        y += 0.6
    ax.set_yticks([p for p, _ in labels])
    ax.set_yticklabels([f"“{t}”" for _, t in labels], fontsize=4.9)
    ax.set_ylim(y - 0.3, -0.7)
    xmax = 0.4 if r["value"].max() < 0.38 else 0.5
    ax.set_xlim(0, xmax)
    ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4] + ([0.5] if xmax > 0.4 else []))
    ax.set_xlabel("AUPR on IDRiD test, zero-shot")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    S.light_grid(ax, "x")
    if D.simulated(r):
        S.tag_simulated(ax, loc="br")


def panel_d(fig, top):
    encs = D.ENCODERS
    note = ["ophthalmic, DINOv2", "ophthalmic, MAE", "natural images, DINOv2", "natural images, MAE"]
    enc = D.encoder_table("DDR")
    internal = enc["internal"].to_numpy()
    # an encoder whose rows are still placeholders is drawn without values (the comparison is C1 / C2 of 16: one
    # measured pair at a time, never a measured encoder against a simulated one)
    indep = np.array([enc["independent"][m] if not enc["simulated"][m] else np.nan for m in encs])
    internal = np.array([v if not enc["simulated"][m] else np.nan for v, m in zip(internal, encs)])
    sets = (f"{len(enc['sets'])} fine sets" if len(enc["sets"]) > 2 else ", ".join(s.replace("TJDR-", "TJDR ") for s in enc["sets"])) or "pending"
    ax = S.axes_mm(fig, 36, top, 44, 20)
    P.dumbbell(ax, encs, [internal, indep], ["#3d3d3a", "#9085e9"], ["internal (DDR test)", f"independent, fine ({sets})"], markers=["o", "D"], xlim=(0.1, 0.84), size=18, fs=5.6)
    for i, t in enumerate(note):
        ax.text(0.835, i, t + ("" if not enc["simulated"][encs[i]] else " (pending)"), fontsize=4.7, color=S.MUTED, ha="right", va="center")
    ax.set_xlabel("mAUPR, same UPerNet head, trained on DDR")
    ax.legend(loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=2, columnspacing=1.2, handletextpad=0.2, fontsize=5.4)
    ax.set_xticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    if any(enc["simulated"].values()):
        S.tag_simulated(ax, loc="bl", dx=0.01, dy=0.02, text="pending" if not all(enc["simulated"].values()) else "simulated")


def panel_e(fig, top):
    def best(rows_fn_test, mode):
        rows = [r for r in rows_for(rows_fn_test) if r[0] == mode]
        i = int(np.argmax([summary_of(r) for r in rows]))
        return rows[i][1].split(" · ")[0].replace(" (LoRA)", ""), summary_of(rows[i])
    cats, internal, indep, colors = [], [], [], []
    sm = D.score_matrix("DDR")
    for mode in MODE_COLOR:
        name, v = best("DDR", mode)
        cats.append(mode.replace(" + ", " +\n") + ("\n(any-lesion)" if mode == "Zero-shot" else "") + "\n" + name.replace(" + UPerNet", ""))
        internal.append(v)
        if mode in ("PEFT", "Encoder + decoder"):
            key = {"SAMed": "SAMed", "SAM-Adapter": "SAM-Adapter", "SAM2-UNet": "SAM2-UNet", "MedSAM": "MedSAM (full FT)", "RETFound + UPerNet": "RETFound + UPerNet"}[name]
            indep.append(sm.loc[key, "TJDR-std"])
        else:
            rows = [r for r in rows_for("TJDR-std") if r[0] == mode and r[1].startswith(name)]
            indep.append(max(summary_of(r) for r in rows))
        colors.append(MODE_COLOR[mode])
    ref_any = []
    for ref in ("M2MRF", "nnU-Net"):
        cats.append(f"Reference\n{ref}")
        internal.append(sm.loc[ref, "DDR"])
        indep.append(sm.loc[ref, "TJDR-std"])
        colors.append(S.FAMILY[D.FAMILY_OF[ref]])
        ref_any.append(D.reference_any_lesion(ref, "DDR"))
    ax = S.axes_mm(fig, 100, top, 76, 20)
    x = np.arange(len(cats))
    ax.bar(x - 0.19, internal, 0.34, color=colors, lw=0, zorder=2)
    ax.bar(x + 0.19, indep, 0.34, color=colors, lw=0, alpha=0.45, zorder=2)
    for xi, a, b in zip(x, internal, indep):
        ax.text(xi - 0.19, a + 0.012, f"{a:.2f}", ha="center", va="bottom", fontsize=5.0)
        ax.text(xi + 0.19, b + 0.012, f"{b:.2f}", ha="center", va="bottom", fontsize=5.0, color=S.INK2)
    # the zero-shot bar is an any-lesion AUPR (decided 2026-10-14); the references' any-lesion AUPR on the DDR test
    # is drawn as a diamond on their internal bar so the two are read on the same scale (missing where the run
    # predates the metric)
    for xi, a in zip(x[-2:], ref_any):
        if a is not None:
            ax.scatter([xi - 0.19], [a], marker="D", s=11, color=S.PAPER, edgecolors=S.INK, linewidths=0.6, zorder=5)
            ax.text(xi - 0.19, a + 0.02, f"{a:.2f}", ha="center", va="bottom", fontsize=4.6, color=S.INK2)
    ax.set_xticks(x)
    ax.set_xticklabels(cats, fontsize=4.9, linespacing=1.15)
    top_v = float(np.nanmax(internal + indep))
    ymax = 0.6 if top_v < 0.56 else (0.9 if top_v < 0.84 else 1.0)   # the oracle-box bars reach 0.77 / 0.88 (2026-10-24)
    ax.set_ylim(0, ymax)
    ax.set_yticks([0, 0.2, 0.4, 0.6] + ([0.8] if ymax > 0.6 else []) + ([1.0] if ymax > 0.9 else []))
    ax.set_ylabel("mAUPR")
    ax.axvline(3.5, color=S.INK2, lw=0.5, ls=(0, (2, 2)))
    S.light_grid(ax)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(fc=S.INK2, label="DDR test (internal for trained models)"), Patch(fc=S.INK2, alpha=0.45, label="TJDR-std test (independent)"),
                       Line2D([], [], ls="", marker="D", ms=3.2, mfc=S.PAPER, mec=S.INK, label="ref. any-lesion AUPR (DDR)")],
              loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=3, fontsize=4.8, columnspacing=0.8, handlelength=1.0, handletextpad=0.4)
    # the tag follows the rows this panel draws: the selected wording, the all-box upper bound, and the main runs of
    # the trained rows on the two test sets (the k curves of other sets stay placeholders by design, 16 row E5b)
    drawn_sim = any(e5_simulated(variant=v, test_set=t) for v in ("text:selected", "oracle:box:all") for t in ("DDR", "TJDR-std"))
    trained = D.main_runs()
    trained = trained[trained.model.isin(["SAMed", "SAM-Adapter", "SAM2-UNet", "RETFound + UPerNet", "MedSAM (full FT)", "M2MRF", "nnU-Net"])
                      & (trained.train_set == "DDR") & trained.test_set.isin(["DDR", "TJDR-std"])]
    if drawn_sim or D.simulated(trained):
        S.tag_simulated(ax, loc="tl", dx=0.01, dy=-0.2)


def main():
    fig = S.figure(148)
    S.label_mm(fig, "a", 2, 2)
    S.text_mm(fig, "Four kinds of use, reported separately", 8, 2.6, fontsize=6.8, fontweight="bold")
    panel_a(fig, 16)
    S.label_mm(fig, "b", 2, 60)
    S.text_mm(fig, "Oracle prompts", 8, 60.6, fontsize=6.8, fontweight="bold")
    panel_b(fig, 74)
    S.label_mm(fig, "c", 106, 60)
    S.text_mm(fig, "Text prompt wording (SAM 3)", 112, 60.6, fontsize=6.8, fontweight="bold")
    panel_c(fig, 69)
    S.label_mm(fig, "d", 2, 108)
    S.text_mm(fig, "Does ophthalmic pretraining help?", 8, 108.6, fontsize=6.8, fontweight="bold")
    panel_d(fig, 117)
    S.label_mm(fig, "e", 86, 108)
    S.text_mm(fig, "Best of each block against the references (zero-shot: any-lesion AUPR)", 92, 108.6, fontsize=6.8, fontweight="bold")
    panel_e(fig, 117)
    S.save(fig, "fig5_foundation")


if __name__ == "__main__":
    main()
