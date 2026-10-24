"""Write tables/generated/*.tex from data/results.csv and the frozen repository facts.

Each table is a complete table environment that main.tex \\input s. Numbers
whose rows are still placeholders are wrapped in \\ph{}, so they print in a
separate colour until scripts/collect_runs.py has replaced them.

    python paper/scripts/make_tables.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.colors import to_hex

PAPER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PAPER / "figures"))

from retifig import data as D  # noqa: E402
from retifig import style as S  # noqa: E402

OUT = PAPER / "tables" / "generated"
FAMILY_TITLE = {
    "general": "General segmentation",
    "auto": "Self-configuring",
    "fundus": "Fundus-specific",
    "foundation": "Foundation models, adapted",
}


# ----------------------------------------------------------------- helpers
def tint(v, vmin, vmax, cmap=None, lo=0.04, hi=0.56):
    """Light cell colour for a value: the lower half of the figure colour map, so black text stays readable."""
    if not np.isfinite(v):
        return ""
    cmap = cmap or S.CMAP_SEQ
    t = np.clip((v - vmin) / (vmax - vmin), 0, 1)
    return "\\cellcolor[HTML]{" + to_hex(cmap(lo + (hi - lo) * t))[1:].upper() + "}"


def div_tint(v, lim):
    """Diverging tint around zero for a difference."""
    if not np.isfinite(v):
        return ""
    t = 0.5 + 0.27 * np.clip(v / lim, -1, 1)
    return "\\cellcolor[HTML]{" + to_hex(S.CMAP_DIV(t))[1:].upper() + "}"


def rank_marks(values, higher=True):
    """Index of the best and second-best finite value."""
    v = np.asarray(values, float)
    order = np.argsort(-v if higher else v)
    order = [i for i in order if np.isfinite(v[i])]
    return (order[0] if order else -1), (order[1] if len(order) > 1 else -1)


def num(v, fmt="{:.3f}", sim=True, mark=None, strip=True):
    if v is None or not np.isfinite(v):
        return "\\na"
    places = int(fmt[3]) if fmt.startswith("{:.") else None
    if places is not None:  # half up on the decimal value as written: 0.6755 -> .676
        from decimal import ROUND_HALF_UP, Decimal

        s = str(Decimal(repr(round(float(v), 10))).quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP))
    else:
        s = fmt.format(v)
    if strip and s.startswith("0."):
        s = s[1:]
    if strip and s.startswith("-0."):
        s = "-" + s[2:]
    if mark == "best":
        s = f"\\best{{{s}}}"
    elif mark == "second":
        s = f"\\second{{{s}}}"
    return f"\\ph{{{s}}}" if sim else s


def sd(v, sim=True):
    if not np.isfinite(v):   # one seed so far: no s.d. (2026-10-22; the rows used to print "+-nan")
        return ""
    s = f"{v:.3f}"[1:] if v < 1 else f"{v:.3f}"
    return f"\\sd{{\\ph{{{s}}}}}" if sim else f"\\sd{{{s}}}"


def delta(v, sim=True, fmt="{:+.3f}"):
    if not np.isfinite(v):
        return "\\na"
    s = fmt.format(abs(v)).replace("+", "")
    s = s[1:] if s.startswith("0.") else s
    body = f"\\ph{{{s}}}" if sim else s
    if abs(v) < 0.005:
        return f"\\same{{{body}}}"
    return f"\\up{{{body}}}" if v > 0 else f"\\down{{{body}}}"


def signed(v, sim=True, fmt="{:+.2f}"):
    """Signed difference in ink, for cells whose shading already carries the direction."""
    if not np.isfinite(v):
        return "\\na"
    t = fmt.format(v)
    sign, body = t[0], t[1:]
    if body.startswith("0."):
        body = body[1:]
    t = ("$-$" if sign == "-" else sign) + body          # one math minus; the old chain re-replaced its own "-" into "$$-$$"
    return f"\\ph{{{t}}}" if sim else t


def write(name, text):
    OUT.mkdir(parents=True, exist_ok=True)
    # Never wider than the text block: scale the tabular down when it would overflow.
    text = text.replace("\\begin{threeparttable}", "\\begin{adjustbox}{max width=\\textwidth}\n\\begin{threeparttable}")
    text = text.replace("\\end{threeparttable}", "\\end{threeparttable}\n\\end{adjustbox}")
    (OUT / f"{name}.tex").write_text(text)
    print(f"wrote tables/generated/{name}.tex")


def grouped_models():
    out = []
    for fam in S.FAMILY_ORDER:
        out.append((fam, [m for m in D.MAIN_MODELS if D.FAMILY_OF[m] == fam]))
    return out


# ------------------------------------------------------------- Table 1 data
def table_datasets():
    rows = []
    role_title = {"train": "Training sources (internal test = official test set)", "independent": "Independent test sources (never used to train a main-table model)"}
    tier_text = {"fine": "Fine", "mixed": "Mixed", "coarse": "Coarse", "region": "Region"}
    for role in ("train", "independent"):
        rows.append(f"\\grouprow{{9}}{{{role_title[role]}}}")
        for d in D.DATASETS:
            if d["role"] != role:
                continue
            split = f"{d['train']:,} / {d['val']} / {d['test']}" if d["train"] else f"\\na\\ / {d['val']} / {d['test']}"
            bar = f"\\inlinebar{{{max(0.12, 3.2 * d['n'] / 1842):.2f}}}"
            rows.append(
                " & ".join(
                    [
                        d["name"].replace("TJDR-std", "TJDR, standard field").replace("TJDR-uwf", "TJDR, ultra-widefield"),
                        f"\\cite{{{d['cite']}}}",
                        f"{d['n']:,} {bar}",
                        d["res"].replace("×", "$\\times$"),
                        d["fov"].replace("°", "$^\\circ$").replace("–", "--"),
                        d["ann"].replace("≥", "$\\geq$"),
                        f"\\tierbadge{{{d['tier']}}}{{{tier_text[d['tier']]}}}",
                        split,
                        "\\tbd" if d["public"] is None else ("\\yes" if d["public"] else "\\no"),
                    ]
                )
                + " \\\\"
            )
    body = "\n".join(rows)
    write(
        "tab_datasets",
        rf"""\begin{{table*}}[t]
\centering\tablefont
\caption{{\textbf{{Datasets, frozen splits and roles.}} All eight sources carry pixel- or region-level labels for microaneurysms (MA), hemorrhages (HE), hard exudates (EX) and soft exudates (SE). Granularity decides which test sets may be averaged: averages are taken only within one tier. Split is train\,/\,validation\,/\,test in images; official test sets are never changed. ``Maps'' marks sources whose probability maps we can release.}}
\label{{tab:datasets}}
\begin{{threeparttable}}
\begin{{tabular}}{{@{{}}l l l l l l c r c@{{}}}}
\toprule
\colh{{Source}} & \colh{{Ref.}} & \colh{{Images}} & \colh{{Resolution}} & \colh{{FOV}} & \colh{{Annotation}} & \colh{{Granularity}} & \colh{{Split}} & \colh{{Maps}} \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\begin{{tablenotes}}[flushleft]\scriptsize
\item FOV, field of view; n.s., not stated in the source paper. IDRiD: 10 of the 54 official training images are held out for validation in unified retraining; author-recipe retraining uses all 54. DiaRetDB1: ground truth is the mean confidence of four experts thresholded at 0.75; the 28 official training images are used only to choose thresholds. Retinal-Lesions: split by patient; grey-level 127 regions are ignored. MAPLES-DR: labels drawn at $1500\\times1500$ on the field-of-view crop of MESSIDOR images and placed back at the original resolution; the 138 training images are not used, the official 60 test images are; uncertain red (MA or HE) and bright (EX, SE or drusen) lesions are ignored for the two classes they could belong to. The FGADR licence forbids redistribution of derived data; maps for Retinal-Lesions are not released either.
\end{{tablenotes}}
\end{{threeparttable}}
\end{{table*}}
""",
    )


# ------------------------------------------------------------ Table 2 models
MODEL_NOTE = {
    "U-Net": ("reference implementation", "B-std"),
    "DeepLabv3": ("torchvision, ResNet-101", "B-std"),
    "HRNet": ("official, W48", "B-std"),
    "nnU-Net": ("official, 2D; default and ResEnc-M", "own entry point"),
    "Swin-Unet": ("official", "B-std, 224 patches"),
    "FCT": ("official PyTorch port", "B-std, 384 patches"),
    "H2Former": ("official", "deferred: exceeds 40\\,GB at 960"),
    "UNet++": ("official", "tier 2"),
    "UPerNet": ("official", "tier 2"),
    "M2MRF": ("official; weights released", "A, E1r, B-std"),
    "HACDR-Net": ("official; no weights", "E1r, B-std"),
    "HRDecoder": ("official; weights released", "A, B-std"),
    "WSRFNet": ("model and test code only; training loop added", "A if weights arrive; B-std"),
    "L-Seg": ("Caffe prototxt and loss layer", "C"),
    "SAA": ("repository empty", "C"),
    "PMCNet": ("official Keras code rewritten in PyTorch", "B-std"),
    "CLC-Net": ("no code; re-implemented from the paper", "C, B-std"),
    "TC-Net": ("no code; re-implemented from the paper", "C, B-std"),
    "RTNet": ("no code; image + vessel input", "C"),
    "SAM": ("official inference", "zero-shot, oracle prompt"),
    "SAM2": ("official", "zero-shot, oracle prompt"),
    "SAM3": ("official; gated weights", "text prompt"),
    "MedSAM": ("official", "oracle box, full fine-tuning"),
    "SAMed": ("official", "PEFT (LoRA)"),
    "SAM-Adapter": ("official", "PEFT"),
    "SAM2-UNet": ("official", "PEFT"),
    "RETFound": ("official encoder; UPerNet head added", "encoder + decoder"),
    "PerSAM": ("official inference", "one-shot, tier 2"),
}
DISPLAY = {"SAM2": "SAM 2", "SAM3": "SAM 3", "HRNet": "HRNet-W48", "RETFound": "RETFound + UPerNet"}
FAM_OF_REG = {"general": "general", "fundus_multiscale": "fundus", "relational_transformer": "fundus", "semi_supervised": "fundus", "foundation": "foundation"}


def table_models():
    reg = D.registry()
    reg = reg[reg.tier <= 2]
    groups = [("general", "General segmentation and self-configuring baselines"), ("fundus_multiscale", "Fundus-specific, multi-scale CNN"),
              ("relational_transformer", "Fundus-specific, relation or transformer"), ("foundation", "Foundation models")]
    rows = []
    for fam, title in groups:
        sub = reg[reg.family == fam].sort_values(["tier"], kind="stable")
        if not len(sub):
            continue
        rows.append(f"\\grouprow{{8}}{{{title}}}")
        for _, r in sub.iterrows():
            chip = "auto" if r["name"] == "nnU-Net" else FAM_OF_REG[fam]
            code, use = MODEL_NOTE.get(r["name"], ("", ""))
            tr = r["tracks"]
            rows.append(
                " & ".join(
                    [
                        f"\\famchip{{{chip}}}\\; {DISPLAY.get(r['name'], r['name'])}",
                        str(r["tier"]),
                        f"\\rlevel{{{r['repro_level'][1]}}} {{\\scriptsize {r['repro_level']}}}",
                        "\\yes" if "A" in tr else "\\no",
                        "\\yes" if "B" in tr else "\\no",
                        "\\yes" if "C" in tr else "\\no",
                        code,
                        use,
                    ]
                )
                + " \\\\"
            )
    all_reg = D.registry()
    n1, n2, n3 = [(all_reg.tier == t).sum() for t in (1, 2, 3)]
    body = "\n".join(rows)
    write(
        "tab_models",
        rf"""\begin{{table*}}[t]
\centering\tablefont
\caption{{\textbf{{Models entering the retraining tracks.}} {len(all_reg)} variants are registered: {n1} in tier~1 (main tables), {n2} in tier~2 (if time allows) and {n3} in tier~3 (paper audit only; Supplementary Table~2). Reproducibility level: R0, no code; R1, model definition only; R2, inference code and weights; R3, training code; R4, training code, weights, splits and environment. Tracks: A, author weights; B, unified retraining of the official code; C, re-implementation from the paper, reported in a separate column.}}
\label{{tab:models}}
\begin{{threeparttable}}
\begin{{tabular}}{{@{{}}l c l c c c l l@{{}}}}
\toprule
 & & & \multicolumn{{3}}{{c}}{{\colh{{Track}}}} & & \\
\cmidrule(lr){{4-6}}
\colh{{Model}} & \colh{{Tier}} & \colh{{Reproducibility}} & \colh{{A}} & \colh{{B}} & \colh{{C}} & \colh{{Code status}} & \colh{{Use in this study}} \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\begin{{tablenotes}}[flushleft]\scriptsize
\item \famchip{{general}} general segmentation \quad \famchip{{auto}} self-configuring \quad \famchip{{fundus}} fundus-specific \quad \famchip{{foundation}} foundation model. B-std, unified retraining protocol (Methods). PEFT, parameter-efficient fine-tuning. Status as of the code audit of 2--4 October 2026; author replies are due by 31 October 2026.
\end{{tablenotes}}
\end{{threeparttable}}
\end{{table*}}
""",
    )


# --------------------------------------------------- Table 3 original models
def e1(model, variant, train, metric, cls="mean", test=None):
    r = D.sel(experiment="E1", model=model, variant=variant, train_set=train, test_set=test or train, metric=metric, cls=cls)
    if not len(r):
        return np.nan, True
    return float(r["value"].mean()), D.simulated(r)


def table_original():
    rows = []
    for model in ("M2MRF", "HRDecoder", "WSRFNet"):
        for train in D.TRAIN_MAIN:
            paper = e1(model, "paper", train, "aupr_author_impl")
            auth = e1(model, "author_function", train, "aupr_author_impl")
            stored = e1(model, "this_protocol", train, "aupr_11pt")
            exact = e1(model, "this_protocol", train, "aupr")
            cls = [e1(model, "this_protocol", train, "aupr", c) for c in D.CLASSES]
            ind = [e1(model, "this_protocol", train, "aupr", c, test=t) for t in D.FINE_INDEPENDENT for c in D.CLASSES]
            ind_v = np.nanmean([v for v, _ in ind])
            cells = [f"{model}", train, num(paper[0], sim=paper[1]), num(auth[0], sim=auth[1]), num(stored[0], sim=stored[1]), num(exact[0], sim=exact[1])]
            cells += [tint(v, 0, 0.9) + num(v, sim=s) for v, s in cls]
            cells += [delta(exact[0] - auth[0], sim=exact[1] or auth[1]), num(ind_v, sim=any(s for _v, s in ind))]
            rows.append(" & ".join(cells) + " \\\\")
        rows.append("\\addlinespace[2pt]")
    rows = rows[:-1]
    # author recipe: one row per (model, source) that has rows; the class-mean rows only, measured seeds averaged,
    # placeholder seeds shown as placeholder (2026-10-21: the block used to average the per-class rows as if they were seeds)
    rec = []
    for model in ("M2MRF", "HACDR-Net", "HRDecoder"):
        for train in D.TRAIN_MAIN:
            a = D.sel(experiment="E1r", model=model, variant="author_recipe", train_set=train, metric="aupr_author_impl", cls="mean")
            x = D.sel(experiment="E1r", model=model, variant="author_recipe", train_set=train, metric="aupr", cls="mean")
            if not len(a):
                continue
            p = D.sel(experiment="E1r", model=model, variant="paper", train_set=train, metric="aupr_author_impl")
            pv, p_sim = (float(p["value"].iloc[0]), D.simulated(p)) if len(p) else e1(model, "paper", train, "aupr_author_impl")
            a_sim, x_sim = D.simulated(a), D.simulated(x)
            a_use = a if a_sim else a[a.source == "measured"]
            x_use = x if x_sim else x[x.source == "measured"]
            seeds = a_use["seed"].nunique()
            a_sd = sd(a_use["value"].std(), a_sim) if seeds > 1 else ""
            x_sd = sd(x_use["value"].std(), x_sim) if seeds > 1 else ""
            rec.append(
                " & ".join(
                    [model, train, num(pv, sim=p_sim), num(a_use["value"].mean(), sim=a_sim) + a_sd, (f"\\ph{{{seeds}}}" if a_sim else str(seeds)),
                     delta(a_use["value"].mean() - pv, sim=a_sim or p_sim), num(x_use["value"].mean(), sim=x_sim) + x_sd]
                )
                + " \\\\"
            )
    first = D.sel(experiment="E1r", model="M2MRF", variant="bs1_first_run", cls="mean")
    fa = float(first[first.metric == "aupr_author_impl"]["value"].iloc[0])
    fx = float(first[first.metric == "aupr"]["value"].iloc[0])
    rec.insert(1, " & ".join(["\\quad first run, batch 1", "IDRiD", num(0.6724, sim=False), num(fa, sim=False), "1", delta(fa - 0.6724, sim=False), num(fx, sim=False)]) + " \\\\")
    write(
        "tab_original",
        rf"""\begin{{table*}}[t]
\centering\tablefont
\caption{{\textbf{{Original models: three numbers per released checkpoint, and retraining with the authors' recipe.}} \textbf{{a}}, Each checkpoint is scored on the official test set of its training source with the number printed in the paper, the authors' evaluation function (11 thresholds, whole image, overwritten labels) on the float output, the same function on the stored 16-bit maps, and this study's protocol (exact pooled AUPR inside the field of view, overlapping labels). \textbf{{b}}, The authors' pipeline rerun from scratch; mean $\pm$ s.d. over seeds, last iteration, no checkpoint selection.}}
\label{{tab:original}}
\begin{{threeparttable}}
\begin{{tabular}}{{@{{}}l l c c c c c c c c c c@{{}}}}
\toprule
\multicolumn{{12}}{{@{{}}l}}{{\textbf{{a}}\quad Author weights (E1)}}\\
\addlinespace[2pt]
 & & \multicolumn{{3}}{{c}}{{\colh{{mAUPR, author metric}}}} & \multicolumn{{5}}{{c}}{{\colh{{AUPR, this protocol}}}} & & \\
\cmidrule(lr){{3-5}}\cmidrule(lr){{6-10}}
\colh{{Model}} & \colh{{Source}} & Paper & Function & Stored & \colh{{mAUPR}} & MA & HE & EX & SE & \colh{{$\Delta$ metric}} & \colh{{Indep.\ fine}} \\
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{tabular}}

\vspace{{6pt}}
\begin{{tabular}}{{@{{}}l l c c c c c@{{}}}}
\toprule
\multicolumn{{7}}{{@{{}}l}}{{\textbf{{b}}\quad Author recipe, retrained (E1r)}}\\
\addlinespace[2pt]
\colh{{Model}} & \colh{{Source}} & \colh{{Paper}} & \colh{{Retrained, author metric}} & \colh{{Seeds}} & \colh{{$\Delta$ to paper}} & \colh{{Retrained, this protocol}} \\
\midrule
{chr(10).join(rec)}
\bottomrule
\end{{tabular}}
\begin{{tablenotes}}[flushleft]\scriptsize
\item ``$\Delta$ metric'' is this protocol minus the authors' function on the same checkpoint. ``Indep.\ fine'' is the mean mAUPR on the three fine-grained independent test sets (TJDR standard-field, TJDR ultra-widefield and MAPLES-DR). The M2MRF paper reports the mean of three runs (67.24 on IDRiD, 48.94 on DDR), the repository the best run (67.55, 49.20). HRDecoder: the released checkpoints (repository table 71.34, 49.24) through the authors' test pipeline (resize to $2880\times1920$ with sliding $1920$ windows on IDRiD, $2048\times2048$ whole image on DDR). The first M2MRF rerun used an effective batch of 1 instead of the authors' 4.
\phnote
\end{{tablenotes}}
\end{{threeparttable}}
\end{{table*}}
""",
    )


# ------------------------------------------------------ Table 4 main results
def table_main():
    """Table 3. Simulated / measured is decided per (model, source): a measured row reads as a result while other rows
    are still placeholders (2026-10-15), and best / second-best marks are taken among the measured rows of a column."""
    internal = D.internal()
    ms = D.mean_sd()
    dice = D.mean_sd("dice_valopt")
    runs = D.main_runs()
    measured = {(m, t): not D.simulated(g) for (m, t), g in runs[runs.train_set == runs.test_set].groupby(["model", "train_set"])}
    dice_runs = D.main_runs("dice_valopt")
    dice_measured = {(m, t): not D.simulated(g) for (m, t), g in dice_runs[dice_runs.train_set == dice_runs.test_set].groupby(["model", "train_set"])}
    params = D.sel(experiment="E10", metric="params_m").set_index("model")
    cols = {}
    for train in D.TRAIN_MAIN:
        tab = internal.xs(train, level="train_set").loc[D.MAIN_MODELS]
        m = ms[(ms.train_set == train) & (ms.test_set == train)].set_index("model").loc[D.MAIN_MODELS]
        d = dice[(dice.train_set == train) & (dice.test_set == train)].set_index("model").loc[D.MAIN_MODELS]
        cols[train] = (tab, m, d)
    marks = {}
    for train, (tab, m, d) in cols.items():
        keep = np.array([measured.get((model, train), False) for model in D.MAIN_MODELS])
        keep_d = np.array([dice_measured.get((model, train), False) for model in D.MAIN_MODELS])
        for c in D.CLASSES:
            marks[(train, c)] = rank_marks(np.where(keep, tab[c].to_numpy(), np.nan))
        marks[(train, "m")] = rank_marks(np.where(keep, m["mean"].to_numpy(), np.nan))
        marks[(train, "d")] = rank_marks(np.where(keep_d, d["mean"].to_numpy(), np.nan))

    def mk(key, i):
        b, s2 = marks[key]
        return "best" if i == b else ("second" if i == s2 else None)

    rows = []
    for fam, models in grouped_models():
        rows.append(f"\\grouprow{{14}}{{\\famchip{{{fam}}}\\; {FAMILY_TITLE[fam]}}}")
        for model in models:
            i = D.MAIN_MODELS.index(model)
            p = params.loc[model]
            label = model + ("$^{\\rm C}$" if model in D.TRACK_C else "")
            cells = [label, num(float(p["value"]), "{:.1f}", sim=p["source"] != "measured", strip=False)]
            for train in D.TRAIN_MAIN:
                tab, m, d = cols[train]
                sim = not measured.get((model, train), False)
                for c in D.CLASSES:
                    v = tab.loc[model, c]
                    cells.append(tint(v, 0.0, 0.9) + num(v, sim=sim, mark=mk((train, c), i)))
                cells.append(num(m.loc[model, "mean"], sim=sim, mark=mk((train, "m"), i)) + sd(m.loc[model, "sd"], sim))
                cells.append(num(d.loc[model, "mean"], sim=not dice_measured.get((model, train), False), mark=mk((train, "d"), i)))
            rows.append(" & ".join(cells) + " \\\\")
    body = "\n".join(rows)
    write(
        "tab_main",
        rf"""\begin{{table*}}[t]
\centering\tablefont
\caption{{\textbf{{Unified retraining (B-std): internal tests.}} Every model is trained on one source under the shared protocol and scored on that source's official test set. Per-class values are exact pooled AUPR inside the field of view, mean of three seeds; mAUPR is the mean of the four classes $\pm$ s.d.\ over seeds; Dice is pooled, at the per-class threshold chosen on the validation set. Cell shading follows AUPR. \best{{Bold}}, best in column; \second{{underlined}}, second.}}
\label{{tab:main}}
\begin{{threeparttable}}
\begin{{tabular}}{{@{{}}l r cccc c c @{{\hspace{{10pt}}}} cccc c c@{{}}}}
\toprule
 & & \multicolumn{{6}}{{c}}{{\colh{{Trained and tested on IDRiD}} {{\scriptsize(44 / 10 / 27)}}}} & \multicolumn{{6}}{{c}}{{\colh{{Trained and tested on DDR}} {{\scriptsize(383 / 149 / 225)}}}} \\
\cmidrule(lr){{3-8}}\cmidrule(l){{9-14}}
\colh{{Model}} & \colh{{Params}} & MA & HE & EX & SE & \colh{{mAUPR}} & \colh{{Dice}} & MA & HE & EX & SE & \colh{{mAUPR}} & \colh{{Dice}} \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\begin{{tablenotes}}[flushleft]\scriptsize
\item Params in millions. nnU-Net uses its own preprocessing and planning with the same splits, test images and evaluation. $^{{\rm C}}$, no author code: re-implemented from the paper (Track~C); the row shows what the published description yields under this protocol, not the authors' implementation. HRDecoder, M2MRF and HRNet-W48 share the HRNet-W48 backbone. H2Former is registered in tier~1 but deferred: at its 960-pixel input one training step exceeds the 40\,GB of the cards used here.
\phnote
\end{{tablenotes}}
\end{{threeparttable}}
\end{{table*}}
""",
    )


# ----------------------------------------------- Table 5 independent testing
ENSEMBLE_ORDER = ["E-top3", "E-top5", "E-1perfam", "E-all", "E-fam:general", "E-fam:auto", "E-fam:fundus", "E-fam:foundation"]


def ensemble_rows(pix):
    """The ensemble block of Table 5 (docs/paper/14 M15; bench/scripts/ensemble.py): measured rows only, no rank
    (ensembles are a reference, not part of the Friedman test); empty until collect_runs has read runs/ENS."""
    # variant "ensemble" is the registered composition; "ensemble:without <member>" a set on which a member's maps
    # could not be produced (ensemble.py --without; 21 item 87): such a cell is marked with \partial
    all_rows = D.sel(experiment="ENS", metric="aupr")
    all_rows = all_rows[all_rows.variant.str.startswith("ensemble")]
    df = all_rows
    if df.empty:
        return []
    df = df[df["cls"].isin(D.CLASSES)]
    score = df.groupby(["model", "train_set", "test_set"])["value"].mean()
    partial = {k for k, g in df.groupby(["model", "train_set", "test_set"]) if (g.variant != "ensemble").any()}
    auc = D.sel(experiment="ENS", metric="img_auc")
    auc = auc[auc.variant.str.startswith("ensemble")]
    auc = auc.groupby(["model", "train_set"])["value"].mean() if not auc.empty else None
    rows = [r"\grouprow{18}{Ensembles of the rows above (pixelwise mean of the probability maps; members chosen on validation data)}"]
    for model in [m for m in ENSEMBLE_ORDER if m in score.index.get_level_values(0)]:
        cells = [model.replace("fam:", "fam.\\,")]
        for t in D.TRAIN_MAIN:
            def v(c):
                return score.get((model, t, c), np.nan)
            cells.append(num(v(t), "{:.2f}", np.isnan(v(t))) + ("$^{\\rm p}$" if (model, t, t) in partial else ""))
            for c in pix:
                cells.append(tint(v(c), 0.0, 0.7) + num(v(c), "{:.2f}", np.isnan(v(c))) + ("$^{\\rm p}$" if (model, t, c) in partial else ""))
            a = auc.get((model, t), np.nan) if auc is not None else np.nan
            cells.append(num(a, "{:.2f}", np.isnan(a)))
            fine = [v(c) for c in D.FINE_INDEPENDENT]
            d = np.nanmean(fine) - v(t) if not np.isnan(v(t)) and not all(np.isnan(fine)) else np.nan
            cells.append((div_tint(d, 0.5) if not np.isnan(d) else "") + (signed(d, False) if not np.isnan(d) else "--"))
        cells.append("--")
        rows.append(" & ".join(cells) + " \\\\")
    return rows


def human_rows():
    """The human-reference block of Table 5 (docs/paper/14 M16a; bench/scripts/diaretdb1_experts.py): each DiaRetDB1
    expert scored against the consensus of the other three with the protocol. Only the DiaRetDB1 column (image-level
    AUC, as the model rows) applies; the experts trained on nothing, so the value stands under both training blocks.
    Nothing is ranked against them; empty until collect_runs has read runs/HUMAN."""
    df = D.sel(experiment="HUMAN", variant="leave_one_expert_out", metric="img_auc")
    if df.empty:
        return []
    df = df[df["cls"].isin(D.CLASSES)]
    auc = df.groupby("model")["value"].mean()
    if auc.empty:
        return []
    rows = [r"\grouprow{18}{Human reference: one DiaRetDB1 expert against the consensus of the other three (leave-one-expert-out; no training source, same value under both blocks)}"]
    for model in sorted(auc.index):
        cells = [model]
        for _t in D.TRAIN_MAIN:
            cells += ["--"] * 6
            cells.append(num(auc[model], "{:.2f}", False))
            cells.append("--")
        cells.append("--")
        rows.append(" & ".join(cells) + " \\\\")
    if len(auc) > 1:
        cells = ["Expert range"]
        for _t in D.TRAIN_MAIN:
            cells += ["--"] * 6
            cells.append(f"{num(auc.min(), '{:.2f}', False)}--{num(auc.max(), '{:.2f}', False)}")
            cells.append("--")
        cells.append("--")
        rows.append(" & ".join(cells) + " \\\\")
    return rows


def table_independent():
    """Table 5. Simulated / measured is decided per cell (model, source, test set), so the rows measured so far read
    as results while the rest of the table is still placeholder (2026-10-15: nine supervised models on all eight sets);
    best / second-best marks are taken among the measured cells of a column only, so a placeholder row never earns one."""
    runs = D.main_runs()
    measured = {(m, t, s): not D.simulated(g) for (m, t, s), g in runs.groupby(["model", "train_set", "test_set"])}
    auc_runs = D.main_runs("img_auc")
    auc_measured = {(m, t): not D.simulated(g) for (m, t), g in auc_runs.groupby(["model", "train_set"])}
    blocks = D.block_scores()
    fr = D.friedman(blocks)
    rank = dict(zip(blocks.index, fr["avg_rank"]))
    auc = auc_runs.groupby(["model", "train_set"])["value"].mean().unstack("train_set")
    mats = {t: D.score_matrix(t) for t in D.TRAIN_MAIN}
    pix = ["TJDR-std", "TJDR-uwf", "MAPLES-DR", "FGADR", "Retinal-Lesions"]

    def cell_sim(model, t, c):
        return not measured.get((model, t, c), False)

    marks = {}
    for t in D.TRAIN_MAIN:
        for c in [t] + pix:
            vals = [mats[t].loc[m, c] if not cell_sim(m, t, c) else np.nan for m in D.MAIN_MODELS]
            marks[(t, c)] = rank_marks(vals)
        vals = [auc.loc[m, t] if auc_measured.get((m, t), False) and m in auc.index else np.nan for m in D.MAIN_MODELS]
        marks[(t, "auc")] = rank_marks(vals)
    marks["rank"] = rank_marks([rank.get(m, np.nan) for m in D.MAIN_MODELS], higher=False)

    def mk(key, i):
        b, s2 = marks[key]
        return "best" if i == b else ("second" if i == s2 else None)

    rows = []
    for fam, models in grouped_models():
        rows.append(f"\\grouprow{{18}}{{\\famchip{{{fam}}}\; {FAMILY_TITLE[fam]}}}")
        for model in models:
            i = D.MAIN_MODELS.index(model)
            cells = [model]
            for t in D.TRAIN_MAIN:
                m = mats[t]
                cells.append(num(m.loc[model, t], "{:.2f}", cell_sim(model, t, t), mk((t, t), i)))
                for c in pix:
                    v = m.loc[model, c]
                    cells.append(tint(v, 0.0, 0.7) + num(v, "{:.2f}", cell_sim(model, t, c), mk((t, c), i)))
                a = auc.loc[model, t] if model in auc.index else np.nan
                cells.append(num(a, "{:.2f}", not auc_measured.get((model, t), False), mk((t, "auc"), i)))
                d = m.loc[model, D.FINE_INDEPENDENT].mean() - m.loc[model, t]
                d_sim = cell_sim(model, t, t) or any(cell_sim(model, t, c) for c in D.FINE_INDEPENDENT)
                cells.append(div_tint(d, 0.5) + signed(d, d_sim))
            cells.append(num(rank.get(model, np.nan), "{:.2f}", model not in rank, mk("rank", i), strip=False))
            rows.append(" & ".join(cells) + " \\\\")
    rows += ensemble_rows(pix)
    rows += human_rows()
    body = "\n".join(rows)
    hdr = " & ".join(["Int."] + ["Std", "UWF", "Maples", "FGADR", "Ret.-L.", "DiaRet", "$\\Delta$"])
    tiers = " & ".join(["\\tier{fine}", "\\tier{fine}", "\\tier{fine}", "\\tier{fine}", "\\tier{mixed}", "\\tier{coarse}", "\\tier{region}", ""])
    write(
        "tab_independent",
        rf"""\begin{{table*}}[t]
\centering\tablefont\setlength{{\tabcolsep}}{{3.0pt}}
\caption{{\textbf{{Unified retraining (B-std): independent tests.}} The checkpoints of Table~\ref{{tab:main}} are applied without any adaptation to six test sets from sources never used for training. Values are mAUPR (mean of three seeds); the DiaRetDB1 column is the official image-level ROC AUC, averaged over the four lesion types, because its region-consensus ground truth biases pixel precision. $\Delta$ is the mean of the three fine-grained independent test sets (TJDR standard-field, TJDR ultra-widefield and MAPLES-DR) minus the internal test. Average rank is over the eight test sets of Fig.~\ref{{fig:retrained}}e (Friedman $\chi^2$ = \ph{{{fr['statistic']:.1f}}}; Nemenyi critical difference \ph{{{fr['critical_difference']:.2f}}}).}}
\label{{tab:independent}}
\begin{{adjustbox}}{{max width=\textwidth}}
\begin{{threeparttable}}
\begin{{tabular}}{{@{{}}l c ccccc c c @{{\hspace{{8pt}}}} c ccccc c c @{{\hspace{{8pt}}}} c@{{}}}}
\toprule
 & \multicolumn{{8}}{{c}}{{\colh{{Trained on IDRiD}}}} & \multicolumn{{8}}{{c}}{{\colh{{Trained on DDR}}}} & \\
\cmidrule(lr){{2-9}}\cmidrule(lr){{10-17}}
 & & \multicolumn{{2}}{{c}}{{TJDR}} & & & & & & & \multicolumn{{2}}{{c}}{{TJDR}} & & & & & & \colh{{Avg.}} \\
\colh{{Model}} & {hdr} & {hdr} & \colh{{rank}} \\
 & {tiers} & {tiers} & \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\begin{{tablenotes}}[flushleft]\scriptsize
\item Granularity of the test set: \tier{{fine}} fine \quad \tier{{mixed}} mixed \quad \tier{{coarse}} coarse \quad \tier{{region}} region consensus. Int., internal test (official test set of the training source). Std, standard field (35--50$^\circ$); UWF, ultra-widefield (133$^\circ$), predicted at native resolution with sliding windows; Maples, MAPLES-DR (official test split, 60 images). Retinal-Lesions excludes its grey-level 127 regions. Best and second-best are marked among the measured rows of a column. $^{{\rm p}}$, ensemble computed without a registered member whose maps could not be produced on that set (named in the run's record).
\phnote
\end{{tablenotes}}
\end{{threeparttable}}
\end{{adjustbox}}
\end{{table*}}
""",
    )


# ------------------------------------------------ Table 6 foundation models
def table_foundation():
    """Table 6 (docs/ops/00, the Luming list item 5, decided 2026-10-14): the zero-shot block holds only what sees no
    test label and is summarised by the any-lesion AUPR; everything that reads the test labels (automatic masks
    classed by overlap, oracle prompts) is one 'upper bound' block; best / second-best are marked within a block."""

    # simulated / measured is decided per cell (row, test set) from the rows a cell reads (2026-10-22, docs/paper/21 #81):
    # ctx["cell"] names the cell being filled; e5 / any_lesion record whether its rows are all measured
    ctx, sims = {}, {}

    def e5(model, variant, test, train=None):
        kw = dict(experiment="E5", model=model, variant=variant, test_set=test, metric="aupr")
        if train:
            kw["train_set"] = train
        r = D.sel(**kw)
        inference = r[r["seed"] == -1]  # one run per (model, variant, test set) in the inference tiers: never average two
        if len(inference) and int(inference.groupby("cls").size().max()) > 1:
            raise ValueError(f"E5 {model} {variant} {test}: more than one inference-tier row per class; a variant must name one run directory")
        if "cell" in ctx:
            sims[ctx["cell"]] = sims.get(ctx["cell"], False) or (not len(r)) or D.simulated(r)
        return r.groupby("cls")["value"].mean().reindex(D.CLASSES).to_numpy()

    def any_lesion(model, variant, test):
        # the registered zero-shot result is class-agnostic (e5_frozen.yaml zero_shot.outputs.main): collect_runs metric
        # aupr_any_lesion; a missing row is an error, never a substitute number (review N2)
        r = D.sel(experiment="E5", model=model, variant=variant, test_set=test, metric="aupr_any_lesion", cls="any")
        if not len(r):
            raise ValueError(f"E5 {model} {variant} {test}: no aupr_any_lesion row in results.csv; regenerate the placeholder table "
                             "(python scripts/build_placeholder.py --force, then collect_runs.py and collect_literature.py) or collect the runs")
        if "cell" in ctx:
            sims[ctx["cell"]] = sims.get(ctx["cell"], False) or D.simulated(r)
        return float(r["value"].mean())

    blank = lambda t: np.full(4, np.nan)
    # block title, summary column semantics ("any" = any-lesion AUPR, "mean" = mean of the four classes), rows:
    # (model, use, backbone, per-class values, summary) -- summary None means the mean of the four classes
    spec = [
        ("Zero-shot: no test label; density, area cap and wording chosen on the validation pool (summary = any-lesion AUPR$^\\ddagger$)", "any", [
            ("SAM", "automatic masks, class-agnostic", "ViT-H", blank, lambda t: any_lesion("SAM", "zero_shot:auto_any_lesion", t)),
            ("SAM 2", "automatic masks, class-agnostic", "Hiera-L", blank, lambda t: any_lesion("SAM 2", "zero_shot:auto_any_lesion", t)),
            ("SAM 3", "text, wording chosen on validation", "--", lambda t: e5("SAM 3", "text:selected", t), lambda t: any_lesion("SAM 3", "text:selected", t)),
        ]),
        ("Upper bounds that read the test labels (not a usable setting)", "mean", [
            ("SAM", "automatic masks, class by overlap$^\\dagger$", "ViT-H", lambda t: e5("SAM", "zero_shot:auto_oracle_class", t), None),
            ("SAM 2", "automatic masks, class by overlap$^\\dagger$", "Hiera-L", lambda t: e5("SAM 2", "zero_shot:auto_oracle_class", t), None),
            ("SAM", "oracle boxes, every component", "ViT-B", lambda t: e5("SAM", "oracle:box:all", t), None),  # the oracle tier uses ViT-B (16, D3)
            ("SAM 2", "oracle points, every component", "Hiera-L", lambda t: e5("SAM 2", "oracle:point:all", t), None),
            ("SAM 2", "oracle boxes, every component", "Hiera-L", lambda t: e5("SAM 2", "oracle:box:all", t), None),
            ("MedSAM", "oracle boxes, every component", "ViT-B", lambda t: e5("MedSAM", "oracle:box:all", t), None),
        ]),
        ("Parameter-efficient fine-tuning (trained on the source of the test set)", "mean", [
            ("SAMed", "LoRA", "ViT-B", lambda t: e5("SAMed", "peft", t, t), None),
            ("SAM-Adapter", "adapters", "ViT-H", lambda t: e5("SAM-Adapter", "peft", t, t), None),
            ("SAM2-UNet", "adapters + U-shaped decoder", "Hiera-L", lambda t: e5("SAM2-UNet", "peft", t, t), None),
        ]),
        ("Encoder + decoder (trained on the source of the test set)", "mean", [
            ("MedSAM", "full fine-tuning", "ViT-B", lambda t: e5("MedSAM (full FT)", "encoder_decoder", t, t), None),
            ("RETFound", "MAE encoder + UPerNet", "ViT-L", lambda t: e5("RETFound + UPerNet", "encoder_decoder", t, t), None),
        ]),
    ]
    rows = []
    for title, kind, rs in spec:
        vals, summ = {}, {}
        for k in ("IDRiD", "DDR"):
            col, scol = [], []
            for i, r in enumerate(rs):
                ctx["cell"] = (title, i, k)
                col.append(r[3](k))
                scol.append(r[4](k) if r[4] is not None else float(np.nanmean(col[-1])))
            vals[k], summ[k] = np.array(col), np.array(scol)
        ctx.pop("cell", None)
        marks = {}
        for k in vals:   # best and second-best within the block only (decided 2026-10-14); no mark where one row stands alone
            for j in range(4):
                marks[(k, j)] = rank_marks(vals[k][:, j]) if np.isfinite(vals[k][:, j]).sum() >= 2 else (-1, -1)
            marks[(k, "m")] = rank_marks(summ[k]) if np.isfinite(summ[k]).sum() >= 2 else (-1, -1)
        rows.append(f"\\grouprow{{13}}{{{title}}}")
        for i, (name, how, backbone, *_fns) in enumerate(rs):
            cells = [f"\\famchip{{foundation}}\; {name}", how, backbone]
            for k in ("IDRiD", "DDR"):
                sim = sims.get((title, i, k), True)
                for j in range(4):
                    v = vals[k][i, j]
                    b, s2 = marks[(k, j)]
                    cells.append(tint(v, 0, 0.9) + num(v, "{:.2f}", sim, "best" if i == b else "second" if i == s2 else None) if np.isfinite(v) else "\\na")
                b, s2 = marks[(k, "m")]
                cells.append(num(summ[k][i], "{:.2f}", sim, "best" if i == b else "second" if i == s2 else None))
            rows.append(" & ".join(cells) + " \\\\")
    ref = D.internal()
    rows.append("\\midrule")
    any_ref = []
    for model in ("M2MRF", "nnU-Net"):
        cells = [f"\\famchip{{{D.FAMILY_OF[model]}}}\; {model}", "reference, Table~\\ref{tab:main}", "--"]
        for k in ("IDRiD", "DDR"):
            v = ref.loc[(model, k)].to_numpy()
            cells += [num(x, "{:.2f}") for x in v] + [num(v.mean(), "{:.2f}")]
        rows.append(" & ".join(cells) + " \\\\")
        parts = []
        for k in ("IDRiD", "DDR"):
            a = D.reference_any_lesion(model, k)
            parts.append("--" if a is None else f"{a:.2f}")
        any_ref.append(f"{model} {parts[0]} / {parts[1]}")
    write(
        "tab_foundation",
        rf"""\begin{{table*}}[t]
\centering\tablefont
\caption{{\textbf{{Foundation models under four kinds of use.}} The kinds are reported in separate blocks and are not ranked against each other: zero-shot use sees no test label (its density, area cap and wording are chosen on the validation pool); the upper-bound block reads the test labels (masks classed by their overlap with the ground truth, prompts derived from every ground-truth component) and is not a usable setting; the two trained settings follow the unified protocol. AUPR on the IDRiD and DDR test sets. \best{{Bold}} and \second{{underlined}} mark the best and second-best rows of each column within a block.}}
\label{{tab:foundation}}
\begin{{threeparttable}}
\begin{{tabular}}{{@{{}}l l l cccc c @{{\hspace{{10pt}}}} cccc c@{{}}}}
\toprule
 & & & \multicolumn{{5}}{{c}}{{\colh{{IDRiD test}}}} & \multicolumn{{5}}{{c}}{{\colh{{DDR test}}}} \\
\cmidrule(lr){{4-8}}\cmidrule(l){{9-13}}
\colh{{Model}} & \colh{{Use}} & \colh{{Backbone}} & MA & HE & EX & SE & \colh{{mAUPR}} & MA & HE & EX & SE & \colh{{mAUPR}} \\
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{tabular}}
\begin{{tablenotes}}[flushleft]\scriptsize
\item $^\ddagger$The zero-shot block is summarised by the any-lesion AUPR (union of the four classes against the pixelwise maximum of the maps); automatic masks carry no class, so their per-class cells are empty, and SAM~3's per-class cells are its own text prompts. The any-lesion AUPR of the references on the same test sets (IDRiD / DDR): {"; ".join(any_ref)} (-- where the run predates the metric).
\item $^\dagger$Each automatic mask is assigned the class it overlaps most in the ground truth, an upper bound for class-agnostic proposals. Scores of automatic masks use the predicted IoU.
\phnote
\end{{tablenotes}}
\end{{threeparttable}}
\end{{table*}}
""",
    )


# --------------------------------------------------- Table 7 factor summary
def table_factors():
    """Table 7. Each block lists the models that the experiment uses (docs/paper/16); cells are seed-0 values
    (or the seed mean when more seeds exist) followed by the change against the reference row."""
    cols = ["U-Net", "M2MRF", "RETFound + UPerNet"]

    def block(exp, levels, prefix, test, ref, labels=None, metric="aupr"):
        r = D.sel(experiment=exp, metric=metric, test_set=test)
        g = r.groupby(["model", "variant"])["value"].mean()
        present = [m for m in cols if m in g.index.get_level_values(0)]
        rows = []
        for lv, lab in zip(levels, labels or levels):
            cells = [("\\quad " + lab) + (" {\\scriptsize\\color{inkmuted}(ref.)}" if lv == ref else "")]
            vals, base = [], []
            for m in cols:
                if m not in present or (m, prefix + lv) not in g.index:
                    cells.append("--")
                    continue
                v, b = g[(m, prefix + lv)], g[(m, prefix + ref)]
                vals.append(v)
                base.append(b)
                cells.append(num(v, "{:.2f}") + ("" if lv == ref else " " + delta(v - b, True, "{:+.2f}")))
            vals, base = np.array(vals), np.array(base)
            d = vals.mean() - base.mean()
            cells.append((div_tint(d, 0.3) if lv != ref else "") + num(vals.mean(), "{:.2f}") + ("" if lv == ref else " {\\scriptsize " + signed(d, True) + "}"))
            rows.append(" & ".join(cells) + " \\\\")
        return rows

    rows = []
    rows.append("\\grouprow{5}{Annotation granularity of the training set (E4) --- mAUPR on IDRiD test, equal image passes; U-Net and M2MRF}")
    rows += block("E4", ["DDR", "DDR + FGADR (383)", "DDR + FGADR", "DDR + Retinal-Lesions", "FGADR", "Retinal-Lesions"], "", "IDRiD", "DDR",
                  ["DDR \\tier{fine}", "DDR + FGADR, 383 images \\tier{fine}\\tier{mixed}", "DDR + FGADR \\tier{fine}\\tier{mixed}", "DDR + Retinal-Lesions \\tier{fine}\\tier{coarse}", "FGADR \\tier{mixed}", "Retinal-Lesions \\tier{coarse}"])
    rows.append("\\grouprow{5}{Input resolution, long side in pixels (E7) --- mAUPR on DDR test; M2MRF and RETFound + UPerNet}")
    rows += block("E7", ["512", "1440", "native patches"], "res:", "DDR", "1440", ["512", "1,440", "native-resolution patches"])
    rows.append("\\grouprow{5}{Input resolution (E7) --- lesion-level recall of small lesions on DDR test}")
    rows += block("E7", ["512", "1440", "native patches"], "res:", "DDR", "1440", ["512", "1,440", "native-resolution patches"], metric="lesion_recall_small")
    rows.append("\\grouprow{5}{Training amount (E8) --- mAUPR on DDR test; U-Net and M2MRF}")
    rows += block("E8", ["DDR 10%", "DDR 50%", "DDR 100%", "fine merged"], "size:", "DDR", "DDR 100%",
                  ["DDR, 10\\% (38 images)", "DDR, 50\\% (192)", "DDR, 100\\% (383)", "+ IDRiD + TJDR standard field (599)"])
    rows.append("\\grouprow{5}{Loss, architecture fixed (E9) --- mAUPR on DDR test; U-Net}")
    rows += block("E9", ["CE", "Dice", "CE + Dice", "CE + per-image Dice"], "loss:", "DDR", "CE", ["cross-entropy", "batch Dice", "cross-entropy + batch Dice (unified protocol)", "cross-entropy + per-image Dice (registered recipe)"])
    head = " & ".join(f"\\famchip{{{D.FAMILY_OF[m]}}}\; {m.replace('RETFound + UPerNet', 'RETFound+UPer.')}" for m in cols)
    write(
        "tab_factors",
        rf"""\begin{{table*}}[t]
\centering\tablefont
\caption{{\textbf{{One factor at a time.}} One or two representative models are retrained with a single factor changed and everything else fixed to the unified protocol (seed 0). Each cell is the value followed by the change against the reference row of its block; -- marks a model the block does not use. The last column averages the models of the block; its shading follows the change. Differences smaller than twice the median seed standard deviation of the main table are not interpreted.}}
\label{{tab:factors}}
\begin{{threeparttable}}
\begin{{tabular}}{{@{{}}l ccc c@{{}}}}
\toprule
\colh{{Setting}} & {head} & \colh{{Mean}} \\
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{tabular}}
\begin{{tablenotes}}[flushleft]\scriptsize
\item Granularity of the labels used for training: \tier{{fine}} fine \quad \tier{{mixed}} mixed \quad \tier{{coarse}} coarse. In the training-amount block the number of image passes is the same in every row. The FGADR row of the granularity block is the FGADR-trained model of the cross-source experiment.
\phnote
\end{{tablenotes}}
\end{{threeparttable}}
\end{{table*}}
""",
    )


# ------------------------------------------------------ Methods table (real)
def table_recipes():
    rows = [
        # model, init, head, loss, optimizer, lr, wd, schedule, eff. batch, train patch, inference, steps IDRiD
        ("U-Net", "none", "4 sigmoid", "BCE + Dice$^a$", "Adam", "1e-4", "0", "constant", "4", "960", "whole canvas", "21,000"),
        ("DeepLabv3", "ImageNet, ResNet-101", "4 sigmoid", "BCE", "SGD", "7e-3", "1e-4", "poly 0.9", "16", "960", "whole canvas", "6,000"),
        ("HRNet-W48", "ImageNet", "4 sigmoid", "BCE", "SGD", "1e-2", "5e-4", "poly", "12", "960", "whole canvas", "7,000"),
        ("Swin-Unet", "ImageNet, Swin-T", "5 softmax", "0.4 CE + 0.6 Dice", "SGD", "1e-2", "1e-4", "poly 0.9", "24", "224", "sliding window", "\\tbd"),
        ("FCT", "none", "4 sigmoid", "BCE", "Adam", "1e-3", "0", "constant", "4", "384", "sliding window", "129,000"),
        ("M2MRF", "ImageNet, HRNet-W48", "4 sigmoid", "binary Dice", "SGD", "1e-2", "5e-4", "poly 0.9", "4", "960", "whole canvas", "21,000"),
        ("HACDR-Net", "none", "5 softmax", "weighted CE$^b$", "AdamW", "6e-5", "1e-2", "poly 1.0, warm-up", "1", "960", "whole canvas", "83,000"),
    ]
    fam = {"M2MRF": "fundus", "HACDR-Net": "fundus"}
    body = "\n".join(f"\\famchip{{{fam.get(r[0], 'general')}}}\\; " + " & ".join(r) + " \\\\" for r in rows)
    write(
        "tab_recipes",
        rf"""\begin{{table*}}[t]
\centering\tablefont
\caption{{\textbf{{What each model keeps under the unified protocol.}} Data, splits, the 1,440-pixel canvas, augmentation, the training amount in pixels, checkpoint selection and evaluation are shared (Methods). Loss, optimizer, learning rate, weight decay, schedule shape and effective batch are taken from each model's source recipe; the schedule is stretched over the shared number of steps. Models with a fixed input size train on patches of that size and predict with half-overlapping windows; all others train on $960\times960$ patches and predict on the whole canvas.}}
\label{{tab:recipes}}
\begin{{threeparttable}}
\begin{{tabular}}{{@{{}}l l l l l c c l c c l r@{{}}}}
\toprule
\colh{{Model}} & \colh{{Initialization}} & \colh{{Head}} & \colh{{Loss}} & \colh{{Optim.}} & \colh{{LR}} & \colh{{WD}} & \colh{{Schedule}} & \colh{{Batch}} & \colh{{Patch}} & \colh{{Inference}} & \colh{{Steps}} \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\begin{{tablenotes}}[flushleft]\scriptsize
\item Batch is the effective batch (gradient accumulation where one card cannot hold it). Steps is for IDRiD at 1,000 equivalent epochs; one equivalent epoch draws as many patch pixels as the training canvases contain, so steps = $\lceil\lceil$canvas pixels\,/\,patch pixels$\rceil$\,/\,batch$\rceil$ per epoch. DDR uses 400 equivalent epochs (83,600 steps at batch 4).
\item $^a$Per-image, per-class sigmoid Dice with smoothing 1; the other Dice losses use $10^{{-5}}$. $^b$IDRiD: class-weighted cross-entropy; DDR: the same plus the authors' NALoss with weight 0.2.
\end{{tablenotes}}
\end{{threeparttable}}
\end{{table*}}
""",
    )


CITE = {"IDRiD": "porwal2018idrid", "DDR": "li2019ddr", "TJDR-std": "mao2023tjdr", "TJDR-uwf": "mao2023tjdr", "FGADR": "zhou2021fgadr",
        "Retinal-Lesions": "wei2020retinallesionslesionnet", "DiaRetDB1": "kauppi2007diaretdb1", "MAPLES-DR": "maplesdr2024scientificdata"}


def main():
    for d in D.DATASETS:
        d["cite"] = CITE[d["name"]]
    table_datasets()
    table_models()
    table_original()
    table_main()
    table_independent()
    table_foundation()
    table_factors()
    table_recipes()


if __name__ == "__main__":
    main()
