"""Write tables/generated/supp_*.tex and generated/supp_notes.tex for supplement.tex (2026-10-26).

Every table is built from data/results.csv (collect_runs.py), the frozen repository facts (bench/configs,
bench/data/splits, literature/meta) and the small run files the handoff packages carry (runs/REFINED,
runs/CLIN, runs/audit, runs/*/recipe.json). Placeholder rows print in \\ph{} exactly as in the main tables.

    python scripts/make_supplement.py
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PAPER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PAPER / "figures"))
sys.path.insert(0, str(PAPER / "scripts"))

from retifig import data as D  # noqa: E402
from make_tables import num, sd, write  # noqa: E402

REPO = D.REPO
OUT = PAPER / "tables" / "generated"
GEN = PAPER / "generated"
SET_NAMES = {"TJDR-std": "TJDR std", "TJDR-uwf": "TJDR UWF", "Retinal-Lesions": "Retinal-L.", "MAPLES-DR": "MAPLES", "DiaRetDB1": "DiaRetDB1"}
RUN_NAME = {  # main-table models -> B-std / E5 run slug (seed 0), for recipe.json and timing
    "U-Net": "BS_unet", "DeepLabv3": "BS_deeplabv3", "HRNet-W48": "BS_hrnet", "Swin-Unet": "BS_swin_unet", "FCT": "BS_fct",
    "nnU-Net": "NN_default", "nnU-Net ResEnc-M": "NN_resencm", "M2MRF": "BS_m2mrf", "HACDR-Net": "BS_hacdr", "HRDecoder": "BS_hrdecoder",
    "PMCNet": "BS_pmcnet", "WSRFNet": "BS_wsrfnet", "CLC-Net": "BS_clcnet", "TC-Net": "BS_tcnet", "SAMed": "BS_samed",
    "SAM-Adapter": "BS_sam_adapter", "SAM2-UNet": "BS_sam2_unet", "RETFound + UPerNet": "BS_retfound", "MedSAM (full FT)": "BS_medsam",
}


NNUNET_PLAN = {  # from "This is the configuration used by this training" in runs/NN_*/console.log (2026-10-26)
    ("nnU-Net", "IDRiD"): {"patch": "1024$\\times$1536", "batch": 2},
    ("nnU-Net", "DDR"): {"patch": "1024$\\times$1536", "batch": 2},
    ("nnU-Net ResEnc-M", "IDRiD"): {"patch": "384$\\times$640", "batch": 13},
    ("nnU-Net ResEnc-M", "DDR"): {"patch": "512$\\times$512", "batch": 12},
}


def esc(s):
    s = str(s)
    for a, b in (("\\", "\\textbackslash{}"), ("&", "\\&"), ("%", "\\%"), ("_", "\\_"), ("#", "\\#"), ("$", "\\$"), ("{", "\\{"), ("}", "\\}"), ("~", "\\textasciitilde{}"), ("^", "\\textasciicircum{}")):
        s = s.replace(a, b)
    return s


def short(s, n):
    s = str(s)
    return s if len(s) <= n else s[: n - 1].rstrip() + "\u2026"


def df():
    return D.results()


def cell(rows, fmt="{:.3f}"):
    """num() of the mean of a row set, placeholder-aware; \\na when empty."""
    if rows is None or len(rows) == 0:
        return "\\na"
    return num(float(rows["value"].mean()), fmt=fmt, sim=D.simulated(rows))


def longtable(name, caption, label, colspec, header, body_rows, notes=None, font="\\scriptsize", stretch=1.08, landscape=False):
    """A longtable that may break across pages; header repeated; optional notes after the table."""
    head = " & ".join(header) + " \\\\"
    lines = [
        ("\\begin{landscape}" if landscape else ""),
        "{" + font + "\\setlength{\\tabcolsep}{3.2pt}\\setlength{\\LTcapwidth}{\\textwidth}\\renewcommand{\\arraystretch}{" + str(stretch) + "}",
        "\\begin{longtable}{" + colspec + "}",
        "\\caption{" + caption + "}\\label{" + label + "}\\\\",
        "\\toprule", head, "\\midrule", "\\endfirsthead",
        "\\multicolumn{" + str(len(header)) + "}{l}{\\textit{continued}}\\\\", "\\toprule", head, "\\midrule", "\\endhead",
        "\\midrule", "\\multicolumn{" + str(len(header)) + "}{r}{\\textit{continued on the next page}}\\\\", "\\endfoot",
        "\\bottomrule", "\\endlastfoot",
    ]
    lines += body_rows
    lines += ["\\end{longtable}", "}"]
    if notes:
        lines += ["\\vspace{-6pt}\\noindent{\\scriptsize " + notes + "}"]
    lines += [("\\end{landscape}" if landscape else "")]
    write(name, "\n".join(lines))


# ---------------------------------------------------------------- S1 literature audit per paper
def s1_literature():
    a = pd.read_csv(REPO / "literature" / "meta" / "paper_audit.tsv", sep="\t", dtype=str, keep_default_na=False)
    a = a.sort_values(["year", "paper"])
    rows = []
    for r in a.itertuples():
        paper = r.paper.replace("_", " ").replace("-", "-")
        n = re.match(r"\s*(\d+)", r.n_compared)
        rows.append(" & ".join([esc(short(paper, 30)), esc(r.year), esc(short(r.venue, 16)), esc(short(r.datasets, 20)),
                                esc(short(r.idrid_split, 10)), esc(short(r.ddr_split, 10)), esc(short(r.input_resolution, 44)),
                                esc(short(r.aupr_impl, 14)), esc(short(r.per_image_or_pooled, 10)), esc(short(r.baseline_rerun_class, 12)), (n.group(1) if n else esc(short(r.n_compared, 6)))]) + " \\\\")
    longtable("supp_s1_literature",
              "\\textbf{Literature audit: the experimental setup stated by each method paper in the ledger.} Split, input resolution, AUPR implementation and the provenance of baseline numbers as extracted from the paper's text and tables (literature/meta/paper\\_audit.tsv in the repository holds the verbatim evidence for every cell).",
              "stab:literature", "p{34mm} c p{20mm} p{24mm} p{14mm} p{14mm} p{52mm} p{18mm} p{14mm} p{16mm} c",
              ["Paper", "Year", "Venue", "Datasets", "IDRiD split", "DDR split", "Input resolution", "AUPR impl.", "Pooling", "Baselines", "$n$"],
              rows, notes="``not stated'' means the paper gives no information. Baselines: how the comparison numbers were obtained (rerun, copied, mixed, not stated). $n$, number of compared methods.", landscape=True)


# ---------------------------------------------------------------- S2 reported values per method
def s2_reported():
    r = pd.read_csv(REPO / "literature" / "meta" / "reported_results.tsv", sep="\t", dtype=str, keep_default_na=False)
    r = r[(r.metric == "AUPR") & (r.lesion == "mean") & r.dataset.isin(["IDRiD", "DDR"])]
    r["v"] = pd.to_numeric(r.value_norm, errors="coerce")
    r = r.dropna(subset=["v"])
    rows = []
    for (method, ds), g in r.groupby(["method_norm", "dataset"]):
        if len(g) < 2:
            continue
        prov = g.provenance_derived.value_counts()
        rows.append(" & ".join([esc(method), ds, str(len(g)), num(g.v.min(), sim=False), num(g.v.max(), sim=False), num(g.v.max() - g.v.min(), sim=False),
                                str(int(prov.get("rerun", 0))), str(int(prov.get("copied", 0) + prov.get("copied once", 0) + prov.get("copied again", 0))), str(int(prov.get("unverifiable", 0)))]) + " \\\\")
    longtable("supp_s2_reported",
              "\\textbf{One method, many reported values.} Every method that appears with a mean AUPR on IDRiD or DDR in at least two papers of the ledger: the number of papers, the lowest and highest value, their spread, and how many of the values were produced by the reporting paper (rerun), carried from an earlier paper (copied) or cite a source outside the ledger (unverifiable).",
              "stab:reported", "l l c c c c c c c",
              ["Method", "Dataset", "Papers", "Min", "Max", "Spread", "Rerun", "Copied", "Unverif."], rows,
              notes="Values as printed, normalised to [0, 1]. The full extraction, one row per paper, baseline, dataset, class and metric, is literature/meta/reported\\_results.tsv.")


# ---------------------------------------------------------------- S3 training amount of every main-table run
def s3_runs():
    rows = []
    for model in D.MAIN_MODELS:
        slug = RUN_NAME.get(model)
        if slug is None:
            continue
        for train in D.TRAIN_MAIN:
            if slug.startswith("NN_"):
                run = REPO / "runs" / f"{slug}_{train.lower()}_fold0"
            else:
                run = REPO / "runs" / f"{slug}_{train.lower()}_seed0"
            rec = run / "recipe.json"
            env = run / "environment.json"
            if slug.startswith("NN_"):
                # nnU-Net writes no recipe.json; its plan is recorded in runs/NN_*/console.log (NNUNET_PLAN below) and
                # its schedule is the default 1,000 epochs x 250 steps; the GPU hours are the E10 rows
                plan = NNUNET_PLAN[(model, train)]
                hours = D.sel(experiment="E10", model=model, metric="train_gpu_h", train_set=train)["value"]
                rows.append(" & ".join([esc(model), train, "250,000", str(plan["batch"]), "1", str(plan["batch"]), plan["patch"], "0.01",
                                        "\\na$^{\\rm n}$", (f"{float(hours.iloc[0]):.1f}" if len(hours) else "\\na")]) + " \\\\")
                continue
            if not rec.is_file():
                rows.append(" & ".join([esc(model), train] + ["\\na"] * 8) + " \\\\")
                continue
            recipe = json.loads(rec.read_text())
            e = json.loads(env.read_text()) if env.is_file() else {}
            steps = recipe.get("iterations") or recipe.get("steps_per_epoch", 0) * recipe.get("epochs", 0)
            loader = recipe.get("batch_size", "")
            eff = recipe.get("effective_batch_size", loader)
            acc = recipe.get("accumulation", 1)
            patch = ""
            for d in recipe.get("deviations", []):
                m = re.search(r"patches of (\d+x\d+)", d)
                if m:
                    patch = m.group(1).replace("x", "$\\times$")
            lr = recipe.get("lr", "")
            best = e.get("best_val_maupr")
            best_ep = e.get("best_epoch", "")
            hours = e.get("seconds")
            rows.append(" & ".join([esc(model), train, f"{int(steps):,}" if steps else "\\na", str(loader), str(acc), str(eff), patch or "\\na",
                                    f"{lr:g}" if isinstance(lr, (int, float)) else esc(lr), (f"{best:.3f}" if best is not None else "\\na"),
                                    (f"{hours / 3600:.1f}" if hours else "\\na")]) + " \\\\")
    longtable("supp_s3_runs",
              "\\textbf{Training amount, batch and selection of every main-table run (seed 0).} Steps are the shared pixel budget expressed in the model's patch size (Methods); the loader batch is the largest that fits a 40\\,GB card and is accumulated to the recipe's effective batch; the selected checkpoint is the validation \\mAUPR{} maximum over 20 evaluations. nnU-Net follows its own 1,000-epoch schedule and planning.",
              "stab:runs", "l l r c c c c c c c",
              ["Model", "Source", "Steps", "Loader batch", "Accum.", "Eff. batch", "Patch", "LR", "Best val.\\ \\mAUPR{}", "GPU h"], rows,
              notes="Every run's full recipe.json (deviations from the source recipe included) is in the repository's run records. nnU-Net: patch and batch from its own plan for each dataset (runs/NN\\_*/console.log), 1,000 epochs of 250 steps, SGD at 0.01 with polynomial decay; $^{\\rm n}$ its checkpoint is nnU-Net's own (the final checkpoint of its schedule), so no validation \\mAUPR{} maximum applies.")


# ---------------------------------------------------------------- S4 per-class AUPR with CI on every test set
def s4_perclass():
    t = df()
    r = t[(t.experiment.isin(["E2", "E5"])) & (t.variant.isin(D.MAIN_VARIANTS)) & (t.seed == 0) & t.model.isin(D.MAIN_MODELS)]
    rows = []
    for train in D.TRAIN_MAIN:
        rows.append(f"\\multicolumn{{{2 + 2 * len(D.CLASSES) + 1}}}{{l}}{{\\textbf{{Trained on {train}}}}}\\\\")
        for model in D.MAIN_MODELS:
            sub = r[(r.model == model) & (r.train_set == train)]
            if sub.empty:
                continue
            for test in D.TESTS:
                st = sub[sub.test_set == test]
                if st.empty:
                    continue
                cells = []
                vals = []
                for cls in D.CLASSES:
                    v = st[(st.cls == cls) & (st.metric == "aupr")]
                    lo = st[(st.cls == cls) & (st.metric == "aupr_ci_lo")]
                    hi = st[(st.cls == cls) & (st.metric == "aupr_ci_hi")]
                    cells.append(cell(v))
                    cells.append(("\\scriptsize " + num(float(lo.value.iloc[0]), fmt="{:.2f}", sim=D.simulated(lo)) + "--" + num(float(hi.value.iloc[0]), fmt="{:.2f}", sim=D.simulated(hi))) if len(lo) and len(hi) else "\\na")
                    if len(v):
                        vals.append(float(v.value.mean()))
                mean = num(np.mean(vals), sim=D.simulated(st[st.metric == "aupr"])) if len(vals) == 4 else "\\na"
                rows.append(" & ".join([esc(model), SET_NAMES.get(test, test)] + cells + [mean]) + " \\\\")
    longtable("supp_s4_perclass",
              "\\textbf{Per-class AUPR of every main-table model on every test set (seed 0), with 95\\% bootstrap intervals.} Exact pooled AUPR inside the field of view; the interval is the percentile interval of 2,000 bootstrap replicates over test images. Both training sources; the internal test is the row whose test set is the training source.",
              "stab:perclass", "l l " + "c c " * len(D.CLASSES) + "c",
              ["Model", "Test set"] + sum([[c, "95\\% CI"] for c in D.CLASSES], []) + ["\\mAUPR{}"], rows,
              notes="TJDR UWF, ultra-widefield images predicted at native resolution with sliding windows. Retinal-L., Retinal-Lesions without its grey-level 127 regions. DiaRetDB1, pixel AUPR against the $\\geq$ 0.75 expert consensus (the official image-level protocol is Supplementary Table~13).", landscape=True)


# ---------------------------------------------------------------- S5 secondary metrics on the internal tests
def s5_secondary():
    t = df()
    r = t[(t.experiment.isin(["E2", "E5"])) & (t.variant.isin(D.MAIN_VARIANTS)) & t.model.isin(D.MAIN_MODELS)]
    r = r[r.test_set == r.train_set]
    metrics = [("aupr", "\\mAUPR{}"), ("aupr_11pt", "11-thr."), ("dice_valopt", "Dice"), ("iou_valopt", "IoU"), ("lesion_precision", "Lesion prec."),
               ("lesion_recall_small", "Recall S"), ("lesion_recall_medium", "Recall M"), ("lesion_recall_large", "Recall L"), ("img_auc", "Image AUC")]
    rows = []
    for train in D.TRAIN_MAIN:
        rows.append(f"\\multicolumn{{{2 + len(metrics)}}}{{l}}{{\\textbf{{Trained on {train} (internal test; mean over seeds)}}}}\\\\")
        for model in D.MAIN_MODELS:
            sub = r[(r.model == model) & (r.train_set == train)]
            if sub.empty:
                continue
            seeds = sub.seed.nunique()
            cells = []
            for metric, _ in metrics:
                m = sub[(sub.metric == metric) & (sub.cls.isin(D.CLASSES))]
                if m.empty:
                    cells.append("\\na")
                    continue
                per_seed = m.groupby("seed").value.mean()
                cells.append(num(per_seed.mean(), sim=D.simulated(m)) + (sd(per_seed.std(), D.simulated(m)) if seeds > 1 else ""))
            rows.append(" & ".join([esc(model), str(seeds)] + cells) + " \\\\")
    longtable("supp_s5_secondary",
              "\\textbf{Secondary metrics of every main-table model on the internal tests (mean over the four classes and over seeds).} 11-thr., the 11-threshold AUPR of the M2MRF evaluation code on the same maps; Dice and IoU at the threshold chosen on the validation split; lesion-level precision and recall by lesion size (S, M, L: small, medium, large; Methods) at the same threshold; image-level AUC of the maximum probability inside the field of view.",
              "stab:secondary", "l c " + "c " * len(metrics),
              ["Model", "Seeds"] + [m for _, m in metrics], rows,
              notes="$\\pm$, standard deviation over seeds where more than one seed was run.")


# ---------------------------------------------------------------- S6 registered vs revised recipes
def s6_registered():
    t = df()
    rows = []
    for model, label in (("U-Net", "U-Net"), ("SAM2-UNet", "SAM2-UNet"), ("MedSAM (full FT)", "MedSAM")):
        for train in D.TRAIN_MAIN:
            reg = t[(t.experiment.isin(["E2", "E5"])) & (t.variant == "registered") & (t.model == model) & (t.train_set == train) & (t.test_set == train) & (t.metric == "aupr")]
            rev = t[(t.experiment.isin(["E2", "E5"])) & (t.variant.isin(D.MAIN_VARIANTS)) & (t.model == model) & (t.train_set == train) & (t.test_set == train) & (t.metric == "aupr") & (t.seed == 0)]
            for name, sub in (("registered", reg), ("revised (main table)", rev)):
                if sub.empty:
                    continue
                cells = [cell(sub[sub.cls == c]) for c in D.CLASSES]
                m = sub[sub.cls.isin(D.CLASSES)]
                rows.append(" & ".join([esc(label), train, name] + cells + [cell(m)]) + " \\\\")
    longtable("supp_s6_registered",
              "\\textbf{Registered and revised recipes of the three models whose per-image loss term emptied a rare-class channel (seed 0, internal tests).} The registered recipe is the one frozen before training (per-image Dice or IoU with smoothing 1); the revised recipe pools the term over the batch (Methods, rare-class collapse). The main tables carry the revised runs; the registered runs stay in the repository under the \\texttt{\\_registered} suffix.",
              "stab:registered", "l l l c c c c c",
              ["Model", "Source", "Recipe"] + D.CLASSES + ["\\mAUPR{}"], rows)


# ---------------------------------------------------------------- S7 factor cells
def s7_factors():
    t = df()
    exps = [("E3b", "Training source"), ("E4", "Annotation granularity"), ("E7", "Input resolution"), ("E8", "Training amount"), ("E9", "Loss")]
    rows = []
    for exp, title in exps:
        r = t[(t.experiment == exp) & (t.metric == "aupr") & t.cls.isin(D.CLASSES) & (t.source == "measured")]
        if r.empty:
            continue
        rows.append(f"\\multicolumn{{{4 + len(D.CLASSES) + 1}}}{{l}}{{\\textbf{{{title}}}}}\\\\")
        for (model, variant, train), g in r.groupby(["model", "variant", "train_set"], sort=False):
            for test in D.TESTS:
                st = g[g.test_set == test]
                if st.empty:
                    continue
                cells = [cell(st[st.cls == c]) for c in D.CLASSES]
                rows.append(" & ".join([esc(model), esc(variant), esc(train), SET_NAMES.get(test, test)] + cells + [cell(st)]) + " \\\\")
    longtable("supp_s7_factors",
              "\\textbf{Every factor cell, per class (seed 0).} One factor is varied at a time on the representative models at the DDR pixel budget (Methods); the variant column names the level, the training-set column the data the cell was trained on. Each cell is listed on every test set it was applied to; in-source tests of merged cells are the rows whose test set was part of the training set.",
              "stab:factors", "l l l l c c c c c",
              ["Model", "Level", "Trained on", "Test set"] + D.CLASSES + ["\\mAUPR{}"], rows, landscape=True)


# ---------------------------------------------------------------- S8 computational cost
def s8_cost():
    t = df()
    c = t[t.experiment == "E10"]
    rows = []
    for model in D.MAIN_MODELS:
        g = c[c.model == model]
        if g.empty:
            continue
        def v(metric, train="", fmt="{:.1f}"):
            s = g[(g.metric == metric) & (g.train_set.fillna("") == train)]
            return cell(s, fmt=fmt) if len(s) else "\\na"
        rows.append(" & ".join([esc(model), v("params_m", fmt="{:.1f}"), v("params_trainable_m", fmt="{:.1f}"), v("train_gpu_h", "IDRiD"), v("train_gpu_h", "DDR"),
                                v("flops_g", fmt="{:.0f}"), v("peak_mem_gb"), v("infer_s_per_image", fmt="{:.2f}")]) + " \\\\")
    longtable("supp_s8_cost",
              "\\textbf{Computational cost.} Parameters and trainable parameters under the unified protocol, training time of one seed on each source (40\\,GB A100, one task per card), and, where the cost audit has run, floating-point operations for one 1,440-pixel canvas, peak training memory and inference time per image.",
              "stab:cost", "l c c c c c c c",
              ["Model", "Params (M)", "Trainable (M)", "GPU h IDRiD", "GPU h DDR", "GFLOPs", "Peak mem.\\ (GB)", "Infer.\\ (s/image)"], rows,
              notes="nnU-Net follows its own schedule of 1,000 epochs rather than the pixel budget. Cells without a value were not measured; the cost audit (FLOPs, memory, inference) is pending.")


# ---------------------------------------------------------------- S9 refined IDRiD
def s9_refined():
    t = df()
    r = t[(t.experiment == "REF") & (t.metric == "aupr") & t.cls.isin(D.CLASSES)]
    rows = []
    for model, g in r.groupby("model", sort=False):
        ref = g[g.variant == "refined_labels"]
        org = g[g.variant == "original_labels_1024"]
        if ref.empty or org.empty:
            continue
        cells = []
        for c in D.CLASSES:
            cells += [cell(org[org.cls == c]), cell(ref[ref.cls == c])]
        mo, mr = float(org.value.mean()), float(ref.value.mean())
        rows.append(" & ".join([esc(model)] + cells + [num(mo, sim=False), num(mr, sim=False), f"{mr - mo:+.3f}".replace("-", "$-$")]) + " \\\\")
    diff = ""
    path = REPO / "runs" / "REFINED" / "label_difference.json"
    if path.is_file():
        d = json.loads(path.read_text())
        per = d.get("per_image", {})
        agg = {c: {"changed": 0, "iou": [], "ref_px": 0, "org_px": 0, "ref_c": 0, "org_c": 0} for c in D.CLASSES}
        for _img, rec in per.items():
            for c in D.CLASSES:
                x = rec["classes"].get(c)
                if not x:
                    continue
                if x["refined_pixels"] != x["original_pixels"] or (x["iou"] is not None and x["iou"] < 1):
                    agg[c]["changed"] += 1
                agg[c]["ref_px"] += x["refined_pixels"]; agg[c]["org_px"] += x["original_pixels"]
                agg[c]["ref_c"] += x["refined_components"]; agg[c]["org_c"] += x["original_components"]
        diff = "Label differences over the 27 test images: " + "; ".join(
            f"{c}: {a['changed']} images changed, {a['org_px']:,} $\\to$ {a['ref_px']:,} labelled pixels, {a['org_c']:,} $\\to$ {a['ref_c']:,} components" for c, a in agg.items()) + "."
    longtable("supp_s9_refined",
              "\\textbf{Refined IDRiD (label revision): per-class AUPR of every IDRiD-trained model against the official and the refined labels through the same 1,024-pixel geometry.} Test maps of the main-table checkpoints (seed 0), moved through the recovered crop and scored with the protocol evaluator against both label sets; no model is retrained or re-run.",
              "stab:refined", "l " + "c c " * len(D.CLASSES) + "c c c",
              ["Model"] + sum([[f"{c} off.", f"{c} ref."] for c in D.CLASSES], []) + ["\\mAUPR{} off.", "\\mAUPR{} ref.", "$\\Delta$"], rows,
              notes=diff + " The crop geometry of every image (all 27: padded to a square and resized; NCC $\\geq$ 0.9988) is runs/REFINED/RefinedIDRiD\\_geometry.json.", landscape=True)


# ---------------------------------------------------------------- S10 encoders
def s10_encoders():
    t = df()
    r = t[(t.experiment == "E5") & (t.variant == "encoder_control") & (t.metric == "aupr") & t.cls.isin(D.CLASSES)]
    rows = []
    for train in D.TRAIN_MAIN:
        rows.append(f"\\multicolumn{{{2 + len(D.CLASSES) + 1}}}{{l}}{{\\textbf{{Trained on {train}}}}}\\\\")
        for enc in D.ENCODERS:
            g = r[(r.model == enc) & (r.train_set == train)]
            for test in [train] + D.FINE_INDEPENDENT:
                st = g[g.test_set == test]
                if st.empty:
                    continue
                rows.append(" & ".join([esc(enc), SET_NAMES.get(test, test)] + [cell(st[st.cls == c]) for c in D.CLASSES] + [cell(st)]) + " \\\\")
    longtable("supp_s10_encoders",
              "\\textbf{Encoder comparison: four ViT-L encoders under one UPerNet head, recipe and pixel budget (seed 0).} Per-class AUPR on the internal test and on the three independent fine-grained sets. RETFound-MAE is the main-table RETFound run of the same source; the other three are the control cells. The pre-specified contrasts are RETFound-MAE minus MAE-IN1k and RETFound-DINOv2 minus DINOv2.",
              "stab:encoders", "l l c c c c c", ["Encoder", "Test set"] + D.CLASSES + ["\\mAUPR{}"], rows)


# ---------------------------------------------------------------- S11 ensembles and the reference model
def s11_ensembles():
    t = df()
    r = t[(t.experiment == "ENS") & (t.metric == "aupr") & t.cls.isin(D.CLASSES)]
    comps = ["E-all", "E-top5", "E-top3", "E-1perfam", "E-fam:general", "E-fam:auto", "E-fam:fundus", "E-fam:foundation"]
    rows = []
    for train in D.TRAIN_MAIN:
        rows.append(f"\\multicolumn{{{1 + len(D.TESTS)}}}{{l}}{{\\textbf{{Members trained on {train}}}}}\\\\")
        for comp in comps:
            g = r[(r.model == comp) & (r.train_set == train)]
            if g.empty:
                continue
            cells = []
            for test in D.TESTS:
                st = g[g.test_set == test]
                mark = "$^{\\rm p}$" if len(st) and st.variant.str.startswith("ensemble:without").any() else ""
                cells.append(cell(st) + mark)
            rows.append(" & ".join([esc(comp)] + cells) + " \\\\")
    n8 = t[(t.experiment == "N8") & (t.metric == "aupr") & t.cls.isin(D.CLASSES)]
    if not n8.empty:
        rows.append(f"\\multicolumn{{{1 + len(D.TESTS)}}}{{l}}{{\\textbf{{Reference model: trained on IDRiD + DDR + TJDR std merged (DDR budget and validation)}}}}\\\\")
        for (model, variant), g in n8.groupby(["model", "variant"], sort=False):
            cells = [cell(g[g.test_set == test]) for test in D.TESTS]
            rows.append(" & ".join([esc(f"{model} ({variant.split(':', 1)[1]})")] + cells) + " \\\\")
    longtable("supp_s11_ensembles",
              "\\textbf{Ensembles of the main-table probability maps and the reference model, \\mAUPR{} on every test set.} Compositions fixed before any ensemble was scored (Methods): all 18 registered members, the five and three best by validation \\mAUPR{}, the best of each family, and each family alone. The reference model's two cells are trained on the three fine-grained sources merged; their ensemble is the pixel mean of their maps.",
              "stab:ensembles", "l " + "c " * len(D.TESTS), ["Composition"] + [SET_NAMES.get(s, s) for s in D.TESTS], rows,
              notes="$^{\\rm p}$, computed without HACDR-Net, whose maps for that set could not be regenerated (members named in the run record). In-source test sets of the reference model (IDRiD, DDR, TJDR std) are the official test splits of sources that entered its training.")


# ---------------------------------------------------------------- S12 DR-grade strata
def s12_strata():
    path = REPO / "runs" / "CLIN" / "strata_summary.json"
    if not path.is_file():
        return
    s = pd.DataFrame(json.loads(path.read_text()))
    s = s[s.run.str.match(r"^(BS|NN)_")]
    slug_to_model = {v: k for k, v in RUN_NAME.items()}
    rows = []
    for test in ["DDR", "MAPLES-DR", "FGADR", "Retinal-Lesions"]:
        sub = s[s.test_set == test]
        if sub.empty:
            continue
        n01, n24 = int(sub.grade_0_1_n.iloc[0]), int(sub.grade_2_4_n.iloc[0])
        rows.append(f"\\multicolumn{{7}}{{l}}{{\\textbf{{{SET_NAMES.get(test, test)}: {n01} images of grades 0--1, {n24} of grades 2--4}}}}\\\\")
        for r in sub.sort_values("run").itertuples():
            m = re.match(r"^(BS_[a-z0-9_]+?|NN_[a-z]+)_(idrid|ddr)_(seed0|fold0)$", r.run)
            if not m:
                continue
            model = slug_to_model.get(m.group(1), m.group(1))
            train = "IDRiD" if m.group(2) == "idrid" else "DDR"
            rows.append(" & ".join([esc(model), train, num(r.grade_0_1_mAUPR, sim=False), num(r.grade_2_4_mAUPR, sim=False),
                                    num(r.grade_0_1_img_auc, sim=False), num(r.grade_2_4_img_auc, sim=False), f"{r.grade_2_4_mAUPR - r.grade_0_1_mAUPR:+.3f}".replace("-", "$-$")]) + " \\\\")
    longtable("supp_s12_strata",
              "\\textbf{Results by DR grade (seed 0).} On the test sets that carry a DR grade per image, \\mAUPR{} and image-level AUC recomputed within grades 0--1 and within grades 2--4 from the stored per-image histograms, with the thresholds of the full test; ungradable images enter neither stratum.",
              "stab:strata", "l l c c c c c",
              ["Model", "Trained on", "\\mAUPR{} 0--1", "\\mAUPR{} 2--4", "Image AUC 0--1", "Image AUC 2--4", "$\\Delta$ \\mAUPR{}"], rows,
              notes="Image-level AUC within a stratum is undefined when every image of the stratum carries the class or none does; such cells read 0 or 1 in the record and should be read with the stratum size.")


# ---------------------------------------------------------------- S13 human reference
def s13_experts():
    t = df()
    h = t[t.experiment == "HUMAN"]
    rows = []
    for model, g in h.groupby("model", sort=False):
        cells = [cell(g[(g.cls == c) & (g.metric == "aupr")]) for c in D.CLASSES]
        m = g[(g.metric == "aupr") & g.cls.isin(D.CLASSES)]
        auc = g[(g.metric == "img_auc") & g.cls.isin(D.CLASSES)]
        dice = g[(g.metric == "dice_valopt") & g.cls.isin(D.CLASSES)]
        rows.append(" & ".join([esc(model)] + cells + [cell(m), cell(auc), cell(dice)]) + " \\\\")
    # the models against the four-expert consensus, for reference
    r = t[(t.experiment.isin(["E2", "E5"])) & (t.variant.isin(D.MAIN_VARIANTS)) & (t.seed == 0) & (t.test_set == "DiaRetDB1") & t.model.isin(D.MAIN_MODELS)]
    for train in D.TRAIN_MAIN:
        g = r[r.train_set == train]
        if g.empty:
            continue
        m = g[(g.metric == "aupr") & g.cls.isin(D.CLASSES)].groupby("model").value.mean()
        auc = g[(g.metric == "img_auc") & g.cls.isin(D.CLASSES)].groupby("model").value.mean()
        rows.append(f"\\multicolumn{{8}}{{l}}{{\\textit{{Models trained on {train}, against the four-expert consensus (median over {len(m)} models; range in brackets)}}}}\\\\")
        rows.append(" & ".join(["models"] + ["\\na"] * 4 + [f"{m.median():.3f} [{m.min():.2f}--{m.max():.2f}]", f"{auc.median():.3f} [{auc.min():.2f}--{auc.max():.2f}]", "\\na"]) + " \\\\")
    longtable("supp_s13_experts",
              "\\textbf{Human reference on DiaRetDB1: each expert against the consensus of the other three.} The expert's own markings, with their confidence levels mapped to scores, are the probability map and the consensus of the other three experts ($\\geq$ 0.75) is the ground truth; thresholds come from the official training list. A map with four confidence levels gives a four-point precision--recall curve, so the pixel-level AUPR is a lower bound.",
              "stab:experts", "l c c c c c c c",
              ["Reader"] + D.CLASSES + ["\\mAUPR{}", "Image AUC", "Dice"], rows)


# ---------------------------------------------------------------- S14 label-style audit
def s14_labels():
    path = REPO / "runs" / "audit" / "label_style.json"
    if not path.is_file():
        return
    d = json.loads(path.read_text())
    rows = []
    for src, rec in d["sources"].items():
        for c in D.CLASSES:
            x = rec["per_class"].get(c)
            if not x:
                continue
            rd = x["rel_diameter"] or {}
            rows.append(" & ".join([esc(src), c, f"{rec['n_images']:,}", f"{x['n_components']:,}", f"{x['per_image_mean']:.1f}",
                                    (f"{x['per_image_mean_positive']:.1f}" if x["per_image_mean_positive"] is not None else "\\na"), str(x["images_positive"]),
                                    (f"{rd['p10'] * 100:.2f}" if rd else "\\na"), (f"{rd['p50'] * 100:.2f}" if rd else "\\na"), (f"{rd['p90'] * 100:.2f}" if rd else "\\na"),
                                    (f"{x['area_px_median']:.0f}" if x["area_px_median"] is not None else "\\na")]) + " \\\\")
    longtable("supp_s14_labels",
              "\\textbf{Annotation style of every source: connected components per image and their size.} All images of each source, every id once; 8-connected components of the prepared masks with ignore regions removed; the relative diameter is the component's equivalent diameter divided by the long side of the field-of-view box, in percent (Methods, annotation style).",
              "stab:labels", "l l r r c c r c c c r",
              ["Source", "Class", "Images", "Components", "Per image", "Per positive image", "Positive images", "Diam.\\ p10 (\\%)", "p50 (\\%)", "p90 (\\%)", "Area px (median)"], rows)


# ---------------------------------------------------------------- S15 leak audit and exclusions
def s15_leaks():
    path = REPO / "bench" / "data" / "splits" / "leak_exclusions.json"
    if not path.is_file():
        return
    d = json.loads(path.read_text())
    rows = []
    for source, per in d["counts"].items():
        for test, n in per.items():
            if n:
                rows.append(" & ".join([esc(source), esc(test), str(n)]) + " \\\\")
    for source in d["by_training_source"]:
        if not any(d["counts"].get(source, {}).values()):
            rows.append(" & ".join([esc(source), "every test set", "0"]) + " \\\\")
    ev = d.get("evidence", [])
    top = sorted(ev, key=lambda e: -e.get("vessel_ncc", 0))[:12]
    ev_text = "; ".join(f"{esc(e['source'])} {esc(e['seen'])} $\\leftrightarrow$ {esc(e['test_set'])} {esc(e['test_id'])} ({e['vessel_ncc']:.2f})" for e in top)
    longtable("supp_s15_leaks",
              "\\textbf{Near-duplicate audit between training sources and test sets, and the images excluded.} Vessel signatures (green-channel black top-hat at 192 $\\times$ 192 inside the field of view) are correlated between every train or validation image of a source and every test image; a test image at or above the pair's threshold is left out of that source's independent tests (default 0.5; 0.33 within FGADR, whose images are near-duplicates of each other at scale).",
              "stab:leaks", "l l r", ["Training source", "Test set", "Excluded images"], rows,
              notes=f"Audit: {esc(d['audit'].get('command', ''))}, {d['audit'].get('n_images', '')} images. Strongest pairs (vessel NCC): {ev_text}. The full lists are bench/data/splits/leak\\_exclusions.json.")


# ---------------------------------------------------------------- S16 author weights per test set
def s16_authors():
    t = df()
    r = t[(t.experiment == "E1") & (t.variant == "this_protocol") & (t.metric == "aupr") & t.cls.isin(D.CLASSES)]
    rows = []
    for model in ("M2MRF", "HRDecoder"):
        for train in D.TRAIN_MAIN:
            g = r[(r.model == model) & (r.train_set == train)]
            for test in D.TESTS:
                st = g[g.test_set == test]
                if st.empty:
                    continue
                rows.append(" & ".join([esc(model), train, SET_NAMES.get(test, test)] + [cell(st[st.cls == c]) for c in D.CLASSES] + [cell(st)]) + " \\\\")
    longtable("supp_s16_authors",
              "\\textbf{Released checkpoints on every test set, per class, under this protocol.} Each checkpoint is run through its authors' own test pipeline (resize, normalisation, whole image or sliding window as configured) and the maps are scored with the protocol evaluator; a source the checkpoint was trained on is its internal test.",
              "stab:authors", "l l l c c c c c", ["Checkpoint", "Trained on", "Test set"] + D.CLASSES + ["\\mAUPR{}"], rows)


# ---------------------------------------------------------------- S17 inference tiers on every set
def s17_inference():
    t = df()
    r = t[(t.experiment == "E5") & (t.metric.isin(["aupr", "aupr_any_lesion"])) & (~t.test_set.astype(str).str.endswith("-val"))]
    rows = []
    for variant_prefix, title in (("zero_shot", "Zero-shot (any-lesion AUPR of the class-agnostic map; text prompts per class)"), ("oracle:box:all", "Oracle boxes, every component"), ("oracle:point:all", "Oracle points, one per component"), ("text:selected", "SAM 3, selected wording")):
        g = r[r.variant.astype(str).str.startswith(variant_prefix)]
        if g.empty:
            continue
        rows.append(f"\\multicolumn{{{2 + len(D.TESTS)}}}{{l}}{{\\textbf{{{title}}}}}\\\\")
        for (model, variant), h in g.groupby(["model", "variant"], sort=False):
            cells = []
            for test in D.TESTS:
                st = h[h.test_set == test]
                any_ = st[st.metric == "aupr_any_lesion"]
                cls = st[(st.metric == "aupr") & st.cls.isin(D.CLASSES)]
                cells.append(cell(cls) if len(cls) else (cell(any_) + "$^{\\rm a}$" if len(any_) else "\\na"))
            rows.append(" & ".join([esc(model), esc(variant)] + cells) + " \\\\")
    longtable("supp_s17_inference",
              "\\textbf{Foundation models without training, on every test set.} Zero-shot automatic masks (density and area cap chosen on the validation pool), SAM 3 text prompts with the wording chosen on the validation pool, and prompts derived from the ground truth (an upper bound that reads the test labels). Values are \\mAUPR{} where the output carries classes and the any-lesion AUPR$^{\\rm a}$ where it does not.",
              "stab:inference", "l l " + "c " * len(D.TESTS), ["Model", "Setting"] + [SET_NAMES.get(s, s) for s in D.TESTS], rows,
              notes="$^{\\rm a}$ any-lesion AUPR of a class-agnostic map. Variant tokens as in the run records (density, cap, canvas).", landscape=True)


# ---------------------------------------------------------------- Supplementary Note 1: models
def note_models():
    reg = D.registry()
    lines = []
    fam_title = {"general": "General segmentation networks", "auto": "Self-configuring", "fundus": "Fundus-specific", "foundation": "Foundation models", "fundus_multiscale": "Fundus-specific: multi-scale CNNs",
                 "relational_transformer": "Fundus-specific: relational and transformer models", "semi_supervised": "Fundus-specific: semi-supervised", "adapted_foundation": "Foundation models adapted to this task"}
    for family, g in reg.groupby("family", sort=False):
        lines.append(f"\\paragraph{{{esc(fam_title.get(family, family.replace('_', ' ').title()))}.}}")
        items = []
        for r in g.itertuples():
            bits = [f"\\textbf{{{esc(r.name)}}} (tier {r.tier}, reproducibility {esc(r.repro_level)}" + (f", track {esc(r.tracks)}" if r.tracks else "") + ")"]
            if r.repo:
                bits.append(f"code: \\texttt{{{esc(r.repo)}}}")
            if r.notes:
                bits.append(esc(r.notes))
            items.append("; ".join(bits) + ".")
        lines.append("\\begin{itemize}[leftmargin=*,itemsep=1pt]\n" + "\n".join(f"\\item {it}" for it in items) + "\n\\end{itemize}")
    GEN.mkdir(parents=True, exist_ok=True)
    (GEN / "supp_note_models.tex").write_text("\n".join(lines) + "\n")
    print("wrote generated/supp_note_models.tex")


# ---------------------------------------------------------------- S18 experiment names and repository identifiers
# The text names every experiment in words; the repository, the run directories, data/results.csv and the planning
# documents (docs/design/05, docs/paper/16) use short identifiers. One row per identifier (2026-10-26).
EXPERIMENT_NAMES = [
    # identifier, name used in the text, what it is, where it is reported
    ("L1", "Paper audit", "Extraction of the setup and the comparison tables of every method paper in the ledger", "Fig.~2b,c; Supplementary Tables~1 and 2"),
    ("L2", "Reproducibility funnel", "Registered fundus-specific methods by code, training code, weights and reproduced number", "Fig.~2a"),
    ("E1", "Released checkpoints (author weights)", "Released checkpoints scored with the paper's number, the authors' function and this protocol, on all test sets", "Fig.~3a,b,d; Table~3a; Supplementary Table~16"),
    ("E1r", "Author-recipe reruns", "The authors' own pipelines retrained from scratch on IDRiD (M2MRF, HRDecoder)", "Fig.~3c; Table~3b"),
    ("E2 (B-std)", "Unified retraining; the unified protocol", "Every tier-1 model with complete training code retrained on IDRiD and on DDR under the shared protocol", "Fig.~4; Tables~4 and 5"),
    ("E2b (B1)", "Source-recipe track", "Each model trained with its source recipe on the shared data; stopped before any run finished", "Methods only"),
    ("E2c", "Learning-rate sweep", "HRNet-W48 and M2MRF on DDR with the recipe learning rate $\\times$1/3 and $\\times$3, judged on validation only", "Extended Data Fig.~7"),
    ("E3a", "Independent tests", "Every main-table checkpoint applied to the six test sets of sources never used for training", "Fig.~4b--e; Table~5; Extended Data Fig.~9"),
    ("E3b", "Training-source experiment", "Five representative models trained on the standard-field TJDR images and on FGADR", "Fig.~6b--d; Supplementary Table~7"),
    ("E4", "Annotation-granularity experiment", "U-Net and M2MRF on fine, mixed and coarse training labels, alone and combined", "Fig.~6e; Table~7; Supplementary Table~7"),
    ("E5", "Foundation models under four kinds of use", "Zero-shot, oracle prompts, parameter-efficient fine-tuning, encoder + decoder; the encoder comparison", "Fig.~5; Table~6; Supplementary Tables~10 and 17"),
    ("E7", "Input-resolution experiment", "M2MRF and RETFound + UPerNet at a 512- or 1,440-pixel canvas and on native-resolution patches", "Fig.~7d; Table~7"),
    ("E8", "Training-amount experiment", "U-Net and M2MRF on 10\\%, 50\\% and 100\\% of DDR and on the merged fine-grained sources", "Fig.~7e; Table~7"),
    ("E9", "Loss experiment", "U-Net on DDR with four losses, architecture fixed", "Extended Data Fig.~8; Table~7"),
    ("E10", "Computational cost", "Parameters, trainable parameters and GPU hours of every run", "Extended Data Fig.~10; Supplementary Table~8"),
    ("ENS", "Ensembles", "Pixelwise means of the probability maps in the registered compositions", "Table~5; Supplementary Table~11"),
    ("REF", "Label revision (Refined IDRiD)", "IDRiD-trained checkpoints re-scored against the refined and the official labels", "Supplementary Table~9"),
    ("HUMAN", "Human reference", "Each DiaRetDB1 expert against the consensus of the other three", "Table~5; Supplementary Table~13"),
    ("CLIN", "Results by DR grade", "Every score recomputed within grades 0--1 and 2--4", "Supplementary Table~12"),
    ("N8", "Reference model", "Two models trained on the merged fine-grained sources and averaged; in training", "Methods"),
]


def s18_identifiers():
    rows = [" & ".join([f"\\texttt{{{esc(ident)}}}", name, what, where]) + " \\\\" for ident, name, what, where in EXPERIMENT_NAMES]
    longtable("supp_s18_identifiers",
              "\\textbf{Experiment names used in the text and the identifiers used in the repository.} The run directories under runs/, the experiment column of data/results.csv and the planning documents in docs/ carry the identifiers; the text names each experiment by what it does.",
              "stab:identifiers", "l p{40mm} p{66mm} p{38mm}",
              ["Identifier", "Name in the text", "What it is", "Where it is reported"], rows)


def main():
    for fn in (s1_literature, s2_reported, s3_runs, s4_perclass, s5_secondary, s6_registered, s7_factors, s8_cost, s9_refined,
               s10_encoders, s11_ensembles, s12_strata, s13_experts, s14_labels, s15_leaks, s16_authors, s17_inference, s18_identifiers, note_models):
        fn()


if __name__ == "__main__":
    main()
