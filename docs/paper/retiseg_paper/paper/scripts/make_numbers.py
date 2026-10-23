"""Write generated/numbers.tex: every number and model name the running text quotes.

The text calls \\rn{key}. A value computed from placeholder rows is wrapped in
\\ph{}, so it prints in the placeholder colour until the rows are measured.
When the data change, rerun this script; sentences that name "the best model"
follow automatically, but their wording still has to be read again.

    python paper/scripts/make_numbers.py
"""

import sys
from pathlib import Path

import numpy as np
from scipy import stats as sps

PAPER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PAPER / "figures"))

from retifig import data as D  # noqa: E402

OUT = PAPER / "generated" / "numbers.tex"
N = {}


def put(key, text, sim):
    N[key] = (str(text), bool(sim))


def _half_up(v, places):
    """Round half up on the decimal value as written (0.6755 -> 0.676), not on the binary float."""
    from decimal import ROUND_HALF_UP, Decimal

    return str(Decimal(repr(round(float(v), 10))).quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP))


def f2(v):
    return _half_up(v, 2)


def f3(v):
    return _half_up(v, 3)


def facts():
    reg = D.registry()
    put("n.models", len(reg), False)
    for t in (1, 2, 3):
        put(f"n.tier{t}", int((reg.tier == t).sum()), False)
    put("n.sources", len(D.DATASETS), False)
    put("n.images", f"{sum(d['n'] for d in D.DATASETS):,}", False)
    put("n.independent", sum(d["role"] != "train" for d in D.DATASETS), False)
    dr = reg[reg.family.isin(["fundus_multiscale", "relational_transformer", "semi_supervised"]) | reg.name.isin(["TP-DRSeg", "SAM2-DR-Chen2025", "Ophthalmology-SAM"])]
    lvl = dr.repro_level.str[1].astype(int)
    put("funnel.dr", len(dr), False)
    put("funnel.code", int((lvl >= 1).sum()), False)
    put("funnel.train", int((lvl >= 3).sum()), False)
    put("funnel.full", int((lvl >= 4).sum()), False)
    put("n.main", len(D.MAIN_MODELS), False)
    audit = D.input_label_audit()["datasets"]
    for ds in ("IDRiD", "DDR"):
        for key, tag in (("swin_224", "224"), ("fct_384", "384"), ("long_side_1440", "1440")):
            for cls in ("MA", "EX"):
                put(f"vanish.{ds}.{tag}.{cls}", f"{100 * audit[ds]['train']['lesion_disappearance'][key][cls]['ratio']:.1f}", False)


def literature():
    """Paper audit L1 (literature/meta/paper_audit.tsv, reported_results.tsv). All measured."""
    import pandas as pd
    meta = D.REPO / "literature" / "meta"
    a = pd.read_csv(meta / "paper_audit.tsv", sep="\t", dtype=str, keep_default_na=False)
    r = pd.read_csv(meta / "reported_results.tsv", sep="\t", dtype=str, keep_default_na=False)
    r["v"] = r.value_norm.astype(float)
    put("lit.n", len(a), False)
    put("lit.n.methods", int((a.paper != "Playout2024_cross_dataset_generalization").sum()), False)
    for ds, col in (("idrid", "idrid_split"), ("ddr", "ddr_split")):
        used = a[a[col] != "NA"]
        put(f"lit.{ds}.n", len(used), False)
        put(f"lit.{ds}.official", int((used[col] == "official").sum()), False)
        put(f"lit.{ds}.own", int((used[col] == "own").sum()), False)
        put(f"lit.{ds}.notstated", int((used[col] == "not stated").sum()), False)
    aupr = a[a.aupr_impl != "NA"]
    put("lit.impl.n", len(aupr), False)
    put("lit.impl.notstated", int((aupr.aupr_impl == "not stated").sum()), False)
    put("lit.impl.fixed", int((aupr.aupr_impl == "11-threshold").sum()), False)
    put("lit.impl.exact", int((aupr.aupr_impl == "exact").sum()), False)
    put("lit.impl.perimage", int((aupr.aupr_impl == "per-image mean").sum()), False)
    put("lit.fov.notstated", int((a.fov_only.str.startswith("not stated")).sum()), False)
    put("lit.pooled.stated", int((a.per_image_or_pooled.str.startswith("pooled") | a.per_image_or_pooled.str.startswith("per-image")).sum()), False)
    put("lit.code.yes", int(a.code_stated.str.startswith("yes").sum()), False)
    cls = a.baseline_rerun_class
    put("lit.rerun.all", int((cls == "rerun all").sum()), False)
    put("lit.rerun.some", int((cls == "rerun some").sum()), False)
    put("lit.rerun.copied", int((cls == "copied").sum()), False)
    put("lit.rerun.notstated", int((cls == "not stated").sum()), False)
    base = r[(r.metric == "AUPR") & r.dataset.isin(["IDRiD", "DDR"]) & (r.is_proposed != "yes")]
    put("lit.base.n", f"{len(base):,}", False)
    for key, tag in (("rerun", "rerun"), ("copied-1", "copied1"), ("copied-2", "copied2"), ("unverified", "unverified")):
        put(f"lit.base.{tag}", f"{100 * (base.provenance_derived == key).mean():.0f}", False)
    rep = r[(r.metric == "AUPR") & (r.dataset == "IDRiD") & (r.lesion == "mean")].drop_duplicates(["paper", "method_norm"])
    for name, tag in (("L-Seg", "lseg"), ("DeepLabv3+", "deeplab"), ("U-Net", "unet"), ("HED", "hed"), ("M2MRF", "m2mrf")):
        s = rep[rep.method_norm == name]
        put(f"lit.{tag}.n", s.paper.nunique(), False)
        put(f"lit.{tag}.lo", f2(s.v.min()), False)
        put(f"lit.{tag}.hi", f2(s.v.max()), False)
    pub = D.sel(experiment="L1", variant="published_idrid", metric="reported_maupr_n").set_index("model")["value"]
    put("lit.published.multi", int((pub > 1).sum()), False)


def original():
    def g(model, variant, train, metric):
        r = D.sel(experiment="E1", model=model, variant=variant, train_set=train, test_set=train, metric=metric, cls="mean")
        return float(r["value"].iloc[0]), D.simulated(r)
    for train in D.TRAIN_MAIN:
        for variant, metric, tag in (("paper", "aupr_author_impl", "paper"), ("author_function", "aupr_author_impl", "fn"),
                                     ("this_protocol", "aupr_11pt", "stored"), ("this_protocol", "aupr", "exact")):
            v, s = g("M2MRF", variant, train, metric)
            put(f"e1.m2mrf.{train}.{tag}", f3(v), s)
        a, _ = g("M2MRF", "author_function", train, "aupr_author_impl")
        b, _ = g("M2MRF", "this_protocol", train, "aupr")
        put(f"e1.m2mrf.{train}.gap", f3(a - b), False)
    first = D.sel(experiment="E1r", model="M2MRF", variant="bs1_first_run", cls="mean")
    put("e1r.first.fn", f3(float(first[first.metric == "aupr_author_impl"]["value"].iloc[0])), False)
    put("e1r.first.exact", f3(float(first[first.metric == "aupr"]["value"].iloc[0])), False)
    for model in ("M2MRF", "HACDR-Net", "HRDecoder"):
        for train in D.TRAIN_MAIN:
            pr = D.sel(experiment="E1r", model=model, variant="paper", train_set=train, metric="aupr_author_impl")
            if not len(pr):
                pr = D.sel(experiment="E1", model=model, variant="paper", train_set=train, test_set=train, metric="aupr_author_impl", cls="mean")
            rr = D.sel(experiment="E1r", model=model, variant="author_recipe", train_set=train, metric="aupr_author_impl", cls="mean")
            if not len(pr) or not len(rr):
                continue
            sim_r = D.simulated(rr) or D.simulated(pr)
            r = rr if sim_r else rr[rr.source == "measured"]
            p = float(pr["value"].iloc[0])
            put(f"e1r.{model}.{train}.delta", f"{r['value'].mean() - p:+.3f}".replace("-", "−"), sim_r)
            put(f"e1r.{model}.{train}.fn", f3(r["value"].mean()), D.simulated(rr))
            xx = D.sel(experiment="E1r", model=model, variant="author_recipe", train_set=train, metric="aupr", cls="mean")
            xx = xx if D.simulated(xx) else xx[xx.source == "measured"]
            put(f"e1r.{model}.{train}.exact", f3(xx["value"].mean()), D.simulated(xx))


def retrained():
    ms = D.mean_sd()
    sim = D.simulated(D.main_runs())
    internal = D.internal()
    for train in D.TRAIN_MAIN:
        sub = ms[(ms.train_set == train) & (ms.test_set == train)].set_index("model").sort_values("mean", ascending=False)
        put(f"bstd.{train}.best.model", sub.index[0], sim)
        put(f"bstd.{train}.best.value", f2(sub["mean"].iloc[0]), sim)
        put(f"bstd.{train}.second.model", sub.index[1], sim)
        put(f"bstd.{train}.second.value", f2(sub["mean"].iloc[1]), sim)
        put(f"bstd.{train}.worst.model", sub.index[-1], sim)
        put(f"bstd.{train}.worst.value", f2(sub["mean"].iloc[-1]), sim)
        put(f"bstd.{train}.sd.median", f3(sub["sd"].median()), sim)
        put(f"bstd.{train}.sd.max", f3(sub["sd"].max()), sim)
        gap = sub["mean"].iloc[0] - sub["mean"].iloc[1]
        put(f"bstd.{train}.gap12", f3(gap), sim)
        tab = internal.xs(train, level="train_set")
        for cls in D.CLASSES:
            put(f"bstd.{train}.{cls}.best", f2(tab[cls].max()), sim)
            put(f"bstd.{train}.{cls}.median", f2(tab[cls].median()), sim)
        put(f"bstd.{train}.unet.SE", f2(tab.loc["U-Net", "SE"]), sim)
        # paired tests and correlations over the measured models only (a placeholder model would enter the pairs)
        measured_models = D.measured_models()
        m = D.score_matrix(train, models=measured_models)
        part = D.simulated(D.main_runs()[D.main_runs().model.isin(measured_models)])
        for test in ("TJDR-std", "MAPLES-DR", "TJDR-uwf", "FGADR", "Retinal-Lesions"):
            d = m[test] - m[train]
            put(f"indep.{train}.{test}.drop", f2(-d.mean()), part)
            put(f"indep.{train}.{test}.gain", f2(d.mean()), part)
            put(f"indep.{train}.{test}.nlower", int((d < 0).sum()), part)   # models whose independent score is below their internal one
            put(f"indep.{train}.{test}.p", "< 0.001" if sps.wilcoxon(m[train], m[test]).pvalue < 0.001 else f"= {sps.wilcoxon(m[train], m[test]).pvalue:.3f}", part)
        put(f"indep.{train}.ndrop", int((m[D.FINE_INDEPENDENT].mean(axis=1) < m[train]).sum()), part)
        rho = sps.spearmanr(m[train], m[D.FINE_INDEPENDENT].mean(axis=1)).statistic
        put(f"indep.{train}.rho", f2(rho), part)
        top_int = m[train].idxmax()
        top_ind = m[D.FINE_INDEPENDENT].mean(axis=1).idxmax()
        put(f"indep.{train}.top.model", top_ind, part)
        put(f"indep.{train}.top.value", f2(m.loc[top_ind, D.FINE_INDEPENDENT].mean()), part)
        put(f"indep.{train}.sametop", "the same model" if top_int == top_ind else "a different model", part)
    blocks = D.block_scores()          # measured models only (data.measured_models)
    fr = D.friedman(blocks)
    rank = np.array(fr["avg_rank"])
    order = np.argsort(rank)
    put("n.ranked", len(blocks), False)
    part = D.simulated(D.main_runs()[D.main_runs().model.isin(list(blocks.index))])
    put("cd.chi2", f"{fr['statistic']:.1f}", part)
    put("cd.p", "< 0.001" if fr["p"] < 0.001 else f"= {fr['p']:.3f}", part)
    put("cd.value", f2(fr["critical_difference"]), part)
    put("cd.first.model", blocks.index[order[0]], part)
    put("cd.first.rank", f2(rank[order[0]]), part)
    put("cd.second.model", blocks.index[order[1]], part)
    put("cd.second.rank", f2(rank[order[1]]), part)
    put("cd.third.model", blocks.index[order[2]], part)
    put("cd.third.rank", f2(rank[order[2]]), part)
    put("cd.last.model", blocks.index[order[-1]], part)
    put("cd.last.rank", f2(rank[order[-1]]), part)
    put("cd.ntop", int((rank - rank.min() <= fr["critical_difference"]).sum()), part)
    fam = ms[ms.model.isin(list(blocks.index))].assign(family=ms["model"].map(D.FAMILY_OF))
    ind = fam[fam.test_set.isin(D.FINE_INDEPENDENT)].groupby("family")["mean"].mean()
    inn = fam[fam.train_set == fam.test_set].groupby("family")["mean"].mean()
    for f in ("general", "auto", "fundus", "foundation"):
        put(f"family.{f}.internal", f2(inn[f]), part)
        put(f"family.{f}.independent", f2(ind[f]), part)
    pub_rows = D.sel(experiment="L1", variant="published_idrid", metric="reported_maupr")
    pub_rows = pub_rows[pub_rows.model.isin(measured_models)]      # a published number is paired with a measured retraining only
    pub = pub_rows.set_index("model")["value"]
    ret = D.score_matrix("IDRiD", models=list(pub.index))["IDRiD"]
    tau = D.bstats.kendall_tau(pub.to_numpy(), ret.to_numpy())
    pub_sim = D.simulated(pub_rows) or len(pub) < 3
    put("tau.published", f2(tau["tau"]), pub_sim)
    put("tau.published.p", f2(tau["p"]), pub_sim)
    put("tau.published.n", len(pub), pub_sim)
    exact = D.score_matrix("IDRiD")["IDRiD"]
    eleven = D.score_matrix("IDRiD", metric="aupr_11pt")["IDRiD"]
    put("impl.mean", f3((eleven - exact).mean()), sim)
    put("impl.max", f3((eleven - exact).max()), sim)
    put("impl.moved", int((sps.rankdata(-exact) != sps.rankdata(-eleven)).sum()), sim)
    put("impl.tau", f2(D.bstats.kendall_tau(exact.to_numpy(), eleven.to_numpy())["tau"]), sim)
    b1 = D.sel(experiment="E2b", metric="aupr").groupby(["model", "train_set"])["value"].mean().unstack("train_set")
    for train in D.TRAIN_MAIN:
        d = b1[train] - D.score_matrix(train, models=list(b1.index))[train]
        put(f"b1.{train}.worst.model", d.idxmin(), True)
        put(f"b1.{train}.worst.delta", f2(-d.min()), True)
        put(f"b1.{train}.nlower", int((d < 0).sum()), True)
    put("b1.n", len(b1), True)
    for metric, tag in (("lesion_recall_small", "small"), ("lesion_recall_large", "large")):
        v = D.score_matrix("DDR", metric=metric)["DDR"]
        put(f"recall.DDR.{tag}.median", f2(v.median()), sim)
        put(f"recall.DDR.{tag}.best", f2(v.max()), sim)


def backbone_controlled():
    """HRNet-W48, M2MRF and HRDecoder share the HRNet-W48 backbone (docs/paper/18 section 1.2)."""
    rows = D.sel(experiment="E2", variant="bstd", metric="aupr", model=D.SAME_BACKBONE)
    if not len(rows):
        return
    sim = D.simulated(rows)
    best = {}
    for train in D.TRAIN_MAIN:
        m = rows[(rows.train_set == train) & (rows.test_set == train)].groupby("model")["value"].mean()
        if len(m) == len(D.SAME_BACKBONE):
            best[train] = (m.idxmax(), float(m.max() - m.min()))
    if "IDRiD" in best:
        put("backbone.best", best["IDRiD"][0], sim)
    if "DDR" in best:
        put("backbone.best.ddr", best["DDR"][0], sim)
    if best:
        put("backbone.span", "--".join(f2(v[1]) for v in sorted(best.values(), key=lambda x: x[1])) if len(best) == 2 else f2(list(best.values())[0][1]), sim)


def foundation():
    def e5(model, variant, test, train=None):
        kw = dict(experiment="E5", model=model, variant=variant, test_set=test, metric="aupr")
        if train:
            kw["train_set"] = train
        return D.sel(**kw)["value"].mean()
    # the zero-shot block is summarised by the any-lesion AUPR (decided 2026-10-14): SAM / SAM 2 class-agnostic maps,
    # SAM 3 with the wording chosen on the validation pool (text:selected); nothing is picked on the test set
    def any_of(model, variant):
        r = D.sel(experiment="E5", model=model, variant=variant, test_set="IDRiD", metric="aupr_any_lesion", cls="any")
        return (float(r["value"].mean()), D.simulated(r)) if len(r) else (float("nan"), True)
    zs = {"SAM": any_of("SAM", "zero_shot:auto_any_lesion"), "SAM 2": any_of("SAM 2", "zero_shot:auto_any_lesion"), "SAM 3": any_of("SAM 3", "text:selected")}
    best = max(zs, key=lambda m: zs[m][0])
    put("fm.zero.model", best, any(v[1] for v in zs.values()))
    put("fm.zero.value", f2(zs[best][0]), any(v[1] for v in zs.values()))
    # the upper bound of the automatic masks when each mask takes the class it overlaps most (reads the test labels)
    ub = {"SAM": e5("SAM", "zero_shot:auto_oracle_class", "IDRiD"), "SAM 2": e5("SAM 2", "zero_shot:auto_oracle_class", "IDRiD")}
    put("fm.zero.ub.model", max(ub, key=ub.get), True)
    put("fm.zero.ub.value", f2(max(ub.values())), True)
    ref_any = D.reference_any_lesion("nnU-Net", "IDRiD")
    put("fm.any.ref.nnunet.IDRiD", "--" if ref_any is None else f2(ref_any), ref_any is None)
    # the class-agnostic main number of tier a: the protocol's merged any-lesion AUPR (cls "any"), measured only
    for model, tag in (("SAM", "sam"), ("SAM 2", "samtwo")):
        for test in ("IDRiD", "DDR"):
            r = D.sel(experiment="E5", model=model, variant="zero_shot:auto_any_lesion", test_set=test, metric="aupr_any_lesion", cls="any")
            if len(r):
                put(f"fm.any.{tag}.{test}", f2(r["value"].iloc[0]), D.simulated(r))
    put("fm.oracle.all", f2(e5("SAM 2", "oracle:box:all", "IDRiD")), True)
    put("fm.oracle.five", f2(e5("SAM 2", "oracle:box:5", "IDRiD")), True)
    put("fm.oracle.twenty", f2(e5("SAM 2", "oracle:box:20", "IDRiD")), True)
    w = D.sel(experiment="E5", model="SAM 3", test_set="IDRiD", metric="aupr").groupby("cls")["value"]
    put("fm.wording.ratio", f"{(w.max() / w.min()).mean():.1f}", True)
    m = D.score_matrix("DDR")
    peft = m.loc[["SAMed", "SAM-Adapter", "SAM2-UNet"], "DDR"]
    put("fm.peft.model", peft.idxmax(), True)
    put("fm.peft.value", f2(peft.max()), True)
    put("fm.ref.nnunet", f2(m.loc["nnU-Net", "DDR"]), True)
    # encoder comparison (16 section 3 E5d2; C1 = RETFound-MAE - MAE-IN1k, C2 = RETFound-DINOv2 - DINOv2): placeholder
    # flags per encoder; a difference is flagged when either side is
    enc = D.encoder_table("DDR")
    key = {"RETFound-MAE": "rm", "MAE-IN1k ViT-L": "mi", "RETFound-DINOv2": "rd", "DINOv2 ViT-L": "di"}
    for model, short in key.items():
        put(f"fm.enc.{short}.int", f2(enc["internal"][model]), enc["simulated"][model])
        put(f"fm.enc.{short}.ind", f2(enc["independent"][model]), enc["simulated"][model])
        for t in ("TJDR-std", "TJDR-uwf"):
            put(f"fm.enc.{short}.{t}", f2(enc["table"].loc[model, t]), not enc["measured"].loc[model, t])
    for pair, (a, b) in {"c1": ("RETFound-MAE", "MAE-IN1k ViT-L"), "c2": ("RETFound-DINOv2", "DINOv2 ViT-L")}.items():
        sim = enc["simulated"][a] or enc["simulated"][b]
        put(f"fm.enc.{pair}.int", f3(enc["internal"][a] - enc["internal"][b]), sim)
        put(f"fm.enc.{pair}.ind", f3(enc["independent"][a] - enc["independent"][b]), sim)
        put(f"fm.enc.{pair}.int.abs", f3(abs(enc["internal"][a] - enc["internal"][b])), sim)
        put(f"fm.enc.{pair}.std.abs", f3(abs(enc["table"].loc[a, "TJDR-std"] - enc["table"].loc[b, "TJDR-std"])), sim)
    names = {"TJDR-std": "the standard-field TJDR set", "TJDR-uwf": "the ultra-widefield TJDR set", "MAPLES-DR": "MAPLES-DR"}
    put("fm.enc.sets", " and ".join(names.get(t, t) for t in enc["sets"]) or "pending", not enc["sets"])
    # the old keys keep working for the simtext of R5 until it is rewritten
    put("fm.enc.retfound.int", f2(enc["internal"]["RETFound-DINOv2"]), enc["simulated"]["RETFound-DINOv2"])
    put("fm.enc.imagenet.int", f2(enc["internal"]["MAE-IN1k ViT-L"]), enc["simulated"]["MAE-IN1k ViT-L"])
    put("fm.enc.retfound.ind", f2(enc["independent"]["RETFound-DINOv2"]), enc["simulated"]["RETFound-DINOv2"])
    put("fm.enc.dinov2.ind", f2(enc["independent"]["DINOv2 ViT-L"]), enc["simulated"]["DINOv2 ViT-L"])
    put("fm.enc.imagenet.ind", f2(enc["independent"]["MAE-IN1k ViT-L"]), enc["simulated"]["MAE-IN1k ViT-L"])


def refined():
    """Refined IDRiD (26 N4; 14 note 54): every IDRiD-trained main-table checkpoint's maps through the release geometry,
    scored against the refined labels and against the official labels through the same geometry (experiment REF,
    bench/scripts/refined_idrid.py). Keys: the change refined - official (class mean, per class), its rank agreement,
    and the distance of the 1024 re-scoring from the main-table test at 1440."""
    r = D.sel(experiment="REF", metric="aupr", variant=["refined_labels", "original_labels_1024"])
    if not len(r):
        return
    sim = D.simulated(r)
    m = r.groupby(["model", "variant"])["value"].mean().unstack()
    delta = m["refined_labels"] - m["original_labels_1024"]
    put("ref.n", str(len(delta)), sim)
    put("ref.delta.mean", f3(delta.mean()), sim)
    put("ref.delta.min", f3(delta.min()), sim)
    put("ref.delta.max", f3(delta.max()), sim)
    put("ref.tau", f2(sps.kendalltau(m["refined_labels"], m["original_labels_1024"]).statistic), sim)
    pc = r.pivot_table(index="model", columns=["cls", "variant"], values="value")
    for cls in D.CLASSES:
        d = pc[(cls, "refined_labels")] - pc[(cls, "original_labels_1024")]
        put(f"ref.{cls}.delta.mean", f3(d.mean()), sim)
        put(f"ref.{cls}.delta.min", f3(d.min()), sim)
    main_rows = D.sel(experiment=["E2", "E5"], variant=D.MAIN_VARIANTS, metric="aupr", train_set="IDRiD", test_set="IDRiD", seed=0)
    main_rows = main_rows[main_rows.model.isin(delta.index)]
    bstd = main_rows.groupby("model")["value"].mean()
    both = m.index.intersection(bstd.index)
    d1024 = m.loc[both, "original_labels_1024"] - bstd[both]
    put("ref.geometry.absmax", f3(d1024.abs().max()), sim or D.simulated(main_rows))
    put("ref.geometry.tau", f2(sps.kendalltau(m.loc[both, "original_labels_1024"], bstd[both]).statistic), sim or D.simulated(main_rows))


def _measured(df):
    """(measured rows, whether any placeholder row was left out): factor means are taken over measured cells only,
    and the key is marked as placeholder while a cell of its selection is still missing."""
    keep = df[df["source"] == "measured"]
    return keep, len(keep) < len(df)


def factors():
    import pandas as pd

    a = D.sel(experiment=["E2", "E5"], variant=D.MAIN_VARIANTS, metric="aupr", model=D.REPRESENTATIVE, seed=0)
    b = D.sel(experiment="E3b", metric="aupr", model=D.REPRESENTATIVE)
    trains = ["IDRiD", "DDR", "TJDR-std", "FGADR"]
    square, square_sim = _measured(pd.concat([a, b]).query("test_set in @trains"))
    g = square.groupby(["model", "train_set", "test_set"])["value"].mean().groupby(["train_set", "test_set"]).mean().unstack("test_set")
    A = g.reindex(index=trains, columns=trains).to_numpy()
    off = A - np.diag(A)[None, :]
    put("cross.mean", f2(-np.nanmean(off[~np.eye(4, dtype=bool)])), square_sim)
    put("cross.max", f2(-np.nanmin(off)), square_sim)
    put("cross.fgadr.fine", f2(-np.nanmean(off[3, :3])), square_sim)
    e4_rows, e4_sim = _measured(D.sel(experiment="E4", metric="aupr", test_set="IDRiD"))
    e4 = e4_rows.groupby("variant")["value"].mean()
    for cfg, tag in (("DDR + FGADR", "fgadr"), ("DDR + Retinal-Lesions", "retles"), ("DDR + FGADR (383)", "fgadr383"), ("Retinal-Lesions", "retlesonly"), ("FGADR", "fgadronly")):
        put(f"e4.{tag}", f"{e4[cfg] - e4['DDR']:+.2f}".replace("-", "−") if cfg in e4 and "DDR" in e4 else "n/a", e4_sim or cfg not in e4)
    put("e4.ref", f2(e4["DDR"]) if "DDR" in e4 else "n/a", e4_sim)
    e7_rows, e7_sim = _measured(D.sel(experiment="E7", test_set="DDR"))
    m = e7_rows[e7_rows.metric == "aupr"].groupby("variant")["value"].mean()
    s = e7_rows[e7_rows.metric == "lesion_recall_small"].groupby("variant")["value"].mean()
    for key, series, variant in (("e7.aupr.512", m, "res:512"), ("e7.aupr.1440", m, "res:1440"), ("e7.aupr.native", m, "res:native patches"),
                                 ("e7.small.512", s, "res:512"), ("e7.small.1440", s, "res:1440")):
        put(key, f2(series[variant]) if variant in series else "n/a", e7_sim or variant not in series)
    e8_rows, e8_sim = _measured(D.sel(experiment="E8", metric="aupr"))
    e8 = e8_rows.groupby(["test_set", "variant"])["value"].mean()
    for key, index in (("e8.ddr.10", ("DDR", "size:DDR 10%")), ("e8.ddr.50", ("DDR", "size:DDR 50%")), ("e8.ddr.100", ("DDR", "size:DDR 100%")),
                       ("e8.ddr.all", ("DDR", "size:fine merged")), ("e8.tjdr.100", ("TJDR-std", "size:DDR 100%")), ("e8.tjdr.fine", ("TJDR-std", "size:fine merged"))):
        put(key, f2(e8[index]) if index in e8.index else "n/a", e8_sim or index not in e8.index)
    e9_rows, e9_sim = _measured(D.sel(experiment="E9", metric="aupr"))
    e9 = e9_rows.groupby("variant")["value"].mean()
    e9 = e9[e9.index != "loss:CE + per-image Dice"]  # the collapsed registered recipe is reported separately
    put("e9.range", f2(e9.max() - e9.min()), e9_sim)
    cost, cost_sim = _measured(D.sel(experiment="E10", metric="train_gpu_h", train_set="IDRiD"))
    c = cost.set_index("model")["value"]
    put("cost.min.model", c.idxmin(), cost_sim)
    put("cost.min", f"{c.min():.1f}", cost_sim)
    put("cost.max.model", c.idxmax(), cost_sim)
    put("cost.max", f"{c.max():.0f}", cost_sim)
    cost_ddr, _ = _measured(D.sel(experiment="E10", metric="train_gpu_h", train_set="DDR"))
    cd_ = cost_ddr.set_index("model")["value"]
    put("cost.ddr.max", f"{cd_.max():.0f}", cost_sim)
    put("cost.ddr.max.model", cd_.idxmax(), cost_sim)
    both = c.index.intersection(cd_.index)
    put("cost.ddr.ratio", f"{(cd_[both] / c[both]).median():.1f}", cost_sim)


def main():
    facts()
    literature()
    original()
    retrained()
    foundation()
    backbone_controlled()
    factors()
    refined()
    lines = ["% Generated by scripts/make_numbers.py. Do not edit by hand.", "\\makeatletter"]
    for key, (text, sim) in sorted(N.items()):
        body = text.replace("%", "\\%").replace("−", "$-$").replace("<", "$<$")
        body = f"\\ph{{{body}}}" if sim else body
        lines.append(f"\\@namedef{{rn@{key}}}{{{body}}}")
    lines += ["\\makeatother", ""]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines))
    n_sim = sum(s for _, s in N.values())
    print(f"wrote {OUT.relative_to(PAPER)}: {len(N)} keys, {n_sim} from placeholder rows")


if __name__ == "__main__":
    main()
