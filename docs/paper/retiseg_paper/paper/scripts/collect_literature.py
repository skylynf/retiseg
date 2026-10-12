"""Replace the placeholder L1 rows of data/results.csv with the paper audit.

Reads literature/meta/paper_audit.tsv and literature/meta/reported_results.tsv
(both written by literature/l1_build.py from the per-paper extraction sheets)
and writes, with source = measured:

    L1  papers   split:<official|own split|not stated>   train_set = IDRiD|DDR, seed = year, n_papers
    L1  papers   impl:<exact AP|11-threshold|per-image mean|not stated>       seed = year, n_papers
    L1  papers   provenance:<re-run|copied once|copied again|unverifiable>    seed = year, n_baseline_numbers
    L1  <method> paper:<paper>      test_set = IDRiD, cls = mean, reported_maupr   (one row per citing paper)
    L1  <model>  published_idrid    test_set = IDRiD, cls = mean, reported_maupr   (one row per main-table model)

Every existing L1 row is dropped first, so the per-year counts, the set of
methods in Fig. 2c and the published values in Fig. 4f all come from the
ledger. Nothing else in the table is touched.

Definitions
    impl        the paper's own statement or its released code; a fixed grid of
                thresholds (11 or 33) counts as 11-threshold; papers whose main
                metric is not AUPR are left out of this row.
    provenance  baseline AUPR values on IDRiD and DDR, excluding each paper's own
                method, by provenance_derived of l1_build.py: rerun -> re-run,
                copied-1 -> copied once, copied-2 -> copied again, unverified ->
                unverifiable (cited to a source outside the ledger, or no source).
    reported    mAUPR on the IDRiD test set printed for a method, one value per
                citing paper (the first printed value when a paper prints it
                twice); methods shown are those reported by at least three papers,
                challenge teams excluded.
    published   for a fundus-specific model, the value in its own paper (best
                variant); for a general or foundation model, the median of the
                values reported for it in the ledger's DR papers, with the number
                of papers recorded as published_idrid_n.

    python paper/scripts/collect_literature.py            # write
    python paper/scripts/collect_literature.py --dry-run  # only report
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

PAPER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PAPER / "figures"))
from retifig.data import REPO  # noqa: E402

CSV = PAPER / "data" / "results.csv"
META = REPO / "literature" / "meta"

IMPL = {"exact": "exact AP", "11-threshold": "11-threshold", "per-image mean": "per-image mean", "not stated": "not stated"}
PROV = {"rerun": "re-run", "copied-1": "copied once", "copied-2": "copied again", "unverified": "unverifiable"}
TEAMS = {"VRT", "PATech", "iFLYTEK-MIG", "SOONER", "SAIHST", "SDNU", "MedLabs", "CIL", "AIMIA", "lzyuncc_fusion"}
# ledger method name -> name used in the main tables; fundus-specific models take their own paper's value
PUBLISHED = {
    "M2MRF": ("M2MRF", "own"),
    "HACDR-Net": ("HACDR-Net", "own"),
    "U-Net": ("U-Net", "median"),
    "DeepLabv3+": ("DeepLabv3", "median"),
    "HRNet": ("HRNet-W48", "median"),
    "Swin-Unet": ("Swin-Unet", "median"),
    "FCT": ("FCT", "median"),
    "SAMed": ("SAMed", "median"),
    "SAM-Adapter": ("SAM-Adapter", "median"),
    "MedSAM": ("MedSAM (full FT)", "median"),
}


def row(model, variant, train, seed, test, cls, metric, value):
    return dict(experiment="L1", model=model, variant=variant, train_set=train, seed=int(seed), test_set=test, cls=cls, metric=metric, value=float(value), source="measured")


def build_rows(audit, res):
    rows = []
    audit = audit.assign(year=audit.year.astype(int))
    for year, grp in audit.groupby("year"):
        for ds, col in (("IDRiD", "idrid_split"), ("DDR", "ddr_split")):
            for name, label in (("official", "official"), ("own", "own split"), ("not stated", "not stated")):
                rows.append(row("papers", f"split:{label}", ds, year, "", "", "n_papers", int((grp[col] == name).sum())))
        for name, label in IMPL.items():
            rows.append(row("papers", f"impl:{label}", "", year, "", "", "n_papers", int((grp.aupr_impl == name).sum())))
    res = res.assign(value_norm=res.value_norm.astype(float))
    base = res[(res.metric == "AUPR") & res.dataset.isin(["IDRiD", "DDR"]) & (res.is_proposed != "yes")]
    base = base.merge(audit[["paper", "year"]], on="paper")
    for year, grp in base.groupby("year"):
        for name, label in PROV.items():
            rows.append(row("papers", f"provenance:{label}", "", year, "", "", "n_baseline_numbers", int((grp.provenance_derived == name).sum())))
    rep = res[(res.metric == "AUPR") & (res.dataset == "IDRiD") & (res.lesion == "mean") & ~res.method_norm.isin(TEAMS)]
    rep = rep.drop_duplicates(subset=["paper", "method_norm"])
    counts = rep.groupby("method_norm").paper.nunique()
    for method in counts[counts >= 3].index:
        for _, r in rep[rep.method_norm == method].iterrows():
            rows.append(row(method, f"paper:{r.paper}", "IDRiD", -1, "IDRiD", "mean", "reported_maupr", r.value_norm))
    for ledger_name, (model, how) in PUBLISHED.items():
        sub = rep[rep.method_norm == ledger_name]
        if how == "own":
            sub = sub[sub.is_proposed == "yes"]
            own = res[(res.method_norm == ledger_name) & (res.is_proposed == "yes") & (res.metric == "AUPR") & (res.dataset == "IDRiD") & (res.lesion == "mean")]
            if own.empty:
                continue
            value, n = own.value_norm.max(), 1
        else:
            if sub.empty:
                continue
            value, n = sub.value_norm.median(), sub.paper.nunique()
        rows.append(row(model, "published_idrid", "IDRiD", -1, "IDRiD", "mean", "reported_maupr", value))
        rows.append(row(model, "published_idrid", "IDRiD", -1, "IDRiD", "mean", "reported_maupr_n", n))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    audit = pd.read_csv(META / "paper_audit.tsv", sep="\t", dtype=str, keep_default_na=False)
    res = pd.read_csv(META / "reported_results.tsv", sep="\t", dtype=str, keep_default_na=False)
    new = build_rows(audit, res)
    df = pd.read_csv(CSV)
    old = df[df.experiment == "L1"]
    print(f"ledger: {len(audit)} papers, {len(res)} numbers; L1 rows: {len(old)} placeholder -> {len(new)} measured")
    print(new[new.variant.str.startswith("provenance:")].groupby("variant").value.sum().astype(int).to_string())
    print(new[new.variant.str.startswith("impl:")].groupby("variant").value.sum().astype(int).to_string())
    print(new[(new.variant == "published_idrid") & (new.metric == "reported_maupr")][["model", "value"]].to_string(index=False))
    if args.dry_run:
        return
    out = pd.concat([df[df.experiment != "L1"], new], ignore_index=True)
    out.to_csv(CSV, index=False)
    print(f"wrote {CSV}")


if __name__ == "__main__":
    main()
