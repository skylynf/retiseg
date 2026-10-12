#!/usr/bin/env python3
"""L1: merge the per-paper extraction sheets into the two ledger tables and derive
where every baseline number came from.

Inputs  (literature/meta/l1_raw/<group>_audit.tsv, <group>_results.tsv):
    hand-extracted from the PDFs, one sheet per extraction batch; column
    definitions in literature/meta/l1_raw/SCHEMA.md.
Outputs (literature/meta/):
    paper_audit.tsv       one row per method paper: split, resolution, metric
                          implementation, statement about baseline numbers.
    reported_results.tsv  one row per paper x compared method x dataset x lesion
                          x metric, with the printed value, the value on a 0-1
                          scale, and the derived provenance:
        provenance_derived  self        the paper's own method
                            rerun       the paper says it ran the baseline and the
                                        number matches no earlier paper
                            copied-1    identical to the number in the method's
                                        own paper (first-generation copy)
                            copied-2    identical to a number that another paper
                                        already copied or reran, and not to the
                                        method's own paper (second generation or
                                        later)
                            unverified  cited or unstated, and no matching number
                                        in the corpus (source outside the ledger,
                                        or the first appearance of a rerun by a
                                        third party)
        copy_source         the earliest paper in the corpus carrying the same
                            number (for copied-*)

Numbers are compared on the 0-1 scale after rounding to 3 decimals, which
tolerates the 4-vs-3 decimal printing seen across papers.
Run from the repository root:  python literature/l1_build.py
"""
import glob
import os
import re
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "literature", "meta", "l1_raw")
OUT = os.path.join(ROOT, "literature", "meta")

# Method aliases -> canonical name. Challenge teams keep their team name.
ALIAS = {
    r"^(VRT|PATech|iFLYTEK-MIG|IFLYTEK-MIG|iFLYTEK|SOONER|SAIHST|SHAIST|SDNU|MedLabs|CIL|AIMIA|lzyuncc_fusion).*": lambda m: {"IFLYTEK-MIG": "iFLYTEK-MIG", "iFLYTEK": "iFLYTEK-MIG", "SHAIST": "SAIHST"}.get(m.group(1), m.group(1)),
    r"^Yan2019.*|^Yan et al\..*": "Yan2019 (local-global U-Net)",
    r"^Xie2020.*|^SESV$": "SESV (Xie2020)",
    r"^Dual-branch \(Liu2022\)|^Dual-Branch U-Net \(Yin2023\)|^Dual PSPNet\+DSM \(Liu2022\)|^Dual-PSPNet": "Dual-branch (Liu2022)",
    r"^Deep membrane systems \(Xue2019\)|^DMS \(Xue2019\)": "DMS (Xue2019)",
    r"^HEDNet\+cGAN|^HEDNet$": "HEDNet+cGAN",
    r"^Sarhan et al\..*|^Sarhan2019": "Sarhan2019",
    r"^LKD \(Wang2024\)|^Lesion-aware KD \(Wang2024\)": "Lesion-aware KD (Wang2024)",
    r"^Swin-T$|^Swin-base$|^Swin-Tv2$|^Swin Transformer v2$": "Swin (UPerNet)",
    r"^DenseUNet$|^Dense U-Net \(Li2018\)$": "DenseUNet",
    r"^Zhu et al\. \(Zhu2023\)$": "Zhu2023",
    r"^HRNet\+OCR.*": "HRNet+OCR",
}


def canon(name):
    for pat, rep in ALIAS.items():
        m = re.match(pat, name)
        if m:
            return rep(m) if callable(rep) else rep
    return name


def norm_scale(df):
    """Values printed as percentages become fractions. Decided per (paper, table, metric)."""
    df = df.copy()
    df["value_num"] = pd.to_numeric(df["value"], errors="coerce")
    key = ["paper", "table", "metric"]
    mx = df.groupby(key)["value_num"].transform("max")
    pct = mx > 1.5
    df["value_norm"] = df["value_num"].where(~pct, df["value_num"] / 100.0)
    df["printed_as_percent"] = pct.map({True: "yes", False: "no"})
    return df


def derive_provenance(res, audit):
    """Match every baseline number to the same method/dataset/lesion/metric in other papers."""
    year = audit.set_index("paper")["year"].astype(int).to_dict()
    res = res.copy()
    res["year"] = res["paper"].map(year)
    res["key3"] = res["value_norm"].round(3)
    res["provenance_derived"] = ""
    res["copy_source"] = ""
    # own-paper values of each method: proposed rows
    own = res[res.is_proposed == "yes"]
    own_idx = {}
    for _, r in own.iterrows():
        own_idx.setdefault((r.method_norm, r.dataset, r.lesion, r.metric), set()).add(r.key3)
    # all values per key with (year, paper)
    by_key = {}
    for i, r in res.iterrows():
        by_key.setdefault((r.method_norm, r.dataset, r.lesion, r.metric, r.key3), []).append((r.year, r.paper, r.is_proposed, i))
    for i, r in res.iterrows():
        if r.is_proposed == "yes":
            res.at[i, "provenance_derived"] = "self"
            continue
        k = (r.method_norm, r.dataset, r.lesion, r.metric)
        if r.key3 in own_idx.get(k, set()):
            res.at[i, "provenance_derived"] = "copied-1"
            src = [p for (y, p, s, j) in by_key[k + (r.key3,)] if s == "yes"]
            res.at[i, "copy_source"] = src[0]
            continue
        others = [(y, p) for (y, p, s, j) in by_key[k + (r.key3,)] if p != r.paper and s != "yes"]
        earlier = sorted([(y, p) for (y, p) in others if y < r.year or (y == r.year and p < r.paper)])
        if earlier:
            res.at[i, "provenance_derived"] = "copied-2"
            res.at[i, "copy_source"] = earlier[0][1]
            continue
        if r.provenance_stated == "rerun":
            res.at[i, "provenance_derived"] = "rerun"
        else:
            res.at[i, "provenance_derived"] = "unverified"
    # A number that a later paper copied from a "rerun"/"unverified" row leaves the
    # earlier row's label unchanged: the earlier paper is the origin of that number.
    return res.drop(columns=["key3", "year"])


def main():
    audits = [pd.read_csv(f, sep="\t", dtype=str, keep_default_na=False) for f in sorted(glob.glob(os.path.join(RAW, "*_audit.tsv")))]
    results = [pd.read_csv(f, sep="\t", dtype=str, keep_default_na=False) for f in sorted(glob.glob(os.path.join(RAW, "*_results.tsv")))]
    if not audits:
        sys.exit(f"no extraction sheets under {RAW}")
    audit = pd.concat(audits, ignore_index=True)
    res = pd.concat(results, ignore_index=True)
    # drop rows without a usable number
    res = res[pd.to_numeric(res["value"], errors="coerce").notna()].copy()
    res["method_raw_norm"] = res["method_norm"]
    res["method_norm"] = res["method_norm"].map(canon)
    res = norm_scale(res)
    res = derive_provenance(res, audit)
    # de-duplicate rows that the same paper prints twice (abstract and table, two tables)
    res = res.drop_duplicates(subset=["paper", "dataset", "method_norm", "lesion", "metric", "value_norm"]).reset_index(drop=True)
    audit = audit.sort_values(["year", "paper"]).reset_index(drop=True)
    audit["baseline_rerun_class"] = audit["baseline_rerun_statement"].str.split("(").str[0].str.strip()
    res.to_csv(os.path.join(OUT, "reported_results.tsv"), sep="\t", index=False)
    audit.to_csv(os.path.join(OUT, "paper_audit.tsv"), sep="\t", index=False)
    print(f"papers {len(audit)}  rows {len(res)}")
    print(res["provenance_derived"].value_counts().to_string())
    main_tbl = res[(res.metric == "AUPR") & res.dataset.isin(["IDRiD", "DDR"])]
    print("AUPR rows on IDRiD/DDR by derived provenance:")
    print(main_tbl["provenance_derived"].value_counts().to_string())


if __name__ == "__main__":
    main()
