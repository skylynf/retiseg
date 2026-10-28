"""Join the readers' exported answers to the key and compute the pilot / main-study summaries.

docs/clinical/28 section 7. Inputs: selection.json (the key) and one exported JSON per reader. Outputs in --out:

    regions_long.csv   one row per reader x region: hidden metadata (set, class, kind, arm, weight, size, model
                       probability, model consensus, grade) + the reader's answers + derived fields + active time
    images_long.csv    one row per reader x image: grade, DME, quality, post-review judgements, active time
    report.md          completion and time; the blinded verdict and the stated cause by class and kind;
                       design-weighted shares of the representative arm with 95% intervals (bootstrap over
                       images, 2000 replicates, seed 0); agreement between readers per field (percent agreement,
                       Cohen's kappa per pair, Fleiss' kappa, Gwet's AC1; quadratic-weighted kappa for ordinal
                       fields); clinical impact; issue and remedy counts; free-text comments; the pilot checks

The blinded verdict comes from step A only (what the reader saw before any outline was shown):
    the reader's finding is mapped to a lesion class (ma -> MA, he -> HE, ex -> EX, se -> SE), to "other DR lesion"
    (irma, nv, prh), "other lesion", "no lesion" (normal structures, artifacts) or "indeterminate";
    then for a region of class c
      FP (model only)     finding c -> model_right; no lesion -> dataset_right; another class -> other_class
      FN (dataset only)   finding c -> dataset_right; no lesion -> model_right; another class -> other_class
      agree / extent      finding c -> both_right;  no lesion -> both_wrong;   another class -> other_class
      indeterminate -> unresolved
Practice cases are excluded.

    python clinical/scripts/analyze_reviews.py --selection clinical/work/pilot01/selection.json \
        --reviews clinical/reviews/pilot01/*.json --out clinical/work/pilot01/analysis
"""

import argparse
import csv
import itertools
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

FINDING_CLASS = {"ma": "MA", "he": "HE", "ex": "EX", "se": "SE"}
FINDING_GROUP = {
    "ma": "MA", "he": "HE", "ex": "EX", "se": "SE",
    "irma": "other_dr", "nv": "other_dr", "prh": "other_dr",
    "drusen": "other_lesion", "laser": "other_lesion", "pigment": "other_lesion", "other_lesion": "other_lesion",
    "vessel": "no_lesion", "disc": "no_lesion", "reflex": "no_lesion", "background": "no_lesion",
    "artifact_dust": "no_lesion", "artifact_light": "no_lesion", "artifact_edge": "no_lesion",
    "indeterminate": "indeterminate",
}
ORDINAL = {"b_impact": ["0", "1", "2", "3"], "pre_icdr": ["0", "1", "2", "3", "4"], "b_difficulty": ["easy", "moderate", "hard"]}


def blind_verdict(kind, cls, finding):
    if finding is None:
        return None
    group = FINDING_GROUP.get(finding, "other_lesion")
    if group == "indeterminate":
        return "unresolved"
    if group == cls:
        return {"FP": "model_right", "FN": "dataset_right"}.get(kind, "both_right")
    if group == "no_lesion":
        return {"FP": "dataset_right", "FN": "model_right"}.get(kind, "both_wrong")
    return "other_class"


# ------------------------------------------------------------------------------------------------ agreement


def cohen_kappa(a, b, categories=None, weights=None):
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if len(pairs) < 2:
        return None
    cats = categories or sorted({x for p in pairs for x in p})
    idx = {c: i for i, c in enumerate(cats)}
    k = len(cats)
    if k < 2:
        return None
    m = np.zeros((k, k))
    for x, y in pairs:
        m[idx[x], idx[y]] += 1
    m /= m.sum()
    if weights == "quadratic":
        w = np.array([[(i - j) ** 2 for j in range(k)] for i in range(k)], dtype=float) / (k - 1) ** 2
    else:
        w = 1.0 - np.eye(k)
    expected = np.outer(m.sum(1), m.sum(0))
    denom = (w * expected).sum()
    return None if denom == 0 else float(1 - (w * m).sum() / denom)


def fleiss_kappa(table):
    """table: items x categories counts, each item rated by the same number of raters."""
    t = np.asarray(table, dtype=float)
    if t.shape[0] < 2:
        return None
    n = t.sum(1)
    if not np.all(n == n[0]) or n[0] < 2:
        return None
    n = n[0]
    p = t.sum(0) / t.sum()
    P = ((t * t).sum(1) - n) / (n * (n - 1))
    Pe = (p * p).sum()
    return None if Pe == 1 else float((P.mean() - Pe) / (1 - Pe))


def gwet_ac1(table):
    t = np.asarray(table, dtype=float)
    if t.shape[0] < 2 or t.shape[1] < 2:
        return None
    n = t.sum(1)
    keep = n >= 2
    t, n = t[keep], n[keep]
    if t.shape[0] < 2:
        return None
    pa = (((t * (t - 1)).sum(1)) / (n * (n - 1))).mean()
    pi = (t / n[:, None]).mean(0)
    pe = (pi * (1 - pi)).sum() / (t.shape[1] - 1)
    return None if pe == 1 else float((pa - pe) / (1 - pe))


def agreement(values_by_reader, ordinal=None):
    """values_by_reader: {reader: {item: value}} -> dict of agreement statistics over items rated by all."""
    readers = sorted(values_by_reader)
    if len(readers) < 2:
        return None
    if ordinal:   # values outside the scale ("cannot grade", "cannot say") leave the item out of the ordinal statistics
        values_by_reader = {r: {i: (v if v in ordinal else None) for i, v in vals.items()} for r, vals in values_by_reader.items()}
    items = sorted(set.intersection(*[{i for i, v in values_by_reader[r].items() if v is not None} for r in readers]))
    if len(items) < 2:
        return {"n_items": len(items)}
    cats = ordinal or sorted({values_by_reader[r][i] for r in readers for i in items})
    table = [[sum(1 for r in readers if values_by_reader[r][i] == c) for c in cats] for i in items]
    exact = np.mean([len({values_by_reader[r][i] for r in readers}) == 1 for i in items])
    pairs = {}
    for r1, r2 in itertools.combinations(readers, 2):
        a = [values_by_reader[r1][i] for i in items]
        b = [values_by_reader[r2][i] for i in items]
        pairs[f"{r1}-{r2}"] = {"kappa": cohen_kappa(a, b, cats),
                               "kappa_quadratic": cohen_kappa(a, b, cats, "quadratic") if ordinal else None}
    return {"n_items": len(items), "all_agree": float(exact), "fleiss_kappa": fleiss_kappa(table),
            "gwet_ac1": gwet_ac1(table), "pairs": pairs}


# ------------------------------------------------------------------------------------------------ weighted shares


def weighted_share(rows, field, value, n_boot=2000, seed=0):
    """Design-weighted share of rows whose field equals value (representative arm), with an image bootstrap."""
    rows = [r for r in rows if r.get(field) is not None and r.get("weight")]
    if not rows:
        return None
    by_case = defaultdict(list)
    for r in rows:
        by_case[r["case_id"]].append(r)
    cases = sorted(by_case)

    def est(sample):
        num = sum(r["weight"] for c in sample for r in by_case[c] if r[field] == value)
        den = sum(r["weight"] for c in sample for r in by_case[c])
        return num / den if den else float("nan")

    point = est(cases)
    gen = common.rng(seed)
    boots = [est([cases[i] for i in gen.integers(0, len(cases), len(cases))]) for _ in range(n_boot)]
    boots = [b for b in boots if b == b]
    lo, hi = (np.percentile(boots, [2.5, 97.5]) if boots else (float("nan"), float("nan")))
    return {"share": point, "lo": float(lo), "hi": float(hi), "n_regions": len(rows), "n_images": len(cases)}


# ------------------------------------------------------------------------------------------------ main


def flatten(selection, reviews):
    inst = {x["inst_id"]: x for x in selection["instances"]}
    cases = {c["case_id"]: c for c in selection["cases"]}
    regions, images = [], []
    for rev in reviews:
        reader = rev["reader_id"]
        for cid, cs in rev["cases"].items():
            if cs.get("practice") or cid not in cases:
                continue
            c = cases[cid]
            pre, post = cs["pre"]["answers"], cs["post"]["answers"]
            images.append({"reader": reader, "case_id": cid, "set": c["set"], "image_id": c["image_id"], "grade": c.get("grade"),
                           **{k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in {**pre, **post}.items()},
                           "done": cs.get("done"), "active_s_pre": cs["pre"]["t"].get("active_ms", 0) / 1000,
                           "active_s_post": cs["post"]["t"].get("active_ms", 0) / 1000})
            for rid, rr in cs["regions"].items():
                if rid not in inst:
                    continue
                x = inst[rid]
                row = x["row"]
                a, b = rr["a"]["answers"], rr["b"]["answers"]
                regions.append({
                    "reader": reader, "inst_id": rid, "case_id": cid, "set": x["set"], "image_id": x["image_id"], "cls": x["cls"],
                    "kind": x["kind"], "subtype": x.get("subtype"), "arm": x["arm"], "stratum": x["stratum"], "weight": x.get("weight"),
                    "size_bin": row["size_bin"], "prob_max": row["prob_max"], "zoo_frac": row.get("zoo_frac"), "grade": row.get("grade"),
                    "dist_center_rel": row.get("dist_center_rel"),
                    "a_finding": a.get("a_finding"), "a_group": FINDING_GROUP.get(a.get("a_finding")), "a_confidence": a.get("a_confidence"),
                    "a_multiple": bool(a.get("a_multiple")), "blind_verdict": blind_verdict(x["kind"], x["cls"], a.get("a_finding")),
                    "b_cause": b.get("b_cause"), "b_label_issue": ";".join(b.get("b_label_issue", [])), "b_model_issue": ";".join(b.get("b_model_issue", [])),
                    "b_difficulty": b.get("b_difficulty"), "b_impact": b.get("b_impact"), "b_pivotal": b.get("b_pivotal"),
                    "b_decisions": ";".join(b.get("b_decisions", [])), "b_fix": ";".join(b.get("b_fix", [])),
                    "b_teaching": bool(b.get("b_teaching")), "b_comment": b.get("b_comment"),
                    "complete": bool(rr["b"]["t"].get("completed")),
                    "active_s": (rr["a"]["t"].get("active_ms", 0) + rr["b"]["t"].get("active_ms", 0)) / 1000,
                })
    return regions, images


def write_csv(path, rows):
    if not rows:
        Path(path).write_text("")
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def fmt(x, nd=2):
    return "-" if x is None or (isinstance(x, float) and x != x) else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def crosstab(rows, row_key, col_key):
    rows = [r for r in rows if r.get(col_key) is not None]
    rk = sorted({r[row_key] for r in rows})
    ck = sorted({r[col_key] for r in rows})
    cnt = Counter((r[row_key], r[col_key]) for r in rows)
    lines = ["| " + row_key + " | " + " | ".join(ck) + " | n |", "|" + "---|" * (len(ck) + 2)]
    for a in rk:
        tot = sum(cnt[(a, c)] for c in ck)
        lines.append(f"| {a} | " + " | ".join(str(cnt[(a, c)]) for c in ck) + f" | {tot} |")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selection", required=True)
    ap.add_argument("--reviews", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    selection = json.loads(Path(args.selection).read_text())
    reviews = [json.loads(Path(p).read_text()) for p in args.reviews]
    pack_ids = {r["pack_id"].split("+")[0] for r in reviews}
    if pack_ids != {selection["design"]["name"]}:
        raise SystemExit(f"reviews are for {pack_ids}, selection is {selection['design']['name']}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    regions, images = flatten(selection, reviews)
    write_csv(out / "regions_long.csv", regions)
    write_csv(out / "images_long.csv", images)
    readers = sorted({r["reader_id"] for r in reviews})
    n_regions = len(selection["instances"])

    md = [f"# Clinical review summary: {selection['design']['name']}", "",
          f"Readers: {', '.join(readers)}. Regions in the key: {n_regions} ({selection['report']['by_arm']}). Images: {len(selection['cases'])}.", ""]
    md += ["## Completion and time", "", "| reader | regions done | images done | median s / region | median s / image (pre+post) |", "|---|---|---|---|---|"]
    for rd in readers:
        rr = [r for r in regions if r["reader"] == rd and r["complete"]]
        ii = [i for i in images if i["reader"] == rd and i["done"]]
        md.append(f"| {rd} | {len(rr)} / {n_regions} | {len(ii)} / {len(selection['cases'])} | "
                  f"{fmt(float(np.median([r['active_s'] for r in rr])) if rr else None, 0)} | "
                  f"{fmt(float(np.median([i['active_s_pre'] + i['active_s_post'] for i in ii])) if ii else None, 0)} |")
    done = [r for r in regions if r["complete"]]

    md += ["", "## Blinded verdict (step A) by kind", "", crosstab(done, "kind", "blind_verdict"), ""]
    md += ["## Stated cause (step B) by kind", "", crosstab(done, "kind", "b_cause"), ""]
    md += ["## Stated cause by class (FP and FN regions)", "", crosstab([r for r in done if r["kind"] in ("FP", "FN")], "cls", "b_cause"), ""]

    md += ["## Design-weighted shares, representative arm (each reader separately)", "",
           "| reader | kind | quantity | share | 95% interval | regions | images |", "|---|---|---|---|---|---|---|"]
    for rd in readers:
        for kind, field, value in [("FP", "blind_verdict", "model_right"), ("FP", "b_cause", "label_error"),
                                   ("FN", "blind_verdict", "model_right"), ("FN", "b_cause", "model_error"),
                                   ("agree", "blind_verdict", "both_wrong")]:
            rows = [r for r in done if r["reader"] == rd and r["arm"] == "representative" and r["kind"] == kind]
            s = weighted_share(rows, field, value)
            if s:
                md.append(f"| {rd} | {kind} | {field} = {value} | {fmt(s['share'])} | {fmt(s['lo'])}-{fmt(s['hi'])} | {s['n_regions']} | {s['n_images']} |")

    md += ["", "## Agreement between readers", ""]
    if len(readers) < 2:
        md.append("One reader so far; agreement needs two or more.")
    else:
        md += ["| field | items | all agree | Fleiss kappa | Gwet AC1 | Cohen kappa by pair (quadratic) |", "|---|---|---|---|---|---|"]
        reg_fields = ["a_group", "a_finding", "blind_verdict", "b_cause", "b_impact", "b_pivotal", "b_difficulty"]
        for field in reg_fields:
            vals = {rd: {r["inst_id"]: r[field] for r in done if r["reader"] == rd} for rd in readers}
            ag = agreement(vals, ORDINAL.get(field))
            if ag and ag.get("n_items", 0) >= 2:
                pairs = "; ".join(f"{k} {fmt(v['kappa'])}" + (f" ({fmt(v['kappa_quadratic'])})" if v.get("kappa_quadratic") is not None else "") for k, v in ag["pairs"].items())
                md.append(f"| {field} | {ag['n_items']} | {fmt(ag['all_agree'])} | {fmt(ag['fleiss_kappa'])} | {fmt(ag['gwet_ac1'])} | {pairs} |")
        for field in ["pre_icdr", "pre_dme", "pre_quality", "post_model_grade", "post_model_referral", "post_annotation_quality", "post_model_quality"]:
            vals = {rd: {i["case_id"]: i.get(field) for i in images if i["reader"] == rd and i["done"]} for rd in readers}
            ag = agreement(vals, ORDINAL.get(field))
            if ag and ag.get("n_items", 0) >= 2:
                pairs = "; ".join(f"{k} {fmt(v['kappa'])}" + (f" ({fmt(v['kappa_quadratic'])})" if v.get("kappa_quadratic") is not None else "") for k, v in ag["pairs"].items())
                md.append(f"| {field} | {ag['n_items']} | {fmt(ag['all_agree'])} | {fmt(ag['fleiss_kappa'])} | {fmt(ag['gwet_ac1'])} | {pairs} |")

    errs = [r for r in done if r["b_cause"] and r["b_cause"] != "agree"]
    md += ["", "## Clinical impact of the differences (all arms, all readers)", "", crosstab(errs, "cls", "b_impact"), "",
           f"Decisive on its own: {sum(1 for r in errs if r['b_pivotal'] == 'yes')} of {len(errs)} region readings.", ""]
    for field, title in [("b_label_issue", "Annotation issues"), ("b_model_issue", "Model issues"), ("b_fix", "Suggested remedies"), ("b_decisions", "Decisions affected")]:
        cnt = Counter(code for r in done for code in (r[field].split(";") if r[field] else []))
        md += [f"## {title}", "", ", ".join(f"{k} {v}" for k, v in cnt.most_common()) or "-", ""]

    other = [r for r in done if r["a_finding"] in ("other_lesion", "indeterminate")]
    md += ["## Pilot checks (docs/clinical/28 section 8)", "",
           f"- median active time per region: {fmt(float(np.median([r['active_s'] for r in done])) if done else None, 0)} s (target <= 60)",
           f"- step A 'other lesion' or 'cannot determine': {len(other)} of {len(done)} ({fmt(len(other) / len(done) if done else None)}; target < 0.15)",
           f"- regions with a free-text comment: {sum(1 for r in done if r['b_comment'])}", ""]
    md += ["## Free text", ""]
    for r in done:
        if r["b_comment"]:
            md.append(f"- {r['reader']} {r['inst_id']} ({r['cls']}, {r['kind']}, {r['b_cause']}): {r['b_comment']}")
    for i in images:
        if i.get("post_key_problem") or i.get("pre_comment"):
            md.append(f"- {i['reader']} {i['case_id']} (image): {i.get('pre_comment') or ''} {i.get('post_key_problem') or ''}".rstrip())
    for rev in reviews:
        for k, v in rev.get("debrief", {}).get("answers", {}).items():
            if isinstance(v, str) and k in ("d_taxonomy", "d_priorities", "d_tool"):
                md.append(f"- {rev['reader_id']} debrief {k}: {v}")
    (out / "report.md").write_text("\n".join(md) + "\n")
    print(f"[analyze] {len(regions)} region readings, {len(images)} image readings -> {out}")


if __name__ == "__main__":
    main()
