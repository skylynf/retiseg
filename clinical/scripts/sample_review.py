"""Draw the images and regions a clinician will review, by a design file that is frozen before anyone reads.

docs/clinical/28 section 5 and docs/clinical/32. Two stages and three arms:

  stage 1  images: every image of a set belongs to the FIRST stratum of the design it matches (set; optionally
           grade_band, grades, `where` on the image row, e.g. {ex_px.dd1_ge: 1}, and `has_regions`, e.g.
           {where: {cls: EX, etdrs_zone: [C, inner]}, n: 1}); so strata are mutually exclusive by construction.
           Within a stratum, a simple random sample of the eligible images (an image is eligible when it holds the
           minimum numbers of clusters the design asks for; a stratum may set its own rule).
           Inclusion probability pi_image = n / N_eligible of the stratum.
  stage 2  regions inside the chosen images
           stratified arm (formal batch, docs/clinical/32) -- `stratified.strata` is an ordered list of region
               strata; every cluster of an image belongs to the first stratum whose `where` it matches (clusters
               matching none are outside the population and reported). In each image, n_h clusters are drawn at
               random from stratum h (all of them when fewer). Design weight 1 / pi_image x N_ih / n_ih, with
               N_ih the clusters of stratum h in image i and n_ih those actually drawn: Horvitz-Thompson totals over
               all strata estimate the whole population of clusters, while small strata near the fovea are read far
               more often than their share.
           representative arm -- per image, for each entry of `representative.per_image` (a kind, optionally a
               class): pick a class uniformly among the classes that have such clusters in the image (or the
               given class), then a cluster uniformly within it. pi_within = 1 / (#classes available) x
               n / (#clusters of that class and kind). These regions carry design weights
               1 / (pi_image x pi_within) and give the prevalence estimates of the paper.
           enriched arm -- purposive draws for mechanisms and case material (confident false positives,
               confident misses, class confusions, extent disagreements, model-consensus regions); drawn at
               random among the clusters of the chosen images that meet the condition; no design weight.
  `exclude` lists images that are left out of every stratum (e.g. those the same reader saw in the trial).
  Crop windows of regions in one image never overlap, so revealing one region's outlines does not show the
  reader the answer of another (docs/clinical/28 section 6). A draw that would overlap is redrawn; the number
  of redraws is recorded, and weights of an image with redraws are flagged as approximate.

Output: selection.json (the key: everything the reader must not see -- source, image id, class, kind, arm,
weights) and selection.tsv. Case ids are anonymous (c01, c02, ...) in a random order.

    python clinical/scripts/sample_review.py --design clinical/designs/trial01.yaml --out runs/CLINREV/selection/trial01
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from common import CLASSES  # noqa: E402


def crop_box(row, geom):
    """Square crop window (x0, y0, x1, y1) around a cluster, inside the image."""
    x0, y0, x1, y1 = row["bbox"]
    W, H = row["image_size"]
    side = max(x1 - x0, y1 - y0)
    side = int(max(geom.get("crop_min", 384), side * (1 + 2 * geom.get("crop_margin", 0.6))))
    side = min(side, W, H)
    cx, cy = row["centroid"]
    bx0 = int(round(min(max(cx - side / 2, 0), W - side)))
    by0 = int(round(min(max(cy - side / 2, 0), H - side)))
    return [bx0, by0, bx0 + side, by0 + side]


def marker(row, geom):
    x0, y0, x1, y1 = row["bbox"]
    r = 0.5 * float(np.hypot(x1 - x0, y1 - y0)) + geom.get("marker_pad", 6)
    return {"cx": row["centroid"][0], "cy": row["centroid"][1], "r": round(max(r, geom.get("marker_min_radius", 14)), 1)}


def overlaps(a, b, gap=0):
    return not (a[2] + gap <= b[0] or b[2] + gap <= a[0] or a[3] + gap <= b[1] or b[3] + gap <= a[1])


def get(row, key):
    """row[key] with dotted paths (ex_px.dd1); None when absent."""
    v = row
    for part in key.split("."):
        if not isinstance(v, dict) or part not in v:
            return None
        v = v[part]
    return v


def matches(row, where):
    """Conditions of a design, all of which must hold.

    key: value (a list means 'one of'); key_ge / key_le: numeric bounds (absent values fail); subtype_prefix;
    ext_any: [NV, IRMA] (the cluster overlaps one of these extra labelled classes); any: [where, ...] (at least one
    holds); not: where. Keys may be dotted paths into nested fields.
    """
    for key, val in (where or {}).items():
        if key == "any":
            if not any(matches(row, w) for w in val):
                return False
        elif key == "not":
            if matches(row, val):
                return False
        elif key == "subtype_prefix":
            if not (row.get("subtype") or "").startswith(val):
                return False
        elif key == "ext_any":
            ov = row.get("ext_overlap") or {}
            if not any(ov.get(e, 0) > 0 for e in (val if isinstance(val, list) else [val])):
                return False
        elif key.endswith("_ge"):
            v = get(row, key[:-3])
            if v is None or v < val:
                return False
        elif key.endswith("_le"):
            v = get(row, key[:-3])
            if v is None or v > val:
                return False
        else:
            v = get(row, key)
            if isinstance(val, list):
                if v not in val:
                    return False
            elif v != val:
                return False
    return True


def image_matches(image_row, rows, stratum):
    if image_row["set"] != stratum["set"]:
        return False
    if stratum.get("grade_band") is not None and image_row["grade_band"] != stratum["grade_band"]:
        return False
    if stratum.get("grades") is not None and image_row.get("grade") not in stratum["grades"]:
        return False
    if not matches(image_row, stratum.get("where")):
        return False
    has = stratum.get("has_regions")
    for h in ([has] if isinstance(has, dict) else has or []):
        if sum(1 for r in rows if matches(r, h.get("where"))) < h.get("n", 1):
            return False
    return True


def region_stratum(row, strata):
    for h in strata:
        if matches(row, h.get("where")):
            return h["name"]
    return None


def eligible(image_row, rule):
    tot = {k: sum(image_row["counts"][c][k] for c in CLASSES) for k in ("FN", "FP", "extent", "agree", "confusion")}
    tot["discrepancy"] = tot["FN"] + tot["FP"] + tot["extent"]
    return all(tot.get(k, 0) >= n for k, n in (rule or {}).items())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--design", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--root", default=".", help="directory the design's relative paths are read from (default: cwd)")
    args = ap.parse_args(argv)

    design = yaml.safe_load(Path(args.design).read_text())
    root = Path(args.root)
    gen = common.rng(design["seed"])
    geom = design.get("geometry", {})
    gap = geom.get("min_gap", 0)

    sets, cands, images = {}, {}, {}
    for entry in design["sets"]:
        mdir = root / entry["candidates"]
        meta = json.loads((mdir / "meta.json").read_text())
        sets[entry["set"]] = {"meta": meta, "candidates_dir": str(entry["candidates"]),
                              "display": entry.get("display", {}), "reference_grade_scale": entry.get("reference_grade_scale")}
        for r in common.read_jsonl(mdir / "candidates.jsonl"):
            cands.setdefault((r["set"], r["image_id"]), []).append(r)
        for r in common.read_jsonl(mdir / "images.jsonl"):
            images[(r["set"], r["image_id"])] = r

    # ---- stage 1: images (each image in the first stratum it matches)
    excluded = {(e["set"], i) for e in design.get("exclude", []) for i in e["image_ids"]}
    for key in excluded:
        images.pop(key, None)
    strata = design["images"]["strata"]
    reg_total = sum(h.get("n", 1) for h in design.get("stratified", {}).get("strata", []))
    if reg_total > design.get("max_per_image", 6):
        print(f"[sample] warning: stratified quotas add up to {reg_total} > max_per_image; enriched draws only fill gaps")
    member = {}
    for key in sorted(images):
        for si, stratum in enumerate(strata):
            if image_matches(images[key], cands.get(key, []), stratum):
                member[key] = si
                break
    chosen = []
    report = {"strata": [], "excluded_images": sorted("|".join(k) for k in excluded)}
    for si, stratum in enumerate(strata):
        population = [k for k in sorted(images) if member.get(k) == si]
        pool = [k for k in population if eligible(images[k], stratum.get("eligibility", design["images"].get("eligibility")))]
        n = min(stratum["n"], len(pool))
        pick = [pool[i] for i in sorted(gen.choice(len(pool), size=n, replace=False))] if n else []
        label = stratum.get("name") or f"{stratum['set']}|{stratum.get('grade_band', 'all')}"
        for key in pick:
            chosen.append({"key": key, "stratum": label, "pi_image": n / len(pool)})
        report["strata"].append({"stratum": label, "population": len(population), "eligible": len(pool), "drawn": n,
                                 "asked": stratum["n"]})

    order = gen.permutation(len(chosen))
    cases = []
    for rank, i in enumerate(order, 1):
        c = chosen[i]
        cases.append({**c, "case_id": f"c{rank:02d}"})

    # ---- stage 2: regions
    instances = []
    redraws_total = 0
    max_per_image = design.get("max_per_image", 6)
    for case in cases:
        rows = cands.get(case["key"], [])
        taken, boxes, redraws = set(), [], 0

        def try_take(row, arm, stratum, pi_within=None, extra=None):
            nonlocal redraws
            box = crop_box(row, geom)
            if any(overlaps(box, b, gap) for b in boxes):
                redraws += 1
                return False
            taken.add((row["cls"], row["cluster"]))
            boxes.append(box)
            inst = {"case_id": case["case_id"], "arm": arm, "stratum": stratum, "set": row["set"], "image_id": row["image_id"],
                    "cls": row["cls"], "cluster": row["cluster"], "kind": row["kind"], "subtype": row.get("subtype"),
                    "crop": box, "marker": marker(row, geom), "pi_image": case["pi_image"], "pi_within": pi_within,
                    "weight": (1.0 / (case["pi_image"] * pi_within)) if pi_within else None, "row": row}
            inst.update(extra or {})
            instances.append(inst)
            return True

        reg_strata = design.get("stratified", {}).get("strata", [])
        if reg_strata:
            by_h = {}
            for r in rows:
                by_h.setdefault(region_stratum(r, reg_strata), []).append(r)
            case["outside_population"] = len(by_h.pop(None, []))
            case["region_strata"] = {}
            for h in reg_strata:
                pool = by_h.get(h["name"], [])
                want = min(h.get("n", 1), len(pool))
                got = []
                for j in gen.permutation(len(pool)):
                    if len(got) >= want:
                        break
                    if try_take(pool[j], "stratified", h["name"]):
                        got.append(instances[-1])
                for inst in got:                    # weight after the draw: N_ih / n_ih actually drawn
                    inst["pi_within"] = len(got) / len(pool)
                    inst["weight"] = len(pool) / (case["pi_image"] * len(got))
                    inst["N_stratum_image"] = len(pool)
                case["region_strata"][h["name"]] = {"N": len(pool), "asked": want, "drawn": len(got)}

        for entry in design.get("representative", {}).get("per_image", []):
            kind, want = entry["kind"], entry.get("n", 1)
            classes = [entry["cls"]] if entry.get("cls") else list(CLASSES)
            avail = [c for c in classes if any(r["kind"] == kind and r["cls"] == c for r in rows)]
            if not avail:
                continue
            for _ in range(want):
                cls = avail[int(gen.integers(len(avail)))]
                pool = [r for r in rows if r["kind"] == kind and r["cls"] == cls and (r["cls"], r["cluster"]) not in taken]
                if not pool:
                    break
                pi_within = (1.0 / len(avail)) * (want / len([r for r in rows if r["kind"] == kind and r["cls"] == cls]))
                for j in gen.permutation(len(pool)):
                    if try_take(pool[j], "representative", f"{kind}|{cls}", pi_within=min(pi_within, 1.0)):
                        break
        case["redraws"] = redraws
        redraws_total += redraws

    shortfall = []
    for entry in design.get("enriched", []):
        need = entry.get("n", 0)
        where = dict(entry.get("where") or {})
        if entry.get("kind"):
            where["kind"] = entry["kind"]
        pool = []
        for case in cases:
            for r in cands.get(case["key"], []):
                if matches(r, where):
                    pool.append((case, r))
        got = 0
        for j in gen.permutation(len(pool)):
            if got >= need:
                break
            case, r = pool[j]
            n_case = sum(1 for x in instances if x["case_id"] == case["case_id"])
            if n_case >= max_per_image or any(x["case_id"] == case["case_id"] and (x["cls"], x["cluster"]) == (r["cls"], r["cluster"]) for x in instances):
                continue
            boxes = [x["crop"] for x in instances if x["case_id"] == case["case_id"]]
            box = crop_box(r, geom)
            if any(overlaps(box, b, gap) for b in boxes):
                continue
            instances.append({"case_id": case["case_id"], "arm": "enriched", "stratum": entry["name"], "set": r["set"],
                              "image_id": r["image_id"], "cls": r["cls"], "cluster": r["cluster"], "kind": r["kind"],
                              "subtype": r.get("subtype"), "crop": box, "marker": marker(r, geom), "pi_image": case["pi_image"],
                              "pi_within": None, "weight": None, "row": r})
            got += 1
        if got < need:
            shortfall.append({"enriched": entry["name"], "asked": need, "drawn": got, "pool": len(pool)})

    # instance ids: random order inside each case
    for case in cases:
        mine = [x for x in instances if x["case_id"] == case["case_id"]]
        for k, i in enumerate(gen.permutation(len(mine)), 1):
            mine[i]["inst_id"] = f"{case['case_id']}-{k}"
    instances.sort(key=lambda x: x["inst_id"])

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    selection = {
        "design": design, "design_file": str(args.design), "written": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "sets": sets,
        "cases": [{"case_id": c["case_id"], "set": c["key"][0], "image_id": c["key"][1], "stratum": c["stratum"],
                   "pi_image": c["pi_image"], "grade": images[c["key"]].get("grade"), "redraws": c.get("redraws", 0),
                   "image_size": images[c["key"]]["image_size"], "fov": images[c["key"]]["fov"],
                   "landmark": images[c["key"]].get("landmark"), "ex_px": images[c["key"]].get("ex_px"),
                   "region_strata": c.get("region_strata"), "outside_population": c.get("outside_population")}
                  for c in sorted(cases, key=lambda c: c["case_id"])],
        "instances": instances,
        "report": {**report, "n_cases": len(cases), "n_instances": len(instances),
                   "by_arm": {a: sum(1 for x in instances if x["arm"] == a) for a in ("stratified", "representative", "enriched")},
                   "by_region_stratum": {h["name"]: sum(1 for x in instances if x["arm"] == "stratified" and x["stratum"] == h["name"])
                                         for h in design.get("stratified", {}).get("strata", [])},
                   "redraws": redraws_total, "enriched_shortfall": shortfall},
    }
    (out / "selection.json").write_text(json.dumps(selection, indent=1, ensure_ascii=False) + "\n")
    with open(out / "selection.tsv", "w") as f:
        f.write("inst_id\tarm\tstratum\tset\timage_id\tcls\tkind\tsubtype\tsize_bin\tetdrs_zone\tdist_min_um\telongation\t"
                "prob_max\tzoo_frac\tweight\n")
        for x in instances:
            r = x["row"]
            f.write("\t".join(str(v) for v in (x["inst_id"], x["arm"], x["stratum"], x["set"], x["image_id"], x["cls"], x["kind"],
                                               x["subtype"] or "", r["size_bin"], r.get("etdrs_zone") or "",
                                               "" if r.get("dist_min_um") is None else r["dist_min_um"], r.get("elongation", ""),
                                               r["prob_max"], r.get("zoo_frac", ""),
                                               "" if x["weight"] is None else round(x["weight"], 3))) + "\n")
    print(json.dumps(selection["report"], indent=1))


if __name__ == "__main__":
    main()
