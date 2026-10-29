"""Draw the images and regions a clinician will review, by a design file that is frozen before anyone reads.

docs/clinical/28 section 5. Two stages and two arms:

  stage 1  images: within each stratum of the design (set, grade band), a simple random sample of the eligible
           images (an image is eligible when it holds the minimum numbers of clusters the design asks for; a
           stratum may set its own rule, e.g. grade 0-1 images need only one discrepancy of any kind).
           Inclusion probability pi_image = n / N of the stratum.
  stage 2  regions inside the chosen images
           representative arm -- per image, for each entry of `representative.per_image` (a kind, optionally a
               class): pick a class uniformly among the classes that have such clusters in the image (or the
               given class), then a cluster uniformly within it. pi_within = 1 / (#classes available) x
               n / (#clusters of that class and kind). These regions carry design weights
               1 / (pi_image x pi_within) and give the prevalence estimates of the paper.
           enriched arm -- purposive draws for mechanisms and case material (confident false positives,
               confident misses, class confusions, extent disagreements, model-consensus regions); drawn at
               random among the clusters of the chosen images that meet the condition; no design weight.
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


def matches(row, where):
    for key, val in (where or {}).items():
        if key == "cls":
            if row["cls"] not in (val if isinstance(val, list) else [val]):
                return False
        elif key == "size_bin":
            if row["size_bin"] not in (val if isinstance(val, list) else [val]):
                return False
        elif key == "subtype_prefix":
            if not (row.get("subtype") or "").startswith(val):
                return False
        elif key.endswith("_ge"):
            if row.get(key[:-3]) is None or row[key[:-3]] < val:
                return False
        elif key.endswith("_le"):
            if row.get(key[:-3]) is None or row[key[:-3]] > val:
                return False
        elif row.get(key) != val:
            return False
    return True


def eligible(image_row, rule):
    tot = {k: sum(image_row["counts"][c][k] for c in CLASSES) for k in ("FN", "FP", "extent", "agree", "confusion")}
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

    # ---- stage 1: images
    chosen = []
    report = {"strata": []}
    for stratum in design["images"]["strata"]:
        pool = sorted(k for k, r in images.items()
                      if r["set"] == stratum["set"]
                      and (stratum.get("grade_band") is None or r["grade_band"] == stratum["grade_band"])
                      and eligible(r, stratum.get("eligibility", design["images"].get("eligibility"))))
        n = min(stratum["n"], len(pool))
        pick = [pool[i] for i in sorted(gen.choice(len(pool), size=n, replace=False))] if n else []
        label = f"{stratum['set']}|{stratum.get('grade_band', 'all')}"
        for key in pick:
            chosen.append({"key": key, "stratum": label, "pi_image": n / len(pool)})
        report["strata"].append({"stratum": label, "eligible": len(pool), "drawn": n, "asked": stratum["n"]})

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
                   "image_size": images[c["key"]]["image_size"], "fov": images[c["key"]]["fov"]} for c in sorted(cases, key=lambda c: c["case_id"])],
        "instances": instances,
        "report": {**report, "n_cases": len(cases), "n_instances": len(instances),
                   "by_arm": {a: sum(1 for x in instances if x["arm"] == a) for a in ("representative", "enriched")},
                   "redraws": redraws_total, "enriched_shortfall": shortfall},
    }
    (out / "selection.json").write_text(json.dumps(selection, indent=1, ensure_ascii=False) + "\n")
    with open(out / "selection.tsv", "w") as f:
        f.write("inst_id\tarm\tstratum\tset\timage_id\tcls\tkind\tsubtype\tsize_bin\tprob_max\tzoo_frac\tweight\n")
        for x in instances:
            r = x["row"]
            f.write("\t".join(str(v) for v in (x["inst_id"], x["arm"], x["stratum"], x["set"], x["image_id"], x["cls"], x["kind"],
                                               x["subtype"] or "", r["size_bin"], r["prob_max"], r.get("zoo_frac", ""),
                                               "" if x["weight"] is None else round(x["weight"], 3))) + "\n")
    print(json.dumps(selection["report"], indent=1))


if __name__ == "__main__":
    main()
