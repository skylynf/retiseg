"""Find every region where a model and the dataset annotation disagree, and the regions where they agree.

docs/clinical/28 section 4. For one model (the model under review), one prepared test set and one split:

  for each image and lesion class c
    valid  = field of view minus the ignore region of c (bench.common.io; Retinal-Lesions grey 127,
             MAPLES-DR uncertain layers)
    G      = annotation of c inside valid
    P      = probability of c >= threshold of c, inside valid (the model's own validation-chosen threshold)
    U      = G | P, labelled with 8-connectivity (docs/paper/14 M11); every connected region of U is a cluster

  each cluster gets a kind
    FN        annotation only (no predicted pixel in the cluster)
    FP        prediction only (no annotated pixel in the cluster)
    extent    both present, IoU of the two inside the cluster < --extent-iou (default 0.5); this includes an
              annotation and a prediction that touch without overlapping
    agree     both present, IoU >= --extent-iou
  and a subtype 'confusion' when an FP cluster lies mostly on the annotation of another class, or an FN
  cluster mostly under the prediction of another class (share >= --confusion-share).

Isolated clusters coincide with the lesion-level units of M11; where a prediction bridges several annotated
lesions they form one cluster, which is the unit a reader can judge in one look.

Per cluster: bounding box, centroid, areas, IoU, relative equivalent diameter and size bin (frozen edges of
bench/eval/protocol.yaml), the maximum and mean probability of c inside the cluster, overlap with the other
classes' annotation and prediction, distance from the FOV centre relative to the FOV radius, the image grade
when known, and -- with --zoo -- the share of other models that put a positive pixel of c in the cluster
("model consensus": a cluster that nearly every model flags and the annotation lacks is a candidate
annotation omission; one that no model finds is either a hard lesion or a doubtful label).

Output: <out>/candidates.jsonl (one row per cluster), <out>/images.jsonl (per image counts by class and kind,
the denominators of the sampling weights), <out>/meta.json (inputs, thresholds, checksums). Nothing in the
benchmark's own directories is written.

    python clinical/scripts/mine_discrepancies.py --set DDR --data dataset/prepared/DDR \
        --pred runs/ENS/REF/reference/DDR/prob --thresholds release/retiseg_ref/thresholds.json \
        --model RetiSeg-Ref --grades runs/CLIN/grades.json --out clinical/work/mine/RetiSeg-Ref/DDR \
        [--zoo-train DDR | --zoo SAMed=runs/BS_samed_ddr_seed0/prob_test:runs/BS_samed_ddr_seed0/metrics_test.json ...]

--zoo-train DDR takes the 18 registered ensemble members (bench/scripts/ensemble.py MEMBERS, seed 0, trained on
DDR) with the ensemble's own map resolver (prob_test for the internal test, runs/E3a/<run>/<set>/prob_test
otherwise) and each member's validation-optimal thresholds (runs/<run>/metrics_test.json); members whose maps for
the set are missing are listed in meta.json and left out.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from common import CLASSES  # noqa: E402

STRUCT8 = ndimage.generate_binary_structure(2, 2)


def parse_zoo(items):
    zoo = []
    for item in items or []:
        name, rest = item.split("=", 1)
        pred, thr = rest.rsplit(":", 1) if ":" in rest else (rest, "0.5")
        thresholds, source = common.load_thresholds(thr)
        zoo.append({"name": name, "pred": pred, "thresholds": thresholds, "threshold_source": source})
    return zoo


def fov_geometry(fov):
    ys, xs = np.nonzero(fov)
    area = int(fov.sum())
    return {"cx": float(xs.mean()), "cy": float(ys.mean()), "radius": float(np.sqrt(area / np.pi)),
            "diameter": common.equivalent_diameter(area), "area": area}


def zoo_from_ensemble(train, set_name):
    from bench.scripts import ensemble

    zoo, missing = [], []
    for name, _family, run_dir in ensemble.member_runs(train):
        pred = ensemble.map_dir(run_dir, train, set_name)
        if not pred.is_dir():
            missing.append({"name": name, "expected": str(pred)})
            continue
        thresholds, source = common.load_thresholds(run_dir / "metrics_test.json")
        zoo.append({"name": name, "pred": str(pred), "thresholds": thresholds, "threshold_source": source})
    return zoo, missing


def mine_image(set_name, data_dir, pred_dir, thresholds, image_id, zoo, grade, args, edges, names):
    fov, gt_all, ignore = common.read_case(data_dir, image_id)
    H, W = fov.shape
    geo = fov_geometry(fov)
    valid = {c: fov & ~ignore[c] for c in CLASSES}
    prob = {c: common.read_prob(pred_dir, c, image_id) for c in CLASSES}
    for c in CLASSES:
        if prob[c].shape != fov.shape:
            raise ValueError(f"{pred_dir} {c} {image_id}: probability map {prob[c].shape} vs image {fov.shape}")
    G = {c: gt_all[c] & valid[c] for c in CLASSES}
    P = {c: (prob[c] >= thresholds[c]) & valid[c] for c in CLASSES}
    zoo_pos = []
    for member in zoo:
        zoo_pos.append({c: (common.read_prob(member["pred"], c, image_id) >= member["thresholds"][c]) & valid[c] for c in CLASSES})

    rows = []
    counts = {c: {"FN": 0, "FP": 0, "extent": 0, "agree": 0, "confusion": 0} for c in CLASSES}
    for c in CLASSES:
        union = G[c] | P[c]
        lab, n = ndimage.label(union, structure=STRUCT8)
        if n == 0:
            continue
        idx = np.arange(1, n + 1)
        area_u = ndimage.sum_labels(np.ones_like(lab), lab, idx)
        area_g = ndimage.sum_labels(G[c], lab, idx)
        area_p = ndimage.sum_labels(P[c], lab, idx)
        area_i = ndimage.sum_labels(G[c] & P[c], lab, idx)
        pmax = ndimage.maximum(prob[c], lab, idx)
        pmean = ndimage.mean(prob[c], lab, idx)
        other_g = {o: ndimage.sum_labels(G[o], lab, idx) for o in CLASSES if o != c}
        other_p = {o: ndimage.sum_labels(P[o], lab, idx) for o in CLASSES if o != c}
        zoo_hit = [ndimage.maximum(z[c], lab, idx) for z in zoo_pos]
        slices = ndimage.find_objects(lab)
        cents = ndimage.center_of_mass(union, lab, idx)
        for k in range(n):
            ag, ap, ai, au = int(area_g[k]), int(area_p[k]), int(area_i[k]), int(area_u[k])
            if ag > 0 and ap == 0:
                kind = "FN"
            elif ap > 0 and ag == 0:
                kind = "FP"
            else:
                iou = ai / max(ag + ap - ai, 1)
                kind = "agree" if iou >= args.extent_iou else "extent"
            subtype = None
            own = ap if kind == "FP" else ag
            if kind == "FP":
                best = max(other_g, key=lambda o: other_g[o][k])
                if other_g[best][k] >= args.confusion_share * own:
                    subtype = f"confusion:{best}->{c}"     # annotated as `best`, predicted as c
            elif kind == "FN":
                best = max(other_p, key=lambda o: other_p[o][k])
                if other_p[best][k] >= args.confusion_share * own:
                    subtype = f"confusion:{c}->{best}"     # annotated as c, predicted as `best`
            sl = slices[k]
            y0, y1, x0, x1 = sl[0].start, sl[0].stop, sl[1].start, sl[1].stop
            cy, cx = cents[k]
            size_area = ag if ag > 0 else ap
            rel = common.equivalent_diameter(size_area) / geo["diameter"]
            row = {
                "set": set_name, "image_id": image_id, "cls": c, "cluster": int(k + 1), "kind": kind, "subtype": subtype,
                "bbox": [int(x0), int(y0), int(x1), int(y1)], "centroid": [round(float(cx), 1), round(float(cy), 1)],
                "area_gt": ag, "area_pred": ap, "area_inter": ai, "area_union": au,
                "iou": round(ai / max(ag + ap - ai, 1), 4),
                "rel_diameter": round(rel, 5), "size_bin": common.size_bin(rel, edges, names),
                "prob_max": round(float(pmax[k]), 5), "prob_mean": round(float(pmean[k]), 5),
                "threshold": thresholds[c],
                "other_gt": {o: int(other_g[o][k]) for o in other_g if other_g[o][k] > 0},
                "other_pred": {o: int(other_p[o][k]) for o in other_p if other_p[o][k] > 0},
                "dist_center_rel": round(float(np.hypot(cx - geo["cx"], cy - geo["cy"]) / geo["radius"]), 4),
                "grade": grade, "grade_band": common.grade_band(grade),
                "image_size": [W, H],
            }
            if zoo:
                hits = [bool(z[k]) for z in zoo_hit]
                row["zoo_n"] = len(hits)
                row["zoo_frac"] = round(sum(hits) / len(hits), 4)
                row["zoo_hits"] = [m["name"] for m, h in zip(zoo, hits) if h]
            rows.append(row)
            counts[c][kind] += 1
            if subtype:
                counts[c]["confusion"] += 1
    image_row = {"set": set_name, "image_id": image_id, "grade": grade, "grade_band": common.grade_band(grade),
                 "image_size": [W, H], "fov": geo, "counts": counts,
                 "n_gt_pixels": {c: int(G[c].sum()) for c in CLASSES}, "n_pred_pixels": {c: int(P[c].sum()) for c in CLASSES}}
    return rows, image_row


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", required=True, help="test set name as in the paper (DDR, IDRiD, TJDR_std, TJDR_uwf, MAPLES-DR, ...)")
    ap.add_argument("--data", required=True, help="prepared dataset directory")
    ap.add_argument("--pred", required=True, help="probability-map directory of the model under review (<CLS>/<id>.png)")
    ap.add_argument("--thresholds", required=True, help="metrics json, thresholds.json, {cls: t} json, or one number")
    ap.add_argument("--model", required=True, help="name of the model under review, e.g. RetiSeg-Ref")
    ap.add_argument("--split", default="test")
    ap.add_argument("--ids", nargs="*", help="restrict to these image ids (default: the whole split)")
    ap.add_argument("--grades", help="runs/CLIN/grades.json or a flat {id: grade} json")
    ap.add_argument("--zoo", nargs="*", help="NAME=PRED_DIR[:THRESHOLDS] of other models, for the consensus feature")
    ap.add_argument("--zoo-train", help="use the 18 registered ensemble members trained on this source (e.g. DDR) as the zoo")
    ap.add_argument("--extent-iou", type=float, default=0.5)
    ap.add_argument("--confusion-share", type=float, default=0.3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    data_dir = Path(args.data)
    ids = args.ids or common.bio.load_split(data_dir, args.split)
    thresholds, source = common.load_thresholds(args.thresholds)
    zoo = parse_zoo(args.zoo)
    zoo_missing = []
    if args.zoo_train:
        extra, zoo_missing = zoo_from_ensemble(args.zoo_train, args.set)
        zoo += extra
        print(f"[mine] zoo: {len(zoo)} models; missing maps for {[m['name'] for m in zoo_missing]}", flush=True)
    grades = common.load_grades(args.grades, args.set)
    edges, names = common.size_edges()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    all_rows, image_rows = [], []
    t0 = time.time()
    for n, image_id in enumerate(ids, 1):
        rows, image_row = mine_image(args.set, data_dir, args.pred, thresholds, image_id, zoo, grades.get(image_id), args, edges, names)
        all_rows.extend(rows)
        image_rows.append(image_row)
        if n % 10 == 0 or n == len(ids):
            print(f"[mine] {args.set} {n}/{len(ids)} images, {len(all_rows)} clusters, {time.time() - t0:.0f} s", flush=True)
    common.write_jsonl(out / "candidates.jsonl", all_rows)
    common.write_jsonl(out / "images.jsonl", image_rows)
    meta = {
        "set": args.set, "data": str(data_dir), "pred": str(args.pred), "split": args.split, "model": args.model,
        "thresholds": thresholds, "threshold_source": source, "extent_iou": args.extent_iou,
        "confusion_share": args.confusion_share, "size_edges": edges, "size_names": names,
        "zoo": [{k: v for k, v in m.items()} for m in zoo], "zoo_missing": zoo_missing, "grades": args.grades, "n_images": len(ids),
        "n_clusters": len(all_rows), "written": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "candidates_sha256": common.sha256_file(out / "candidates.jsonl"),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    tally = {}
    for r in all_rows:
        tally.setdefault(r["cls"], {}).setdefault(r["kind"], 0)
        tally[r["cls"]][r["kind"]] += 1
    print(json.dumps(tally))


if __name__ == "__main__":
    main()
