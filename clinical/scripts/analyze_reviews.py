"""Join the readers' exported answers to the key and compute the trial / formal-batch summaries.

docs/clinical/28 section 7 and docs/clinical/32 (questionnaire 3.x, clinical/schema/review_schema.json; exports of
questionnaire 2.x, the trial of 2026-11-02, are read too). Inputs: selection.json (the key), one exported JSON per
reader and, optionally, the pack the readers read (--pack: its outlines give the DME implied by the label and by the
AI, and the lesions under the IRMA / NV marks). Outputs in --out:

    regions_long.csv   one row per reader x region: hidden metadata (set, class, kind, arm, stratum, weight, size, AI
                       probability, model consensus, grade, estimated ETDRS zone) + the reader's answers + derived
                       fields (blinded verdict, attribution, consequence level, ETDRS location from the reader's own
                       fovea and disc marks) + active time
    images_long.csv    one row per reader x image: grade, DME, other signs and marks, the DME implied by the dataset
                       label and by the AI, landmark check, post-review judgements, active time
    report.md          the summaries below

Blinded verdict (step A only, before any outline was shown): the reader's finding is mapped to a dataset class
(ma -> MA, hg -> HE, hex -> EX, cws -> SE; the readers see MA / HG / HEX / CWS), to "other DR sign" (irma, nv,
prh), "other lesion", "no lesion" (normal anatomy, artifacts) or "indeterminate"; then for a region of class c
      FP (AI only)        finding c -> model_right; no lesion -> dataset_right; another class -> other_class
      FN (label only)     finding c -> dataset_right; no lesion -> model_right; another class -> other_class
      agree / extent      finding c -> both_right;  no lesion -> both_wrong;   another class -> other_class
      indeterminate -> unresolved

Attribution (step B). Questionnaire 3: the reader judges the dataset label and the AI separately (b_label, b_ai:
correct / extent / wrong_type / false / missed / unsure); both correct -> both_right, only the label correct ->
ai_wrong, only the AI correct -> label_wrong, neither -> both_wrong, either unsure -> undecided_definition /
undecided_image (b_unsure_why). Questionnaire 2 (b_cause): agree -> both_right, label_error -> label_wrong,
model_error -> ai_wrong, both_wrong, ambiguous -> undecided_definition, image_limited -> undecided_image.

Rule-based thresholds (docs/clinical/31, 33): the highest ICDR grade threshold the reader ticked, as an ordinal
level: none 0; No DR <-> mild (g01) 1; mild or No DR <-> moderate (g12, g02) 2; moderate <-> severe (g23, any 4-2-1
criterion) 3; NPDR <-> PDR (g34) 4. A region whose
two outlines are both correct has level 0; ticking only HEX-band options leaves the ICDR level at 0. The HEX band
is reported on its own (hex_change: central subfield / 1 DD / none); it describes where the closest hard exudate
lies and is not a DME diagnosis (questionnaire 3.0's dme_1dd / dme_cs and questionnaire 2's "DME status" map onto
it). Whether an error matters for the patient is the reader's own answer (b_relevance), read against the reader's
management for the eye (pre_management, questionnaire 3.1).

ETDRS location (docs/clinical/32, 33): from the reader's own fovea mark; scale = 4,700 um / the reader's fovea-to-disc
distance (the pack's value when it states one), else the pack's per-image estimate (locate_landmarks.py); distance of
the closest point of the region's bounding box (and of its centroid); zones C <= 500 um (the central 1 mm), inner
500-1,500 (within 1 DD = 1,500 um), outer 1,500-3,000, outside. HEX band implied by the label / the AI: the zone of
the closest HEX outline of the pack to the reader's fovea (none when no HEX lies within 3,000 um), against the
reader's own answer.

Design-weighted shares (stratified or representative arm; docs/clinical/28 section 5, 33): ratio estimates
sum(w x indicator) / sum(w) over a domain, each reader separately, over the images that reader finished. The
design weight is multiplied by n_h / r_h of the image stratum (n_h images drawn, r_h finished by that reader), which
is exact when the finished images are a random subset of the stratum: true for a reading part (sample_review.py
`parts`) and for a reader who stops early, because each reader's image order is random. 95% intervals from a
bootstrap of images within image strata (2,000 replicates, seed 0). Practice cases are excluded.

    python clinical/scripts/analyze_reviews.py --selection runs/CLINREV/selection/batch01/selection.json \
        --pack runs/CLINREV/packs/batch01 --reviews runs/CLINREV/reviews/batch01/*.json \
        --out runs/CLINREV/analysis/batch01
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

FINDING_CLASS = {"ma": "MA", "hg": "HE", "hex": "EX", "cws": "SE"}
CLASS_LABEL = {"MA": "MA", "HE": "HG", "EX": "HEX", "SE": "CWS"}     # what the readers saw (docs/clinical/31)
FINDING_GROUP = {
    "ma": "MA", "hg": "HE", "hex": "EX", "cws": "SE",
    "irma": "other_dr", "nv": "other_dr", "prh": "other_dr",
    "drusen": "other_lesion", "laser": "other_lesion", "pigment": "other_lesion", "other_lesion": "other_lesion",
    "vessel": "no_lesion", "disc": "no_lesion", "reflex": "no_lesion", "background": "no_lesion",
    "artifact_dust": "no_lesion", "artifact_light": "no_lesion", "artifact_edge": "no_lesion",
    "indeterminate": "indeterminate",
}
ICDR_ORDER = ["none", "g01", "g02", "g12", "g23", "g34", "hex_1dd", "hex_c"]   # questionnaire 3.1 options of b_icdr
CONSEQUENCE_LEVEL = {"none": 0, "g01": 1, "g02": 2, "g12": 2, "g23": 3, "g34": 4}
LEVEL_NAME = {0: "0 none", 1: "1 No DR-mild", 2: "2 to or from moderate", 3: "3 moderate-severe", 4: "4 NPDR-PDR"}
HEX_CHANGE = {"hex_c": "central", "dme_cs": "central", "hex_1dd": "1dd", "dme_1dd": "1dd", "dme": "unspecified"}
HEX_CHANGE_ORDER = ["none", "1dd", "central", "unspecified"]
MANAGEMENT_ORDER = ["routine", "shorter", "refer", "urgent", "cannot"]
V2_ATTRIBUTION = {"agree": "both_right", "label_error": "label_wrong", "model_error": "ai_wrong",
                  "both_wrong": "both_wrong", "ambiguous": "undecided_definition", "image_limited": "undecided_image"}
ZONE_ORDER = ["none", "outer", "inner", "central"]                       # pre_hex_zone (3.0: pre_dme), by closeness
ORDINAL = {"pre_icdr": ["0", "1", "2", "3", "4"], "hex_zone": ZONE_ORDER, "consequence": ["0", "1", "2", "3", "4"],
           "pre_management": MANAGEMENT_ORDER[:4]}
ETDRS_UM = (500.0, 1500.0, 3000.0)
ZONES = ("C", "inner", "outer", "outside")
FOVEA_DISC_UM = 4700.0          # adult mean about 4.6-4.8 mm (docs/clinical/33); a pack's schema may state its own
DISCREPANCY = ("FN", "FP", "extent")
# step-B codes that cannot describe an outline the region does not have (AI-only: no label outline; label-only: no AI)
IMPLAUSIBLE = {"FP": ("b_label", {"extent", "wrong_type", "false"}), "FN": ("b_ai", {"extent", "wrong_type", "false"})}


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


def attribution(b):
    """Who is wrong in step B, from the two outline judgements (questionnaire 3) or b_cause (questionnaire 2)."""
    if "b_label" in b or "b_ai" in b:
        lab, ai = b.get("b_label"), b.get("b_ai")
        if lab is None or ai is None:
            return None
        if "unsure" in (lab, ai):
            return {"definition": "undecided_definition", "image": "undecided_image"}.get(b.get("b_unsure_why"), "undecided")
        return {(True, True): "both_right", (True, False): "ai_wrong", (False, True): "label_wrong",
                (False, False): "both_wrong"}[(lab == "correct", ai == "correct")]
    return V2_ATTRIBUTION.get(b.get("b_cause"))


def consequence_level(codes, attrib):
    """Highest ICDR threshold ticked (0-4); 0 when only HEX bands were ticked; None when nothing was answered."""
    if attrib == "both_right":
        return 0
    if not codes:
        return None
    return max([CONSEQUENCE_LEVEL[c] for c in codes if c in CONSEQUENCE_LEVEL] or [0])


def hex_change(codes, attrib):
    """The HEX band the error moves the closest hard exudate across: central / 1dd / none (None: unanswered)."""
    if attrib == "both_right":
        return "none"
    if not codes:
        return None
    hits = [HEX_CHANGE[c] for c in codes if c in HEX_CHANGE]
    for z in ("central", "1dd", "unspecified"):
        if z in hits:
            return z
    return "none"


def reader_hex_zone(pre, version=None):
    """The reader's closest-HEX band: pre_hex_zone (3.1), pre_dme (3.0, same codes). Questionnaire 2's pre_dme
    (none / non-central / center-involving DME) is a different question and does not map."""
    v = pre.get("pre_hex_zone")
    if v is None and str(version or "").startswith("3.0"):
        v = pre.get("pre_dme")
    return v if v in ZONE_ORDER or v == "cannot" else None


def zone_of(dist_um):
    if dist_um is None:
        return None
    return ZONES[int(np.searchsorted(ETDRS_UM, dist_um, side="left"))]


def reader_geometry(case_key, case_state, pack_case=None):
    """Fovea (display px), um per display px and its source for one reader and image; None without a fovea mark.

    Display pixels are the pack's (the readers click on the pack image); s = display / native width converts the
    key's native coordinates.
    """
    pre = case_state["pre"]["answers"]
    fovea, disc = pre.get("pre_fovea"), pre.get("pre_disc")
    native_w = case_key["image_size"][0]
    shown_w = (pack_case or {}).get("width") or (case_state.get("display_size") or case_key["image_size"])[0]
    s = shown_w / native_w
    if not fovea or fovea.get("x") is None:
        return None
    umpx, source, nasal = None, None, 0
    if disc and disc.get("x") is not None:
        dist = float(np.hypot(fovea["x"] - disc["x"], fovea["y"] - disc["y"]))
        if dist > 10:
            umpx, source = FOVEA_DISC_UM / dist, "reader"
            nasal = 1 if disc["x"] >= fovea["x"] else -1
    if umpx is None:
        lm = case_key.get("landmark") or {}
        if lm.get("um_per_px"):
            umpx, source = lm["um_per_px"] / s, "estimate"
    if umpx is None:
        return None
    return {"fx": float(fovea["x"]), "fy": float(fovea["y"]), "umpx": umpx, "source": source, "s": s, "nasal": nasal}


def region_location(geo, row):
    """(closest-point distance um, centroid distance um, zone) of a key region from the reader's fovea."""
    if not geo:
        return None, None, None
    s = geo["s"]
    x0, y0, x1, y1 = (v * s for v in row["bbox"])
    cx, cy = (v * s for v in row["centroid"])
    nx, ny = min(max(geo["fx"], x0), x1), min(max(geo["fy"], y0), y1)
    dmin = float(np.hypot(nx - geo["fx"], ny - geo["fy"])) * geo["umpx"]
    dcen = float(np.hypot(cx - geo["fx"], cy - geo["fy"])) * geo["umpx"]
    return round(dmin), round(dcen), zone_of(dmin)


def _inside(px, py, pts):
    x, y = pts[:, 0], pts[:, 1]
    x2, y2 = np.roll(x, -1), np.roll(y, -1)
    cond = (y > py) != (y2 > py)
    with np.errstate(divide="ignore", invalid="ignore"):
        xint = x + (py - y) * (x2 - x) / (y2 - y)
    return bool(np.sum(cond & (px < xint)) % 2)


def nearest_outline_px(loops, px, py):
    """Distance in px from a point to a set of outline loops (flat [x0, y0, x1, y1, ...]); 0 inside (even-odd)."""
    best, inside = None, False
    for flat in loops or []:
        pts = np.asarray(flat, dtype=float).reshape(-1, 2)
        if len(pts) == 0:
            continue
        if len(pts) >= 3 and _inside(px, py, pts):
            inside = not inside
        a, b = pts, np.roll(pts, -1, axis=0)
        ab = b - a
        L = (ab ** 2).sum(1)
        t = np.clip(np.where(L > 0, ((px - a[:, 0]) * ab[:, 0] + (py - a[:, 1]) * ab[:, 1]) / np.where(L > 0, L, 1), 0), 0, 1)
        d = np.hypot(a[:, 0] + t * ab[:, 0] - px, a[:, 1] + t * ab[:, 1] - py).min()
        best = d if best is None else min(best, d)
    if inside:
        return 0.0
    return None if best is None else float(best)


def implied_zone(geo, loops):
    """Zone of the closest HEX outline to the reader's fovea, as a pre_hex_zone code (none when beyond 3,000 um)."""
    if not geo:
        return None, None
    d = nearest_outline_px(loops, geo["fx"], geo["fy"])
    if d is None:
        return "none", None
    um = d * geo["umpx"]
    z = zone_of(um)
    return {"C": "central", "inner": "inner", "outer": "outer", "outside": "none"}[z], round(um)


def load_pack(paths):
    """({case id: case}, fovea-to-disc um or None) from one or more pack folders (pack.json or pack.js)."""
    cases, fd = {}, None
    for path in ([paths] if isinstance(paths, str) else paths or []):
        p = Path(path)
        if (p / "pack.json").is_file():
            data = json.loads((p / "pack.json").read_text())
        else:
            text = (p / "pack.js").read_text() if p.is_dir() else p.read_text()
            data = json.loads(text[text.index("{"): text.rindex("}") + 1])
        cases.update({c["id"]: c for c in data["cases"]})
        fd = (data.get("schema", {}).get("etdrs") or {}).get("fovea_disc_um") or fd
    return cases, fd


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


def weighted_share(rows, field, value, n_boot=2000, seed=0, wkey="weight"):
    """Design-weighted share of rows whose field equals value, with a bootstrap of images within image strata."""
    rows = [r for r in rows if r.get(field) is not None and r.get(wkey)]
    if not rows:
        return None
    by_case = defaultdict(list)
    for r in rows:
        by_case[r["case_id"]].append(r)
    cases = sorted(by_case)
    strata = defaultdict(list)
    for c in cases:
        strata[by_case[c][0].get("image_stratum")].append(c)

    def est(sample):
        num = sum(r[wkey] for c in sample for r in by_case[c] if r[field] == value)
        den = sum(r[wkey] for c in sample for r in by_case[c])
        return num / den if den else float("nan")

    point = est(cases)
    gen = common.rng(seed)
    boots = [est([cs[i] for cs in strata.values() for i in gen.integers(0, len(cs), len(cs))]) for _ in range(n_boot)]
    boots = [b for b in boots if b == b]
    lo, hi = (np.percentile(boots, [2.5, 97.5]) if boots else (float("nan"), float("nan")))
    return {"share": point, "lo": float(lo), "hi": float(hi), "n_regions": len(rows), "n_images": len(cases)}


# ------------------------------------------------------------------------------------------------ main


def _flat(v):
    return ";".join(v) if isinstance(v, list) and all(isinstance(x, str) for x in v) else json.dumps(v) if isinstance(v, (dict, list)) else v


def flatten(selection, reviews, pack):
    inst = {x["inst_id"]: x for x in selection["instances"]}
    cases = {c["case_id"]: c for c in selection["cases"]}
    regions, images = [], []
    for rev in reviews:
        reader = rev["reader_id"]
        version = rev.get("schema_version")
        for cid, cs in rev["cases"].items():
            if cs.get("practice") or cid not in cases:
                continue
            c = cases[cid]
            pc = pack.get(cid)
            pre, post = cs["pre"]["answers"], cs["post"]["answers"]
            geo = reader_geometry(c, cs, pc)
            marks = pre.get("pre_marks") or []
            img = {"reader": reader, "case_id": cid, "set": c["set"], "image_id": c["image_id"], "grade": c.get("grade"),
                   "image_stratum": c.get("stratum"), "part": c.get("part"), "hex_zone": reader_hex_zone(pre, version),
                   **{k: _flat(v) for k, v in {**pre, **post}.items() if k != "pre_marks"},
                   "n_irma_marks": sum(1 for m in marks if m.get("kind") == "irma"),
                   "n_nv_marks": sum(1 for m in marks if m.get("kind") == "nv"),
                   "scale_source": geo["source"] if geo else None,
                   "um_per_px_reader": round(geo["umpx"], 4) if geo else None,
                   "done": cs.get("done"), "active_s_pre": cs["pre"]["t"].get("active_ms", 0) / 1000,
                   "active_s_post": cs["post"]["t"].get("active_ms", 0) / 1000}
            lm = c.get("landmark") or {}
            if geo and lm.get("fovea"):
                img["fovea_vs_estimate_um"] = round(float(np.hypot(lm["fovea"][0] * geo["s"] - geo["fx"], lm["fovea"][1] * geo["s"] - geo["fy"])) * geo["umpx"])
                if geo["source"] == "reader" and lm.get("um_per_px"):
                    img["scale_reader_over_estimate"] = round(geo["umpx"] / (lm["um_per_px"] / geo["s"]), 3)
            if pc:
                for src, key in (("label", "annotation"), ("ai", "model")):
                    zone, um = implied_zone(geo, pc["overlays"][key].get("EX"))
                    img[f"hex_zone_{src}"], img[f"hex_zone_{src}_um"] = zone, um
                for m in marks:      # what the four-class outlines call the place a reader marked IRMA or NV
                    for src, key in (("label", "annotation"), ("ai", "model")):
                        hit = []
                        for cls in ("HE", "MA", "EX", "SE"):
                            d = nearest_outline_px(pc["overlays"][key].get(cls), m["x"], m["y"])
                            if d is not None and d <= 10:
                                hit.append(CLASS_LABEL[cls])
                        m[f"under_{src}"] = "+".join(hit) or "nothing"
            elif lm.get("fovea") is not None and c.get("ex_px"):
                for src in ("label", "ai"):        # no pack: the miner's figures from the estimated fovea
                    um = c["ex_px"].get(f"nearest_{src}_um")
                    img[f"hex_zone_{src}"] = "none" if um is None else {"C": "central", "inner": "inner", "outer": "outer", "outside": "none"}[zone_of(um)]
                    img[f"hex_zone_{src}_um"] = um
                img["hex_zone_implied_from"] = "estimated fovea"
            img["marks"] = marks
            images.append(img)
            for rid, rr in cs["regions"].items():
                if rid not in inst:
                    continue
                x = inst[rid]
                row = x["row"]
                a, b = rr["a"]["answers"], rr["b"]["answers"]
                dmin, dcen, zone = region_location(geo, row)
                attrib = attribution(b)
                level = consequence_level(b.get("b_icdr"), attrib)
                implausible = None
                if x["kind"] in IMPLAUSIBLE:
                    field, bad = IMPLAUSIBLE[x["kind"]]
                    implausible = b.get(field) in bad
                regions.append({
                    "reader": reader, "inst_id": rid, "case_id": cid, "set": x["set"], "image_id": x["image_id"],
                    "cls": x["cls"], "label": CLASS_LABEL.get(x["cls"], x["cls"]),
                    "kind": x["kind"], "subtype": x.get("subtype"), "arm": x["arm"], "stratum": x["stratum"],
                    "image_stratum": c.get("stratum"), "part": c.get("part"), "case_done": bool(cs.get("done")), "weight": x.get("weight"),
                    "size_bin": row["size_bin"], "prob_max": row["prob_max"], "zoo_frac": row.get("zoo_frac"), "grade": row.get("grade"),
                    "elongation": row.get("elongation"), "zone_estimated": row.get("etdrs_zone"),
                    "reader_icdr": pre.get("pre_icdr"), "reader_hex_zone": reader_hex_zone(pre, version), "reader_management": pre.get("pre_management"),
                    "dist_min_um": dmin, "dist_centroid_um": dcen, "zone": zone, "scale_source": geo["source"] if geo else None,
                    "a_finding": a.get("a_finding"), "a_group": FINDING_GROUP.get(a.get("a_finding")), "a_confidence": a.get("a_confidence"),
                    "blind_verdict": blind_verdict(x["kind"], x["cls"], a.get("a_finding")),
                    "b_label": b.get("b_label"), "b_ai": b.get("b_ai"), "b_unsure_why": b.get("b_unsure_why"),
                    "b_cause": b.get("b_cause"), "attribution": attrib, "implausible": implausible,
                    "b_label_issue": ";".join(b.get("b_label_issue", [])), "b_model_issue": ";".join(b.get("b_model_issue", [])),
                    "b_icdr": ";".join(b.get("b_icdr", [])),
                    "consequence": None if level is None else str(level),
                    "consequence_name": None if level is None else LEVEL_NAME[level],
                    "hex_change": hex_change(b.get("b_icdr"), attrib),
                    "b_relevance": b.get("b_relevance") or ("no" if attrib == "both_right" else None),
                    "b_comment": b.get("b_comment"),
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


def crosstab(rows, row_key, col_key, row_order=None, col_order=None):
    rows = [r for r in rows if r.get(col_key) is not None and r.get(row_key) is not None]
    if not rows:
        return "(none)"
    rk = [k for k in (row_order or []) if any(r[row_key] == k for r in rows)] + sorted({r[row_key] for r in rows} - set(row_order or []), key=str)
    ck = [k for k in (col_order or []) if any(r[col_key] == k for r in rows)] + sorted({r[col_key] for r in rows} - set(col_order or []), key=str)
    cnt = Counter((r[row_key], r[col_key]) for r in rows)
    lines = ["| " + row_key + " | " + " | ".join(str(c) for c in ck) + " | n |", "|" + "---|" * (len(ck) + 2)]
    for a in rk:
        tot = sum(cnt[(a, c)] for c in ck)
        lines.append(f"| {a} | " + " | ".join(str(cnt[(a, c)]) for c in ck) + f" | {tot} |")
    return "\n".join(lines)


def agreement_row(field, vals, ordinal=None):
    ag = agreement(vals, ordinal)
    if not ag or ag.get("n_items", 0) < 2:
        return None
    pairs = "; ".join(f"{k} {fmt(v['kappa'])}" + (f" ({fmt(v['kappa_quadratic'])})" if v.get("kappa_quadratic") is not None else "")
                      for k, v in ag["pairs"].items())
    return f"| {field} | {ag['n_items']} | {fmt(ag['all_agree'])} | {fmt(ag['fleiss_kappa'])} | {fmt(ag['gwet_ac1'])} | {pairs} |"


ATTRIB_ORDER = ["both_right", "label_wrong", "ai_wrong", "both_wrong", "undecided_definition", "undecided_image", "undecided"]
VERDICT_ORDER = ["model_right", "dataset_right", "both_right", "both_wrong", "other_class", "unresolved"]
REL_ORDER = ["no", "followup", "referral", "unsure"]
LEVEL_ORDER = [LEVEL_NAME[k] for k in range(5)]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selection", required=True)
    ap.add_argument("--reviews", nargs="+", required=True)
    ap.add_argument("--pack", nargs="+", help="the pack folder(s) the readers read, e.g. part 1 and part 2 (outlines for the implied "
                                         "HEX band and the IRMA / NV marks)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    selection = json.loads(Path(args.selection).read_text())
    reviews = [json.loads(Path(p).read_text()) for p in args.reviews]
    pack_ids = {r["pack_id"].split("+")[0] for r in reviews}
    if pack_ids != {selection["design"]["name"]}:
        raise SystemExit(f"reviews are for {pack_ids}, selection is {selection['design']['name']}")
    global FOVEA_DISC_UM
    pack, fd = load_pack(args.pack)
    if fd:
        FOVEA_DISC_UM = float(fd)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    regions, images = flatten(selection, reviews, pack)
    # completion: weight x n_h / r_h per reader and image stratum (finished images only; docstring)
    n_h = Counter(c.get("stratum") for c in selection["cases"])
    r_h = Counter((i["reader"], i["image_stratum"]) for i in images if i["done"])
    for r in regions:
        k = r_h.get((r["reader"], r["image_stratum"]), 0)
        r["weight_adj"] = r["weight"] * n_h[r["image_stratum"]] / k if (r["weight"] and r["case_done"] and k) else None
    write_csv(out / "regions_long.csv", regions)
    write_csv(out / "images_long.csv", [{k: (_flat(v) if k == "marks" else v) for k, v in i.items()} for i in images])
    readers = sorted({r["reader_id"] for r in reviews})
    n_regions = len(selection["instances"])
    versions = sorted({str(r.get("schema_version")) for r in reviews})
    for r in regions:
        r["label_kind"] = f"{r['label']} {r['kind']}"
        r["near"] = None if r["zone"] is None else ("within 1 DD" if r["zone"] in ("C", "inner") else "beyond 1 DD")

    md = [f"# Clinical review summary: {selection['design']['name']}", "",
          f"Readers: {', '.join(readers)} (questionnaire {', '.join(versions)}). Regions in the key: {n_regions} "
          f"({selection['report']['by_arm']}). Images: {len(selection['cases'])}.",
          "Lesion labels as the readers saw them: MA, HG (hemorrhage), HEX (hard exudate), CWS (cotton wool spot).",
          f"Pack outlines {'read from ' + ', '.join(args.pack) if pack else 'not given (--pack): implied HEX band from the estimated fovea where the key has it'}; "
          f"fovea-to-disc distance taken as {FOVEA_DISC_UM / 1000:.1f} mm.", ""]
    md += ["## Completion and time", "", "| reader | regions done | images done | median s / region | median s / image (grade + whole image) | median min / image incl. regions | total min |",
           "|---|---|---|---|---|---|---|"]
    for rd in readers:
        rr = [r for r in regions if r["reader"] == rd and r["complete"]]
        ii = [i for i in images if i["reader"] == rd and i["done"]]
        per_img = [(i["active_s_pre"] + i["active_s_post"] + sum(r["active_s"] for r in regions if r["reader"] == rd and r["case_id"] == i["case_id"])) / 60 for i in ii]
        total = (sum(r["active_s"] for r in regions if r["reader"] == rd) + sum(i["active_s_pre"] + i["active_s_post"] for i in images if i["reader"] == rd)) / 60
        md.append(f"| {rd} | {len(rr)} / {n_regions} | {len(ii)} / {len(selection['cases'])} | "
                  f"{fmt(float(np.median([r['active_s'] for r in rr])) if rr else None, 0)} | "
                  f"{fmt(float(np.median([i['active_s_pre'] + i['active_s_post'] for i in ii])) if ii else None, 0)} | "
                  f"{fmt(float(np.median(per_img)) if per_img else None, 1)} | {fmt(total, 0)} |")
    strata_names = sorted(n_h, key=str)
    md += ["", "Images finished by image stratum (weights are scaled by drawn / finished within each stratum)", "",
           "| reader | " + " | ".join(str(h) for h in strata_names) + " |", "|---|" + "---|" * len(strata_names)]
    for rd in readers:
        md.append(f"| {rd} | " + " | ".join(f"{r_h.get((rd, h), 0)} / {n_h[h]}" for h in strata_names) + " |")
    done = [r for r in regions if r["complete"]]
    disc = [r for r in done if r["kind"] in DISCREPANCY]

    md += ["", "## Blinded verdict (step A) by kind", "", crosstab(done, "kind", "blind_verdict", col_order=VERDICT_ORDER), ""]
    md += ["## Attribution (step B) by kind", "",
           "both_right: both outlines acceptable; label_wrong: only the AI right; ai_wrong: only the label right; "
           "both_wrong; undecided_*: the reader could not decide (definition or image).", "",
           crosstab(done, "kind", "attribution", col_order=ATTRIB_ORDER), "",
           "By lesion type, discrepancies only", "", crosstab(disc, "label_kind", "attribution", col_order=ATTRIB_ORDER), ""]
    if any(r["b_label"] for r in done):
        md += ["What is wrong with each outline (questionnaire 3)", "", "Dataset label", "", crosstab(done, "kind", "b_label"), "",
               "AI", "", crosstab(done, "kind", "b_ai"), "",
               f"Implausible combinations (an error ascribed to an outline the region does not have): "
               f"{sum(1 for r in done if r['implausible'])} of {sum(1 for r in done if r['implausible'] is not None)}.", ""]

    located = [r for r in done if r["zone"]]
    md += ["## ETDRS location (docs/clinical/32)", "",
           f"Regions with a location: {len(located)} of {len(done)} (scale from the reader's disc mark: "
           f"{sum(1 for r in located if r['scale_source'] == 'reader')}; from the estimate: {sum(1 for r in located if r['scale_source'] == 'estimate')}).", "",
           "Zone of each region (closest point) by lesion type", "", crosstab(located, "label", "zone", col_order=list(ZONES)), "",
           "Highest ICDR threshold crossed by zone (discrepancies)", "", crosstab([r for r in located if r["kind"] in DISCREPANCY], "zone", "consequence_name", list(ZONES), LEVEL_ORDER), "",
           "HEX band changed by zone (discrepancies)", "", crosstab([r for r in located if r["kind"] in DISCREPANCY], "zone", "hex_change", list(ZONES), HEX_CHANGE_ORDER), "",
           "Would change management (reader's judgement) by zone (discrepancies)", "", crosstab([r for r in located if r["kind"] in DISCREPANCY], "zone", "b_relevance", list(ZONES), REL_ORDER), "",
           "AI-only regions: blinded verdict by zone (model_right = lesion missed by the label)", "",
           crosstab([r for r in located if r["kind"] == "FP"], "zone", "blind_verdict", list(ZONES), VERDICT_ORDER), "",
           "Label-only regions: blinded verdict by zone (dataset_right = lesion missed by the AI)", "",
           crosstab([r for r in located if r["kind"] == "FN"], "zone", "blind_verdict", list(ZONES), VERDICT_ORDER), "",
           "Estimated zone (sampling) against the zone from the reader's marks", "",
           crosstab([r for r in located if r["zone_estimated"]], "zone_estimated", "zone", list(ZONES), list(ZONES)), ""]

    if any(r["arm"] == "stratified" for r in done):
        md += ["## Region strata (unweighted; docs/clinical/32)", "",
               "| stratum | regions | label_wrong | ai_wrong | both_wrong | undecided | ICDR mild-moderate or higher | HEX band changed | management changed |",
               "|---|---|---|---|---|---|---|---|---|"]
        for h in [s["name"] for s in selection["design"].get("stratified", {}).get("strata", [])]:
            rr = [r for r in done if r["arm"] == "stratified" and r["stratum"] == h]
            if not rr:
                continue
            n = len(rr)
            md.append(f"| {h} | {n} | {sum(r['attribution'] == 'label_wrong' for r in rr)} | {sum(r['attribution'] == 'ai_wrong' for r in rr)} | "
                      f"{sum(r['attribution'] == 'both_wrong' for r in rr)} | {sum((r['attribution'] or '').startswith('undecided') for r in rr)} | "
                      f"{sum(r['consequence'] in ('2', '3', '4') for r in rr)} | {sum(r['hex_change'] in ('1dd', 'central', 'unspecified') for r in rr)} | "
                      f"{sum(r['b_relevance'] in ('followup', 'referral') for r in rr)} |")
        md += ["", "Step A finding in the IRMA / NV suspect stratum", "",
               crosstab([r for r in done if r["stratum"] == "nv_irma_suspect"], "label_kind", "a_finding"), ""]

    md += ["## Rule-based thresholds and the reader's management", "",
           "ICDR grade thresholds and HEX bands are rule-based descriptions; whether the error changes management is the "
           "reader's own judgement. Highest ICDR threshold crossed by lesion type (discrepancies)", "",
           crosstab(disc, "label", "consequence_name", col_order=LEVEL_ORDER), "",
           "HEX band changed by lesion type (discrepancies)", "", crosstab(disc, "label", "hex_change", col_order=HEX_CHANGE_ORDER), "",
           "ICDR threshold (rows) against the reader's management answer (columns): a threshold crossed without a change "
           "in management is where the rule and the reader's practice differ.", "",
           crosstab(disc, "consequence_name", "b_relevance", LEVEL_ORDER, REL_ORDER), "",
           "HEX band changed (rows) against the reader's management answer (columns)", "",
           crosstab(disc, "hex_change", "b_relevance", HEX_CHANGE_ORDER, REL_ORDER), "",
           "The reader's management for the eye (rows, before any outline) against the per-region answer (columns)", "",
           crosstab(disc, "reader_management", "b_relevance", MANAGEMENT_ORDER, REL_ORDER), "",
           "Options ticked", "",
           ", ".join(f"{k} {v}" for k, v in Counter(c for r in disc for c in (r["b_icdr"].split(";") if r["b_icdr"] else [])).most_common()) or "-", ""]

    im_done = [i for i in images if i["done"]]
    md += ["## Hard-exudate location on the ETDRS grid", "",
           "The reader's band of the closest HEX to their fovea mark against the band of the closest HEX outline of the "
           "dataset label and of the AI, measured from the same fovea mark with the same scale. These are locations; "
           "they do not diagnose DME. 'within 1 DD' = inner ring or central subfield.", "",
           "Dataset label (rows) against the reader (columns)", "", crosstab(im_done, "hex_zone_label", "hex_zone", ZONE_ORDER, ZONE_ORDER + ["cannot"]), "",
           "AI (rows) against the reader (columns)", "", crosstab(im_done, "hex_zone_ai", "hex_zone", ZONE_ORDER, ZONE_ORDER + ["cannot"]), ""]
    for src in ("label", "ai"):
        pairs = [(i[f"hex_zone_{src}"] in ("inner", "central"), i.get("hex_zone") in ("inner", "central")) for i in im_done
                 if i.get(f"hex_zone_{src}") and i.get("hex_zone") in ZONE_ORDER]
        if pairs:
            both = sum(a and b for a, b in pairs)
            md.append(f"- {src}: reader within 1 DD in {sum(b for _, b in pairs)} images, {src} within 1 DD in {sum(a for a, _ in pairs)}, "
                      f"both {both}; {src} misses {sum(b and not a for a, b in pairs)}, {src} over-calls {sum(a and not b for a, b in pairs)}; "
                      f"central subfield: reader {sum(1 for i in im_done if i.get('hex_zone') == 'central')}, "
                      f"{src} {sum(1 for i in im_done if i.get(f'hex_zone_{src}') == 'central')}")
    md += ["", "The reader's management for the eye by their HEX band", "",
           crosstab(im_done, "hex_zone", "pre_management", ZONE_ORDER + ["cannot"], MANAGEMENT_ORDER), ""]

    marks = [dict(m, reader=i["reader"], case_id=i["case_id"]) for i in images for m in (i.get("marks") or [])]
    md += ["## IRMA and new vessels (optional signs and marks)", "",
           f"Images with IRMA ticked: {sum(1 for i in im_done if 'irma' in str(i.get('pre_other_signs') or ''))}; with NVD / NVE: "
           f"{sum(1 for i in im_done if 'nv' in (str(i.get('pre_other_signs') or '')).split(';'))}; marks: "
           f"{sum(1 for m in marks if m.get('kind') == 'irma')} IRMA, {sum(1 for m in marks if m.get('kind') == 'nv')} NV.", ""]
    if marks and "under_label" in marks[0]:
        md += ["What the four-class outlines put under each mark (within 10 px)", "", crosstab(marks, "kind", "under_label"), "", crosstab(marks, "kind", "under_ai"), ""]

    md += ["## Design-weighted shares (stratified or representative arm; each reader separately, finished images)", "",
           "| reader | domain | quantity | share | 95% interval | regions | images |", "|---|---|---|---|---|---|---|"]
    for r in done:
        r["conseq_ge2"] = None if r["consequence"] is None else ("yes" if r["consequence"] in ("2", "3", "4") else "no")
        r["hex_moved"] = None if r["hex_change"] is None else ("yes" if r["hex_change"] in ("1dd", "central", "unspecified") else "no")
        r["acts"] = None if r["b_relevance"] in (None, "unsure") else ("yes" if r["b_relevance"] in ("followup", "referral") else "no")
    for rd in readers:
        weighted = [r for r in done if r["reader"] == rd and r.get("weight_adj")]
        specs = [("FP", lambda r: r["kind"] == "FP", "blind_verdict", "model_right"),
                 ("FP", lambda r: r["kind"] == "FP", "attribution", "label_wrong"),
                 ("FN", lambda r: r["kind"] == "FN", "blind_verdict", "dataset_right"),
                 ("FN", lambda r: r["kind"] == "FN", "attribution", "ai_wrong"),
                 ("agree", lambda r: r["kind"] == "agree", "blind_verdict", "both_wrong"),
                 ("all discrepancies", lambda r: r["kind"] in DISCREPANCY, "attribution", "both_wrong"),
                 ("all discrepancies", lambda r: r["kind"] in DISCREPANCY, "conseq_ge2", "yes"),
                 ("all discrepancies", lambda r: r["kind"] in DISCREPANCY, "hex_moved", "yes"),
                 ("all discrepancies", lambda r: r["kind"] in DISCREPANCY, "acts", "yes"),
                 ("discrepancies within 1 DD", lambda r: r["kind"] in DISCREPANCY and r["near"] == "within 1 DD", "conseq_ge2", "yes"),
                 ("discrepancies within 1 DD", lambda r: r["kind"] in DISCREPANCY and r["near"] == "within 1 DD", "hex_moved", "yes"),
                 ("discrepancies within 1 DD", lambda r: r["kind"] in DISCREPANCY and r["near"] == "within 1 DD", "acts", "yes"),
                 ("discrepancies beyond 1 DD", lambda r: r["kind"] in DISCREPANCY and r["near"] == "beyond 1 DD", "conseq_ge2", "yes"),
                 ("discrepancies beyond 1 DD", lambda r: r["kind"] in DISCREPANCY and r["near"] == "beyond 1 DD", "acts", "yes")]
        for domain, keep, field, value in specs:
            sh = weighted_share([r for r in weighted if keep(r)], field, value, wkey="weight_adj")
            if sh:
                md.append(f"| {rd} | {domain} | {field} = {value} | {fmt(sh['share'])} | {fmt(sh['lo'])}-{fmt(sh['hi'])} | {sh['n_regions']} | {sh['n_images']} |")

    md += ["", "## Whole image", "", crosstab(im_done, "set", "post_unlabelled", col_order=["none", "few_peripheral", "few_central", "many"]), "",
           crosstab(im_done, "post_label_quality", "post_ai_quality"), ""]
    lm = [i for i in im_done if i.get("fovea_vs_estimate_um") is not None]
    if lm:
        d = [i["fovea_vs_estimate_um"] for i in lm]
        ratios = [i["scale_reader_over_estimate"] for i in lm if i.get("scale_reader_over_estimate")]
        md += [f"Estimated landmarks against the readers' marks: fovea {fmt(float(np.median(d)), 0)} um apart (median; "
               f"max {max(d)}; {sum(x > 500 for x in d)} of {len(d)} beyond 500 um); scale reader / estimate median "
               f"{fmt(float(np.median(ratios)) if ratios else None)}.", ""]

    md += ["## Agreement between readers", ""]
    if len(readers) < 2:
        md.append("One reader so far; agreement needs two or more.")
    else:
        md += ["| field | items | all agree | Fleiss kappa | Gwet AC1 | Cohen kappa by pair (quadratic) |", "|---|---|---|---|---|---|"]
        for field in ["a_group", "a_finding", "blind_verdict", "attribution", "b_label", "b_ai", "consequence", "hex_change", "b_relevance", "zone"]:
            line = agreement_row(field, {rd: {r["inst_id"]: r[field] for r in done if r["reader"] == rd} for rd in readers}, ORDINAL.get(field))
            if line:
                md.append(line)
        for field in ["pre_icdr", "hex_zone", "pre_management", "pre_quality", "post_unlabelled", "post_label_quality", "post_ai_quality"]:
            line = agreement_row(field, {rd: {i["case_id"]: i.get(field) for i in im_done if i["reader"] == rd} for rd in readers}, ORDINAL.get(field))
            if line:
                md.append(line)
        fov = defaultdict(dict)
        for rev in reviews:
            for cid, cs in rev["cases"].items():
                f = cs["pre"]["answers"].get("pre_fovea")
                if f and f.get("x") is not None:
                    fov[cid][rev["reader_id"]] = f
        gaps = []
        for cid, byr in fov.items():
            geo_case = next((i for i in images if i["case_id"] == cid and i.get("um_per_px_reader")), None)
            if geo_case and len(byr) >= 2:
                for r1, r2 in itertools.combinations(sorted(byr), 2):
                    gaps.append(np.hypot(byr[r1]["x"] - byr[r2]["x"], byr[r1]["y"] - byr[r2]["y"]) * geo_case["um_per_px_reader"])
        if gaps:
            md += ["", f"Fovea marks between readers: median {fmt(float(np.median(gaps)), 0)} um apart (pairs {len(gaps)})."]

    for field, title in [("b_label_issue", "Dataset-label issues (questionnaire 2)"), ("b_model_issue", "AI issues (questionnaire 2)")]:
        cnt = Counter(code for r in done for code in (r[field].split(";") if r[field] else []))
        if cnt:
            md += ["", f"## {title}", "", ", ".join(f"{k} {v}" for k, v in cnt.most_common())]

    other = [r for r in done if r["a_finding"] in ("other_lesion", "indeterminate")]
    md += ["", "## Checks", "",
           f"- median active time per region: {fmt(float(np.median([r['active_s'] for r in done])) if done else None, 0)} s",
           f"- step A 'other' or 'cannot determine': {len(other)} of {len(done)} ({fmt(len(other) / len(done) if done else None)}; target < 0.15)",
           f"- regions with a comment: {sum(1 for r in done if r['b_comment'])}", ""]
    md += ["## Reader feedback on the questions", ""]
    for rev in reviews:
        for k, v in rev.get("debrief", {}).get("answers", {}).items():
            if isinstance(v, str):
                md.append(f"- {rev['reader_id']} {k}: {v}")
            elif isinstance(v, list):
                md.append(f"- {rev['reader_id']} {k}: {', '.join(v)}")
    md += ["", "## Comments", ""]
    for r in done:
        if r["b_comment"]:
            md.append(f"- {r['reader']} {r['inst_id']} ({r['label']}, {r['kind']}, {r['zone'] or 'no zone'}, {r['attribution']}): {r['b_comment']}")
    for i in images:
        if i.get("post_comment"):
            md.append(f"- {i['reader']} {i['case_id']} (image): {i['post_comment']}")
    (out / "report.md").write_text("\n".join(md) + "\n")
    print(f"[analyze] {len(regions)} region readings, {len(images)} image readings -> {out}")


if __name__ == "__main__":
    main()
