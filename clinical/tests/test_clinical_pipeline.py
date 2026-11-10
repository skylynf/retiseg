"""The clinical review pipeline on a synthetic prepared source (docs/clinical/28, 32): landmarks -> mine -> sample ->
pack -> analyse.

No dataset, weights or torch needed. Checks the parts whose failure would silently bias the study: the outline
tracer encloses exactly the mask, the cluster kinds follow the stated rules, ETDRS zones and the stratified design
weights are what the design says, the pack carries none of the hidden metadata a reader must not see, crop windows
of one image never overlap, and the analysis reads exports of questionnaires 2 and 3.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))

import analyze_reviews  # noqa: E402
import build_pack  # noqa: E402
import contours  # noqa: E402
import locate_landmarks  # noqa: E402
import mine_discrepancies  # noqa: E402
import sample_review  # noqa: E402

CLASSES = ("MA", "HE", "EX", "SE")


def test_outline_area_equals_mask():
    gen = np.random.default_rng(0)
    from scipy import ndimage
    for _ in range(50):
        m = ndimage.binary_opening(gen.random((30, 40)) < 0.45) | (gen.random((30, 40)) < 0.05)
        loops = contours.mask_loops(m, eps=0)
        assert abs(sum(contours.polygon_area(lp) for lp in loops) - m.sum()) < 1e-9


def test_corner_touching_pixels_get_separate_loops():
    m = np.zeros((4, 4), bool)
    m[1, 1] = m[2, 2] = True
    loops = contours.mask_loops(m, eps=0)
    assert len(loops) == 2 and all(abs(contours.polygon_area(lp) - 1) < 1e-9 for lp in loops)


def _disk(shape, cx, cy, r):
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r


def _write_mask(path, mask):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask * 255).astype(np.uint8)).save(path)


def _write_prob(path, prob):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.rint(prob * 65535).astype(np.uint16)).save(path)


@pytest.fixture()
def synthetic(tmp_path):
    """Two 600x500 'fundus' images. Image a: an MA the model misses (FN), an MA only the model sees (FP), an HE both
    see (agree), an EX the model sees as HE (confusion). Image b: an agreed EX, a model-only SE."""
    H, W = 500, 600
    data = tmp_path / "prepared" / "SYN"
    pred = tmp_path / "runs" / "SYN_model" / "prob_test"
    fov = _disk((H, W), 300, 250, 240)
    spec = {
        "a": {"gt": {"MA": [(150, 150, 4)], "HE": [(300, 200, 12)], "EX": [(420, 300, 10)]},
              "pr": {"MA": [(220, 330, 4)], "HE": [(301, 201, 12), (420, 300, 10)]}},
        "b": {"gt": {"EX": [(250, 250, 15)]}, "pr": {"EX": [(251, 250, 15)], "SE": [(380, 180, 20)]}},
    }
    for iid, s in spec.items():
        rgb = np.zeros((H, W, 3), np.uint8)
        rgb[fov] = (180, 80, 30)
        (data / "images").mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb).save(data / "images" / f"{iid}.png")
        _write_mask(data / "fov" / f"{iid}.png", fov)
        for c in CLASSES:
            g = np.zeros((H, W), bool)
            for cx, cy, r in s["gt"].get(c, []):
                g |= _disk((H, W), cx, cy, r)
            if g.any():
                _write_mask(data / "masks" / c / f"{iid}.png", g)
            p = np.full((H, W), 0.01)
            for cx, cy, r in s["pr"].get(c, []):
                p[_disk((H, W), cx, cy, r)] = 0.9
            _write_prob(pred / c / f"{iid}.png", p)
    (data / "splits.json").write_text(json.dumps({"train": [], "val": [], "test": ["a", "b"]}))
    (data / "meta.json").write_text(json.dumps({"name": "SYN", "classes": list(CLASSES), "granularity": "fine"}))
    return tmp_path


def test_pipeline_end_to_end(synthetic, monkeypatch):
    root = synthetic
    monkeypatch.chdir(root)
    mine_out = root / "clinical" / "work" / "mine" / "SYN"
    # landmarks given as 45 um / px (as if 4,500 um over 100 px); C <= 11.1 px, 1 DD = 33.3 px, grid 66.7 px.
    # image a: fovea (300, 250); image b: fovea (255, 250), inside its agreed EX disc
    lms = {"a": {"fovea": [300.0, 250.0], "disc": [400.0, 250.0], "um_per_px": 45.0, "source": "manual"},
           "b": {"fovea": [255.0, 250.0], "disc": [355.0, 250.0], "um_per_px": 45.0, "source": "manual"}}
    (root / "landmarks.json").write_text(json.dumps({"set": "SYN", "landmarks": lms}))
    mine_discrepancies.main(["--set", "SYN", "--data", "prepared/SYN", "--pred", "runs/SYN_model/prob_test",
                             "--thresholds", "0.5", "--model", "synthetic", "--landmarks", "landmarks.json", "--out", str(mine_out)])
    rows = [json.loads(x) for x in (mine_out / "candidates.jsonl").read_text().splitlines()]
    kinds = {(r["image_id"], r["cls"], r["kind"]) for r in rows}
    assert ("a", "MA", "FN") in kinds and ("a", "MA", "FP") in kinds and ("a", "HE", "agree") in kinds
    assert ("b", "EX", "agree") in kinds and ("b", "SE", "FP") in kinds
    conf = [r for r in rows if r["image_id"] == "a" and r["cls"] == "EX"]
    assert conf and conf[0]["kind"] == "FN" and conf[0]["subtype"] == "confusion:EX->HE"
    by = {(r["image_id"], r["cls"], r["kind"]): r for r in rows}
    # agreed HE disc at (300, 200) r 12: closest pixel 38 px = 1,710 um from the fovea -> outer ring, superior
    he = by[("a", "HE", "agree")]
    assert he["etdrs_zone"] == "outer" and he["etdrs_sub"] == "S2" and abs(he["dist_min_um"] - 38 * 45) <= 90
    # the agreed EX of image b covers the fovea -> central subfield, distance 0
    ex = by[("b", "EX", "agree")]
    assert ex["etdrs_zone"] == "C" and ex["dist_min_um"] == 0 and ex["etdrs_sub"] == "C"
    images = {r["image_id"]: r for r in map(json.loads, (mine_out / "images.jsonl").read_text().splitlines())}
    assert images["b"]["ex_px"]["c"] > 0 and images["b"]["ex_px"]["nearest_label_um"] == 0
    assert images["a"]["ex_px"]["c"] == 0 and images["a"]["landmark"]["um_per_px"] == 45.0
    assert all(r["elongation"] < 1.5 for r in rows)          # discs are round

    design = {
        "name": "syn01", "seed": 1,
        "sets": [{"set": "SYN", "candidates": "clinical/work/mine/SYN", "reference_grade_scale": None}],
        "images": {"eligibility": {}, "strata": [{"set": "SYN", "n": 2}]},
        "representative": {"per_image": [{"kind": "FN", "n": 1}, {"kind": "FP", "n": 1}, {"kind": "agree", "n": 1}]},
        "enriched": [{"name": "confusion", "n": 1, "where": {"subtype_prefix": "confusion"}}],
        "geometry": {"crop_min": 96, "crop_margin": 0.6, "marker_min_radius": 10, "min_gap": 0},
        "max_per_image": 6,
    }
    (root / "design.yaml").write_text(json.dumps(design))
    sample_review.main(["--design", "design.yaml", "--out", "work/syn01"])
    sel = json.loads((root / "work" / "syn01" / "selection.json").read_text())
    assert sel["report"]["n_cases"] == 2
    for cid in {x["case_id"] for x in sel["instances"]}:
        boxes = [x["crop"] for x in sel["instances"] if x["case_id"] == cid]
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                assert not sample_review.overlaps(boxes[i], boxes[j])
    rep = [x for x in sel["instances"] if x["arm"] == "representative"]
    assert rep and all(x["weight"] and x["weight"] >= 1 for x in rep)

    # the formal design: image strata by macular HEX, region strata by ETDRS zone, weights N_ih / (pi_image n_ih),
    # drawn once per image stratum without rejection (reading order blind_first), two reading parts
    design2 = {
        "name": "syn01", "seed": 2, "reading_order": "blind_first", "parts": [1, 1],
        "sets": [{"set": "SYN", "candidates": "clinical/work/mine/SYN", "reference_grade_scale": None}],
        "images": {"eligibility": {"discrepancy": 1}, "strata": [
            {"name": "macular_hex", "set": "SYN", "n": 1, "has_regions": {"where": {"cls": "EX", "etdrs_zone": ["C", "inner"]}, "n": 1}},
            {"name": "rest", "set": "SYN", "n": 1}]},
        "stratified": {"strata": [
            {"name": "hex_macula", "n": 1, "where": {"cls": "EX", "etdrs_zone": ["C", "inner"]}},
            {"name": "other_discrepancy", "n": 1, "where": {"kind": ["FN", "FP", "extent"]}},
            {"name": "agree", "n": 1, "where": {"kind": "agree"}}]},
        "enriched": [],
        "geometry": {"crop_min": 64, "crop_margin": 0.3, "marker_min_radius": 10, "min_gap": 0},
        "max_per_image": 6,
    }
    (root / "design2.yaml").write_text(json.dumps(design2))
    (root / "design3.yaml").write_text(json.dumps({k: v for k, v in design2.items() if k not in ("reading_order", "parts")}))
    with pytest.raises(SystemExit):                       # the stratified arm needs exact inclusion probabilities
        sample_review.main(["--design", "design3.yaml", "--out", "work/syn03"])
    sets_bad = [dict(e, inputs={"candidates_sha256": "0" * 64}) for e in design2["sets"]]
    (root / "design5.yaml").write_text(json.dumps(dict(design2, sets=sets_bad)))
    with pytest.raises(SystemExit):                       # mining outputs differ from the frozen hashes
        sample_review.main(["--design", "design5.yaml", "--out", "work/syn05", "--dry-run"])
    import common
    good = common.sha256_file(root / "clinical/work/mine/SYN/candidates.jsonl")
    (root / "design6.yaml").write_text(json.dumps(dict(design2, sets=[dict(e, inputs={"candidates_sha256": good})
                                                                       for e in design2["sets"]])))
    sample_review.main(["--design", "design6.yaml", "--out", "work/syn06", "--dry-run"])   # matching hashes pass
    (root / "design4.yaml").write_text(json.dumps(dict(design2, status="proposed")))
    sample_review.main(["--design", "design4.yaml", "--out", "work/syn04", "--dry-run"])   # a dry run works on any status
    with pytest.raises(SystemExit):                       # a proposed design does not draw
        sample_review.main(["--design", "design4.yaml", "--out", "work/syn04"])
    sample_review.main(["--design", "design2.yaml", "--out", "work/syn02", "--dry-run"])
    assert not (root / "work" / "syn02").exists()           # a dry run draws nothing
    sample_review.main(["--design", "design2.yaml", "--out", "work/syn02"])
    sel2 = json.loads((root / "work" / "syn02" / "selection.json").read_text())
    strata = {s["stratum"]: s for s in sel2["report"]["strata"]}
    assert strata["macular_hex"]["population"] == 1 and strata["rest"]["population"] == 1   # b has central HEX, a not
    case_b = next(c for c in sel2["cases"] if c["image_id"] == "b")
    assert case_b["landmark"]["um_per_px"] == 45.0 and case_b["region_strata"]["hex_macula"]["N"] == 1
    for x in sel2["instances"]:
        assert x["arm"] == "stratified"
        assert x["weight"] == pytest.approx(1.0 / (x["pi_image"] * x["pi_within"]))
    for c in sel2["cases"]:                                  # every stratum drawn in full: min(n, N), no rejection
        for h in design2["stratified"]["strata"]:
            got = sum(1 for x in sel2["instances"] if x["case_id"] == c["case_id"] and x["stratum"] == h["name"])
            assert got == min(h["n"], c["region_strata"][h["name"]]["N"])
    assert sel2["report"]["redraws"] == 0 and sel2["report"]["reading_order"] == "blind_first"
    assert sorted(c["part"] for c in sel2["cases"]) == [1, 2]
    assert sum(p["part1"] for p in sel2["report"]["parts"]) == 1
    hex_b = [x for x in sel2["instances"] if x["image_id"] == "b" and x["stratum"] == "hex_macula"]
    assert len(hex_b) == 1 and hex_b[0]["weight"] == pytest.approx(1.0)

    build_pack.main(["--selection", "work/syn01/selection.json", "--out", "packs/syn01", "--root", "."])
    pack = json.loads((root / "packs" / "syn01" / "pack.json").read_text())
    text = (root / "packs" / "syn01" / "pack.json").read_text()
    for hidden in ('"kind"', '"arm"', '"weight"', '"image_id"', '"subtype"', '"prob_max"', "SYN_model", "prepared/SYN"):
        assert hidden not in text, f"pack leaks {hidden}"
    assert (root / "packs" / "syn01" / "index.html").is_file() and (root / "packs" / "syn01" / "pack.js").is_file()
    assert {r["id"] for c in pack["cases"] for r in c["regions"]} == {x["inst_id"] for x in sel["instances"]}
    readme = (root / "packs" / "syn01" / "READ_ME_FIRST.txt").read_text()
    assert "Sources and licences" in readme and "SYN: see the study team" in readme
    assert build_pack.IMAGE_NOTE in readme and "resized copies" not in readme
    for img in (root / "packs" / "syn01" / "images").glob("*.jpg"):     # no camera metadata in a pack image
        assert _jpeg_markers(img.read_bytes()) & {0xE1, 0xFE} == set()
    restricted = json.loads((root / "work" / "syn01" / "selection.json").read_text())
    for c in restricted["cases"]:
        c["set"] = "MAPLES-DR"
    (root / "work" / "maples_sel.json").write_text(json.dumps(restricted))
    with pytest.raises(SystemExit):                       # MESSIDOR images do not go into a reader pack
        build_pack.main(["--selection", "work/maples_sel.json", "--out", "packs/maples", "--root", "."])
    assert not (root / "packs" / "maples").exists()

    # readers see the clinical labels (docs/clinical/31), the pack keeps the benchmark's class codes as keys
    assert {k: v["label"] for k, v in pack["classes"].items()} == {"MA": "MA", "HE": "HG", "EX": "HEX", "SE": "CWS"}
    # a reader export: every region read as the lesion the model reported, the label judged wrong
    code = {"MA": "ma", "HE": "hg", "EX": "hex", "SE": "cws"}

    def export_of(reader, b_answers, pre_extra=None, only=None):
        export = {"format": "retiseg-clinical-review/1", "pack_id": "syn01", "reader_id": reader, "cases": {},
                  "schema_version": "3.1.0" if "b_label" in b_answers else "2.0.0-trial", "debrief": {"answers": {"d_useful": "all"}}}
        for c in pack["cases"][:only]:
            regs = {}
            fx, fy = lms[next(k["image_id"] for k in sel["cases"] if k["case_id"] == c["id"])]["fovea"]
            for r in c["regions"]:
                x = next(i for i in sel["instances"] if i["inst_id"] == r["id"])
                regs[r["id"]] = {"a": {"answers": {"a_finding": code[x["cls"]], "a_confidence": "certain"}, "t": {"active_ms": 20000}},
                                 "b": {"answers": dict(b_answers), "t": {"active_ms": 25000, "completed": "x"}}, "revealed_at": "x"}
            export["cases"][c["id"]] = {"practice": False, "done": True, "regions": regs, "display_size": [c["width"], c["height"]],
                                        "pre": {"answers": {"pre_icdr": "2", "pre_fovea": {"x": fx, "y": fy}, "pre_disc": {"x": fx + 100, "y": fy},
                                                            **(pre_extra or {})}, "t": {"active_ms": 30000, "completed": "x"}},
                                        "post": {"answers": {"post_unlabelled": "few_peripheral"}, "t": {"active_ms": 10000, "completed": "x"}}}
        return export

    # R1: questionnaire 2 (the trial of 2026-11-02); R2: questionnaire 3.1, both outlines wrong, central HEX, one NV mark;
    # R3: like R2 but only the first image finished
    (root / "R1.json").write_text(json.dumps(export_of("R1", {"b_cause": "label_error", "b_label_issue": ["omission"],
                                                              "b_icdr": ["none"], "b_relevance": "no"}, {"pre_dme": "central"})))
    b2 = {"b_label": "missed", "b_ai": "extent", "b_icdr": ["g12", "hex_c"], "b_relevance": "referral"}
    pre2 = {"pre_hex_zone": "central", "pre_management": "refer", "pre_other_signs": ["nv"], "pre_marks": [{"x": 301, "y": 200, "kind": "nv"}]}
    (root / "R2.json").write_text(json.dumps(export_of("R2", b2, pre2)))
    (root / "R3.json").write_text(json.dumps(export_of("R3", b2, pre2, only=1)))
    analyze_reviews.main(["--selection", "work/syn01/selection.json", "--pack", "packs/syn01", "--reviews", "R1.json", "R2.json", "R3.json",
                          "--out", "work/syn01/analysis"])
    report = (root / "work" / "syn01" / "analysis" / "report.md").read_text()
    assert "Blinded verdict" in report and "R1" in report and "ETDRS location" in report and "Hard-exudate location" in report
    import csv
    rows = list(csv.DictReader(open(root / "work" / "syn01" / "analysis" / "regions_long.csv")))
    assert rows and all(r["blind_verdict"] in ("model_right", "dataset_right", "both_right") for r in rows)
    assert {r["attribution"] for r in rows if r["reader"] == "R1"} == {"label_wrong"}
    assert {r["attribution"] for r in rows if r["reader"] == "R2"} == {"both_wrong"}
    assert {r["consequence"] for r in rows if r["reader"] == "R2"} == {"2"} and {r["consequence"] for r in rows if r["reader"] == "R1"} == {"0"}
    assert {r["hex_change"] for r in rows if r["reader"] == "R2"} == {"central"} and {r["reader_management"] for r in rows if r["reader"] == "R2"} == {"refer"}
    # the reader's fovea and disc marks give 47 um / px (4,700 um over 100 px); every region gets a zone
    assert all(r["zone"] in ("C", "inner", "outer", "outside") for r in rows)
    # completion: R3 finished one of the two images of the stratum, so its weights are doubled
    w2 = {r["inst_id"]: float(r["weight_adj"]) for r in rows if r["reader"] == "R2" and r["weight_adj"]}
    w3 = {r["inst_id"]: float(r["weight_adj"]) for r in rows if r["reader"] == "R3" and r["weight_adj"]}
    assert w3 and all(w3[k] == pytest.approx(2 * w2[k]) for k in w3)
    imgs = list(csv.DictReader(open(root / "work" / "syn01" / "analysis" / "images_long.csv")))
    b_img = next(i for i in imgs if i["image_id"] == "b" and i["reader"] == "R2")
    assert b_img["hex_zone_label"] == "central" and b_img["hex_zone_ai"] == "central" and b_img["n_nv_marks"] == "1"
    assert b_img["hex_zone"] == "central"
    assert not any(i["hex_zone"] for i in imgs if i["reader"] == "R1")     # questionnaire 2's DME answer is another question
    a_img = next(i for i in imgs if i["image_id"] == "a" and i["reader"] == "R2")
    assert "HG" in a_img["marks"]          # the NV mark at (301, 200) lies inside the labelled and predicted HE disc


def test_blind_verdict_rules():
    bv = analyze_reviews.blind_verdict
    assert bv("FP", "MA", "ma") == "model_right"
    assert bv("FP", "MA", "vessel") == "dataset_right"
    assert bv("FN", "HE", "reflex") == "model_right"
    assert bv("FN", "HE", "hg") == "dataset_right"
    assert bv("agree", "EX", "drusen") == "other_class"
    assert bv("agree", "EX", "artifact_light") == "both_wrong"
    assert bv("FN", "SE", "indeterminate") == "unresolved"
    assert bv("FP", "SE", "cws") == "model_right"


def test_etdrs_location():
    key = {"set": "DDR", "image_size": [1000, 800], "landmark": {"fovea": [500, 400], "disc": [800, 400], "um_per_px": 15.0}}
    state = {"pre": {"answers": {"pre_fovea": {"x": 500, "y": 400}, "pre_disc": {"x": 750, "y": 400}}}, "display_size": [1000, 800]}
    geo = analyze_reviews.reader_geometry(key, state)
    assert geo["source"] == "reader" and geo["umpx"] == pytest.approx(18.8)          # 4,700 um / 250 px
    loc = analyze_reviews.region_location
    assert loc(geo, {"bbox": [495, 395, 505, 405], "centroid": [500, 400]})[2] == "C"
    assert loc(geo, {"bbox": [540, 395, 560, 405], "centroid": [550, 400]}) == (752, 940, "inner")  # 40 px to the box
    assert loc(geo, {"bbox": [600, 395, 610, 405], "centroid": [605, 400]})[2] == "outer"
    assert loc(geo, {"bbox": [700, 395, 710, 405], "centroid": [705, 400]})[2] == "outside"
    # no disc mark: the pack's estimate gives the scale; no fovea mark: no location
    state["pre"]["answers"]["pre_disc"] = {"not_visible": True}
    assert analyze_reviews.reader_geometry(key, state)["source"] == "estimate"
    assert analyze_reviews.reader_geometry(key, {"pre": {"answers": {}}}) is None
    # a pack shown at half size: clicks in display pixels, the key in native pixels
    half = {"pre": {"answers": {"pre_fovea": {"x": 250, "y": 200}, "pre_disc": {"x": 375, "y": 200}}}, "display_size": [500, 400]}
    g2 = analyze_reviews.reader_geometry(key, half)
    assert loc(g2, {"bbox": [540, 395, 560, 405], "centroid": [550, 400]})[0] == 752
    assert analyze_reviews.zone_of(500) == "C" and analyze_reviews.zone_of(500.1) == "inner" and analyze_reviews.zone_of(3000.1) == "outside"


def test_attribution_and_consequence():
    att = analyze_reviews.attribution
    assert att({"b_label": "correct", "b_ai": "correct"}) == "both_right"
    assert att({"b_label": "missed", "b_ai": "correct"}) == "label_wrong"
    assert att({"b_label": "correct", "b_ai": "false"}) == "ai_wrong"
    assert att({"b_label": "extent", "b_ai": "wrong_type"}) == "both_wrong"
    assert att({"b_label": "unsure", "b_ai": "correct", "b_unsure_why": "definition"}) == "undecided_definition"
    assert att({"b_label": "correct"}) is None
    assert att({"b_cause": "model_error"}) == "ai_wrong" and att({"b_cause": "ambiguous"}) == "undecided_definition"
    lvl, hx = analyze_reviews.consequence_level, analyze_reviews.hex_change
    assert lvl(["g01", "g12"], "ai_wrong") == 2 and lvl(["g23", "hex_c"], "both_wrong") == 3 and lvl(["none"], "label_wrong") == 0
    assert lvl(["g02"], "ai_wrong") == 2 and lvl(["g12", "g34"], "label_wrong") == 4
    assert lvl(["hex_c"], "ai_wrong") == 0           # a HEX band is a location, not an ICDR threshold
    assert lvl(None, "both_right") == 0 and lvl([], "ai_wrong") is None and lvl(["dme"], "ai_wrong") == 0
    assert hx(["g12", "hex_1dd"], "ai_wrong") == "1dd" and hx(["hex_1dd", "hex_c"], "ai_wrong") == "central"
    assert hx(["dme_cs"], "ai_wrong") == "central" and hx(["dme"], "ai_wrong") == "unspecified"     # questionnaires 3.0, 2
    assert hx(["none"], "label_wrong") == "none" and hx(None, "both_right") == "none" and hx([], "ai_wrong") is None


def test_outline_distance_and_implied_zone():
    square = [[10, 10, 20, 10, 20, 20, 10, 20]]
    near = analyze_reviews.nearest_outline_px
    assert near(square, 15, 15) == 0.0 and near(square, 25, 15) == pytest.approx(5.0) and near([], 0, 0) is None
    geo = {"fx": 0.0, "fy": 0.0, "umpx": 100.0, "s": 1.0}
    assert analyze_reviews.implied_zone(geo, square) == ("inner", 1414)    # closest corner 14.1 px x 100 um
    assert analyze_reviews.implied_zone(dict(geo, umpx=30.0), square) == ("central", 424)
    assert analyze_reviews.implied_zone(dict(geo, umpx=500.0), square)[0] == "none"
    assert analyze_reviews.implied_zone(geo, None) == ("none", None)


def test_design_conditions():
    m = sample_review.matches
    row = {"cls": "HE", "kind": "FP", "grade": 4, "elongation": 4.2, "dist_disc_um": 2500, "ext_overlap": {"NV": 12},
           "ex_px": {"dd1": 30}}
    assert m(row, {"cls": ["HE", "MA"], "kind": ["FN", "FP", "extent"]})
    assert m(row, {"any": [{"elongation_ge": 3.0}, {"dist_disc_um_le": 1500}]})
    assert not m(row, {"not": {"grade": [3, 4]}})
    assert m(row, {"ext_any": ["NV", "IRMA"]}) and not m(row, {"ext_any": "IRMA"})
    assert m(row, {"ex_px.dd1_ge": 1}) and not m(row, {"ex_px.c_ge": 1})        # absent values fail a bound
    assert m(dict(row, grade=None), {"not": {"grade": [0, 1, 2]}})


def test_shortfall_moves_to_the_named_stratum():
    zero = {k: 0 for k in ("FN", "FP", "extent", "agree", "confusion")}
    row = lambda d: {"counts": {c: dict(zero, FN=d) for c in sample_review.CLASSES}}
    images = {("S", f"i{k}"): row(1) for k in range(10)}
    images[("S", "i9")] = row(0)                                  # not eligible: no discrepancy
    member = {("S", f"i{k}"): (0 if k < 2 else 1 if k < 4 else 2) for k in range(10)}
    strata = [{"name": "a", "n": 5}, {"name": "b", "n": 1}, {"name": "rest", "n": 2}]
    design = {"images": {"eligibility": {"discrepancy": 1}, "strata": strata}}
    plan = sample_review.image_quotas(strata, member, images, design)
    assert [q["n"] for q in plan] == [2, 1, 2]                    # without the rule the batch shrinks to 5
    design["images"]["shortfall_to"] = "rest"
    plan = sample_review.image_quotas(strata, member, images, design)
    assert [q["n"] for q in plan] == [2, 1, 5]                    # a is 3 short: rest gets 2 + 3 (of 5 eligible)
    strata[2]["n"] = 4
    plan = sample_review.image_quotas(strata, member, images, design)
    assert plan[2]["n"] == 5                                      # capped at the eligible images
    design["images"]["shortfall_to"] = "nope"
    with pytest.raises(SystemExit):
        sample_review.image_quotas(strata, member, images, design)


def test_grades_file_with_other_sets(tmp_path):
    import common

    multi = tmp_path / "grades.json"
    multi.write_text(json.dumps({"DDR": {"a": 2, "b": None}, "MAPLES-DR": {"c": 4}}))
    assert common.load_grades(str(multi), "DDR") == {"a": 2, "b": None}
    assert common.load_grades(str(multi), "IDRiD") == {}        # 2026-11-07: used to read another set's table
    flat = tmp_path / "flat.json"
    flat.write_text(json.dumps({"x": 1, "y": 3}))
    assert common.load_grades(str(flat), "IDRiD") == {"x": 1, "y": 3}


def test_landmarks_on_a_synthetic_fundus():
    H, W = 500, 600
    yy, xx = np.mgrid[:H, :W]
    fov = (xx - 300) ** 2 + (yy - 250) ** 2 <= 240 ** 2
    lum = np.full((H, W), 120.0)
    lum += 110 * np.exp(-((xx - 440) ** 2 + (yy - 250) ** 2) / (2 * 16.0 ** 2))       # disc
    lum -= 45 * np.exp(-((xx - 252) ** 2 + (yy - 258) ** 2) / (2 * 22.0 ** 2))        # fovea, 0.39 FOV diameters away
    rgb = np.stack([np.clip(lum * 1.4, 0, 255), np.clip(lum * 0.6, 0, 255), lum * 0.2], -1).astype(np.uint8)
    rgb[~fov] = 0
    lm = locate_landmarks.estimate(Image.fromarray(rgb), fov, 45.0)
    assert np.hypot(lm["disc"][0] - 440, lm["disc"][1] - 250) < 10
    assert np.hypot(lm["fovea"][0] - 252, lm["fovea"][1] - 258) < 25
    assert lm["um_per_px"] == pytest.approx(4700 / np.hypot(lm["fovea"][0] - lm["disc"][0], lm["fovea"][1] - lm["disc"][1]), rel=1e-3)


def _jpeg_markers(data):
    """Marker bytes of the segments before the first scan, and whether anything follows the primary image."""
    found, i = set(), 2
    while data[i + 1] != 0xDA:
        found.add(data[i + 1])
        i += 2 + int.from_bytes(data[i + 2:i + 4], "big")
    return found


def _camera_jpeg(path):
    """A JPEG as some fundus cameras write it: Exif, a comment with an exam id, an MPF-style preview after the EOI."""
    rng = np.random.default_rng(0)
    im = Image.fromarray(rng.integers(0, 255, (48, 64, 3), dtype=np.uint8))
    exif = Image.Exif()
    exif[0x010F] = "CAMERA MAKER"
    exif[0x0132] = "2001:02:03 04:05:06"
    buf = __import__("io").BytesIO()
    im.save(buf, format="JPEG", quality=90, exif=exif, comment=b"CAMTYPE=99;ID=12345678;DATE=01/02/03;SER=0042")
    preview = __import__("io").BytesIO()
    im.resize((32, 24)).save(preview, format="JPEG")
    data = buf.getvalue() + b"\x00" * 66 + preview.getvalue()
    path.write_bytes(data)
    return data


def test_clean_jpeg_drops_camera_metadata(tmp_path):
    import io
    data = _camera_jpeg(tmp_path / "cam.jpg")
    assert {0xE1, 0xFE} <= _jpeg_markers(data)
    clean = build_pack.clean_jpeg(data)
    assert _jpeg_markers(clean) & {0xE1, 0xFE} == set()
    assert b"12345678" not in clean and b"CAMERA MAKER" not in clean
    assert clean.endswith(b"\xff\xd9") and clean.count(b"\xff\xd8") == 1           # no preview after the image
    a = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
    b = np.asarray(Image.open(io.BytesIO(clean)).convert("RGB"))
    assert np.array_equal(a, b)                                                     # same pixels
    assert build_pack.clean_jpeg(clean) == clean


def test_clean_existing_pack(tmp_path):
    import common
    pack = tmp_path / "batchX_part1"
    (pack / "images").mkdir(parents=True)
    _camera_jpeg(pack / "images" / "c01.jpg")
    (pack / "pack.json").write_text(json.dumps({"pack_id": "batchX+part1", "cases": []}))
    (pack / "READ_ME_FIRST.txt").write_text("Sources\n" + build_pack.OLD_IMAGE_NOTE + "\n")
    rep = build_pack.main(["--clean-existing", str(pack), "--zip"])
    assert rep["images_cleaned"] == ["c01.jpg"] and rep["readme_updated"]
    assert b"12345678" not in (pack / "images" / "c01.jpg").read_bytes()
    assert build_pack.IMAGE_NOTE in (pack / "READ_ME_FIRST.txt").read_text()
    sums = dict(reversed(line.split("  ")) for line in (pack / "SHA256SUMS.txt").read_text().splitlines())
    assert sums["images/c01.jpg"] == common.sha256_file(pack / "images" / "c01.jpg")
    assert (tmp_path / "batchX_part1.zip").is_file()
    rep2 = build_pack.main(["--clean-existing", str(pack)])                           # a second run changes nothing
    assert rep2["images_cleaned"] == [] and not rep2["readme_updated"]


def test_kappa_sanity():
    a = ["x", "y", "x", "y", "x", "x"]
    assert analyze_reviews.cohen_kappa(a, a) == pytest.approx(1.0)
    table = [[2, 0], [0, 2], [2, 0], [0, 2]]
    assert analyze_reviews.fleiss_kappa(table) == pytest.approx(1.0)
    assert analyze_reviews.gwet_ac1(table) == pytest.approx(1.0)


def test_schema_is_consistent():
    schema = json.loads((HERE.parent / "schema" / "review_schema.json").read_text())
    ids = set()
    for step in schema["steps"].values():
        qids = {q["id"] for q in step["questions"]}
        for q in step["questions"]:
            assert q["id"] not in ids, f"duplicate question id {q['id']}"
            ids.add(q["id"])
            if q["type"] in ("single", "multi"):
                codes = [o["code"] for o in q["options"]]
                assert len(codes) == len(set(codes)), q["id"]
            cond = q.get("show_if")
            for c in ([cond] if cond and "q" in cond else (cond or {}).get("any", []) + (cond or {}).get("all", [])):
                assert c["q"] in qids, f"{q['id']} depends on a question of another step"
    finding = next(q for q in schema["steps"]["region_a"]["questions"] if q["id"] == "a_finding")
    assert set(o["code"] for o in finding["options"]) == set(analyze_reviews.FINDING_GROUP)
    icdr = next(q for q in schema["steps"]["region_b"]["questions"] if q["id"] == "b_icdr")
    assert [o["code"] for o in icdr["options"] if o.get("exclusive")] == ["none"]
    assert {o["code"] for o in icdr["options"]} == set(analyze_reviews.ICDR_ORDER)
    assert set(analyze_reviews.ICDR_ORDER) <= set(analyze_reviews.CONSEQUENCE_LEVEL) | set(analyze_reviews.HEX_CHANGE)
    hz = next(q for q in schema["steps"]["image_pre"]["questions"] if q["id"] == "pre_hex_zone")
    assert [o["code"] for o in hz["options"] if o["code"] != "cannot"] == analyze_reviews.ZONE_ORDER
    mg = next(q for q in schema["steps"]["image_pre"]["questions"] if q["id"] == "pre_management")
    assert [o["code"] for o in mg["options"]] == analyze_reviews.MANAGEMENT_ORDER
    text = json.dumps(schema)
    assert "referable DME" not in text and "DME:" not in text      # HEX bands are locations, never called DME
    for qid in ("b_label", "b_ai"):
        q = next(q for q in schema["steps"]["region_b"]["questions"] if q["id"] == qid)
        assert {o["code"] for o in q["options"]} == {"correct", "extent", "wrong_type", "false", "missed", "unsure"}
    assert schema["etdrs"] == {"fovea_disc_um": 4700, "central_um": 500, "inner_um": 1500, "outer_um": 3000, "dd_um": 1500}
    labels = " ".join(o["label"] + " " + o.get("help", "") for st in schema["steps"].values() for q in st["questions"] for o in q.get("options", []))
    assert " HE " not in f" {labels} " and " EX " not in f" {labels} " and " SE " not in f" {labels} ", "readers must see HG / HEX / CWS"
