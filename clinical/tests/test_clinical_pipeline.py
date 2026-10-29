"""The clinical review pipeline on a synthetic prepared source (docs/clinical/28): mine -> sample -> pack -> analyse.

No dataset, weights or torch needed. Checks the parts whose failure would silently bias the study: the outline
tracer encloses exactly the mask, the cluster kinds follow the stated rules, the pack carries none of the hidden
metadata a reader must not see, crop windows of one image never overlap, and the analysis reads an export.
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
    mine_discrepancies.main(["--set", "SYN", "--data", "prepared/SYN", "--pred", "runs/SYN_model/prob_test",
                             "--thresholds", "0.5", "--model", "synthetic", "--out", str(mine_out)])
    rows = [json.loads(x) for x in (mine_out / "candidates.jsonl").read_text().splitlines()]
    kinds = {(r["image_id"], r["cls"], r["kind"]) for r in rows}
    assert ("a", "MA", "FN") in kinds and ("a", "MA", "FP") in kinds and ("a", "HE", "agree") in kinds
    assert ("b", "EX", "agree") in kinds and ("b", "SE", "FP") in kinds
    conf = [r for r in rows if r["image_id"] == "a" and r["cls"] == "EX"]
    assert conf and conf[0]["kind"] == "FN" and conf[0]["subtype"] == "confusion:EX->HE"

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

    build_pack.main(["--selection", "work/syn01/selection.json", "--out", "packs/syn01", "--root", "."])
    pack = json.loads((root / "packs" / "syn01" / "pack.json").read_text())
    text = (root / "packs" / "syn01" / "pack.json").read_text()
    for hidden in ('"kind"', '"arm"', '"weight"', '"image_id"', '"subtype"', '"prob_max"', "SYN_model", "prepared/SYN"):
        assert hidden not in text, f"pack leaks {hidden}"
    assert (root / "packs" / "syn01" / "index.html").is_file() and (root / "packs" / "syn01" / "pack.js").is_file()
    assert {r["id"] for c in pack["cases"] for r in c["regions"]} == {x["inst_id"] for x in sel["instances"]}

    # readers see the clinical labels (docs/clinical/31), the pack keeps the benchmark's class codes as keys
    assert {k: v["label"] for k, v in pack["classes"].items()} == {"MA": "MA", "HE": "HG", "EX": "HEX", "SE": "CWS"}
    # a reader export: every region read as the lesion the model reported, the label judged wrong
    code = {"MA": "ma", "HE": "hg", "EX": "hex", "SE": "cws"}
    export = {"format": "retiseg-clinical-review/1", "pack_id": "syn01", "reader_id": "R1", "cases": {}, "debrief": {"answers": {"d_useful": "all"}}}
    for c in pack["cases"]:
        regs = {}
        for r in c["regions"]:
            x = next(i for i in sel["instances"] if i["inst_id"] == r["id"])
            regs[r["id"]] = {"a": {"answers": {"a_finding": code[x["cls"]], "a_confidence": "certain"}, "t": {"active_ms": 20000}},
                             "b": {"answers": {"b_cause": "label_error", "b_label_issue": ["omission"], "b_icdr": ["none"],
                                               "b_relevance": "no"}, "t": {"active_ms": 25000, "completed": "x"}},
                             "revealed_at": "x"}
        export["cases"][c["id"]] = {"practice": False, "done": True, "regions": regs, "display_size": [c["width"], c["height"]],
                                    "pre": {"answers": {"pre_icdr": "2", "pre_fovea": {"x": 300, "y": 250}, "pre_disc": {"x": 400, "y": 250}},
                                            "t": {"active_ms": 30000, "completed": "x"}},
                                    "post": {"answers": {"post_ai_grade": "same", "post_unlabelled": "few_peripheral"}, "t": {"active_ms": 10000, "completed": "x"}}}
    (root / "R1.json").write_text(json.dumps(export))
    analyze_reviews.main(["--selection", "work/syn01/selection.json", "--reviews", "R1.json", "--out", "work/syn01/analysis"])
    report = (root / "work" / "syn01" / "analysis" / "report.md").read_text()
    assert "Blinded verdict" in report and "R1" in report and "Distance from the fovea" in report
    import csv
    rows = list(csv.DictReader(open(root / "work" / "syn01" / "analysis" / "regions_long.csv")))
    assert rows and all(r["blind_verdict"] in ("model_right", "dataset_right", "both_right") for r in rows)
    # fovea at (300, 250), disc 100 px away -> 1 DD = 40 px; every region gets a distance and a band
    assert all(r["dist_band"] for r in rows)


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


def test_fovea_distance_bands():
    case = {"set": "DDR", "image_size": [1000, 800], "fov": {"diameter": 750.0}}
    state = {"pre": {"answers": {"pre_fovea": {"x": 500, "y": 400}, "pre_disc": {"x": 750, "y": 400}}}, "display_size": [1000, 800]}
    # 1 DD = 250 / 2.5 = 100 px
    assert analyze_reviews.fovea_distance(case, state, {"centroid": [550, 400]})[:2] == (0.5, "central <=1 DD")
    assert analyze_reviews.fovea_distance(case, state, {"centroid": [700, 400]})[1] == "posterior 1-3 DD"
    assert analyze_reviews.fovea_distance(case, state, {"centroid": [900, 400]})[1] == "peripheral >3 DD"
    # no disc click: standard field falls back to FOV / 7.5 (approximate); ultra-widefield gets nothing
    state["pre"]["answers"]["pre_disc"] = {"not_visible": True}
    assert analyze_reviews.fovea_distance(case, state, {"centroid": [600, 400]})[2] is True
    assert analyze_reviews.fovea_distance(dict(case, set="TJDR_uwf"), state, {"centroid": [600, 400]}) == (None, None, None)
    # a pack shown at half size: the click is in display pixels, the centroid in native pixels
    half = {"pre": {"answers": {"pre_fovea": {"x": 250, "y": 200}, "pre_disc": {"x": 375, "y": 200}}}, "display_size": [500, 400]}
    assert analyze_reviews.fovea_distance(case, half, {"centroid": [550, 400]})[0] == 0.5


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
            if cond and "q" in cond:
                assert cond["q"] in qids, f"{q['id']} depends on a question of another step"
    finding = next(q for q in schema["steps"]["region_a"]["questions"] if q["id"] == "a_finding")
    assert set(o["code"] for o in finding["options"]) == set(analyze_reviews.FINDING_GROUP)
    icdr = next(q for q in schema["steps"]["region_b"]["questions"] if q["id"] == "b_icdr")
    assert [o["code"] for o in icdr["options"] if o.get("exclusive")] == ["none"]
    assert {o["code"] for o in icdr["options"]} == set(analyze_reviews.ICDR_ORDER)
    labels = " ".join(o["label"] + " " + o.get("help", "") for st in schema["steps"].values() for q in st["questions"] for o in q.get("options", []))
    assert " HE " not in f" {labels} " and " EX " not in f" {labels} " and " SE " not in f" {labels} ", "readers must see HG / HEX / CWS"
