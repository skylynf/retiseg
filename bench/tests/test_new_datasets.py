import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from bench.common import io
from bench.data import diaretdb, fgadr, retlesion, splits
from bench.data.prepare import CASE_LOG, export_cases, write_index, write_report
from bench.eval.evaluate import evaluate, load_protocol

REPO = Path(__file__).resolve().parents[2]


def _needs(*paths):
    missing = [path for path in paths if not (REPO / path).exists()]
    return pytest.mark.skipif(bool(missing), reason=f"raw data not on this machine: {missing}")


def test_holdout_is_seeded_stratified_and_hits_the_rounded_total():
    strata = {"a": [f"a{i}" for i in range(7)], "b": [f"b{i}" for i in range(13)], "c": ["c0"]}
    kept, held = splits.holdout(strata, 0.3, splits.generator())
    again = splits.holdout(strata, 0.3, splits.generator())
    assert (kept, held) == again
    assert len(held) == round(0.3 * 21)
    assert set(kept).isdisjoint(held) and len(kept) + len(held) == 21
    assert 2 <= sum(h.startswith("a") for h in held) <= 3
    assert 3 <= sum(h.startswith("b") for h in held) <= 4


def test_holdout_does_not_depend_on_input_order():
    ids = [f"x{i}" for i in range(30)]
    forward = splits.holdout({"s": ids}, 0.2, splits.generator())
    backward = splits.holdout({"s": ids[::-1]}, 0.2, splits.generator())
    assert forward == backward


def test_frozen_lists_refuse_a_different_draw(tmp_path, monkeypatch):
    monkeypatch.setattr(splits, "FROZEN", tmp_path)
    lists = {"train": ["a", "b"], "val": ["c"], "test": ["d"]}
    splits.write_frozen("toy", lists, "rule")
    splits.write_frozen("toy", lists, "rule")
    with pytest.raises(RuntimeError):
        splits.write_frozen("toy", {"train": ["a"], "val": ["b", "c"], "test": ["d"]}, "rule")
    with pytest.raises(ValueError):
        splits.write_frozen("bad", {"train": ["a"], "val": ["a"], "test": []}, "rule", force=True)
    assert splits.read_frozen("toy") == lists


def test_fgadr_masks_are_positive_from_128_and_rgb_uses_the_gray_value():
    gray = np.array([[0, 1, 127, 128, 255]], dtype=np.uint8)
    positive, dropped = fgadr.binarize(gray)
    assert positive.tolist() == [[False, False, False, True, True]]
    assert dropped == 2
    rgb = np.repeat(gray[..., None], 3, axis=-1)
    assert np.array_equal(fgadr.binarize(rgb)[0], positive)


def test_retinal_lesions_127_is_ignore_and_other_values_stop():
    arr = np.array([[0, 127, 255]], dtype=np.uint8)
    lesion, ignore = retlesion.decode(arr)
    assert lesion.tolist() == [[False, False, True]]
    assert ignore.tolist() == [[False, True, False]]
    with pytest.raises(ValueError):
        retlesion.decode(np.array([[0, 128]], dtype=np.uint8))
    assert retlesion.patient("10037_left") == "10037"


def test_diaretdb1_shapes_levels_and_consensus():
    size = (40, 60)
    circle = diaretdb.rasterize({"geometry": "circle", "center": (30, 20), "radius": 5}, size)
    assert circle[20, 30] and circle[20, 35] and not circle[20, 36]
    flat = {"geometry": "ellipse", "center": (30, 20), "radii": (10, 3), "angle": 0}
    ellipse = diaretdb.rasterize(flat, size)
    assert ellipse[20, 39] and not ellipse[25, 30]
    turned = diaretdb.rasterize(dict(flat, angle=90), size)
    assert turned[29, 30] and not turned[20, 39]
    square = diaretdb.rasterize({"geometry": "polygon", "points": [(5, 5), (15, 5), (15, 15), (5, 15)]}, size)
    assert square[10, 10] and not square[20, 20]

    levels = {e: np.zeros((1, 4), dtype=np.uint8) for e in diaretdb.EXPERTS}
    levels["01"][0] = [3, 3, 2, 1]
    levels["02"][0] = [3, 3, 2, 1]
    levels["03"][0] = [3, 2, 2, 1]
    levels["04"][0] = [0, 2, 2, 0]
    assert np.array_equal(diaretdb.unpack_levels(diaretdb.pack_levels(levels))["03"], levels["03"])
    assert diaretdb.positive(levels).tolist() == [[True, True, True, False]]
    assert diaretdb.union(levels).all()


def _toy_case(image_id, ignore=False):
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    yy, xx = np.ogrid[:32, :32]
    image[(yy - 16) ** 2 + (xx - 16) ** 2 <= 14**2] = 120
    masks = {c: np.zeros((32, 32), dtype=bool) for c in io.LESION_CLASSES}
    masks["HE"][10:14, 10:14] = True
    case = {"bytes": b"not-an-image", "ext": "png", "image": image, "masks": masks}
    if ignore:
        region = np.zeros((32, 32), dtype=bool)
        region[20:24, 20:24] = True
        case["binary"] = {"ignore": {"HE": region}, "masks_ext": {"NV": masks["HE"].copy()}}
    return case


def test_export_resumes_after_the_last_logged_image(tmp_path):
    dest = tmp_path / "TOY"
    lists = {"train": ["a", "b"], "val": [], "test": ["c"]}
    write_index(dest, "TOY", lists, "coarse")
    calls = []

    def load(image_id):
        calls.append(image_id)
        if image_id == "b" and calls.count("b") == 1:
            raise KeyboardInterrupt
        return _toy_case(image_id, ignore=True)

    with pytest.raises(KeyboardInterrupt):
        export_cases(dest, ["a", "b", "c"], load, log=lambda _: None)
    export_cases(dest, ["a", "b", "c"], load, log=lambda _: None)
    assert calls == ["a", "b", "b", "c"]
    assert len((dest / CASE_LOG).read_text().splitlines()) == 3
    assert (dest / "ignore" / "HE" / "a.png").is_file()
    assert (dest / "masks_ext" / "NV" / "c.png").is_file()
    assert not (dest / "masks" / "MA" / "a.png").exists()
    report = write_report(dest)
    assert report["splits"]["train"]["n_images"] == 2
    assert report["splits"]["train"]["images_with"]["HE"] == 2
    assert report["splits"]["test"]["layer_images_with"]["ignore"]["HE"] == 1


def test_ignore_region_is_left_out_of_that_class_only(tmp_path):
    root = tmp_path / "IGN"
    shape = (20, 20)
    write_index(root, "IGN", {"train": [], "val": [], "test": ["t"]}, "coarse")
    (root / "fov").mkdir()
    Image.fromarray(np.full(shape, 255, np.uint8)).save(root / "fov" / "t.png")
    lesion = np.zeros(shape, bool)
    lesion[2:6, 2:6] = True
    ignore = np.zeros(shape, bool)
    ignore[10:14, 10:14] = True
    for cls in ("HE", "MA"):
        (root / "masks" / cls).mkdir(parents=True)
        Image.fromarray(lesion.astype(np.uint8) * 255).save(root / "masks" / cls / "t.png")
    (root / "ignore" / "HE").mkdir(parents=True)
    Image.fromarray(ignore.astype(np.uint8) * 255).save(root / "ignore" / "HE" / "t.png")
    pred = tmp_path / "pred"
    prob = np.where(lesion, 0.6, 0.0)
    prob[ignore] = 0.9
    for cls in io.LESION_CLASSES:
        io.write_prob(pred, cls, "t", prob)
    proto = load_protocol(bootstrap={"n": 10, "seed": 0, "ci": 0.95}, lesion_metrics=False)
    result = evaluate(root, "test", pred, proto)
    assert result["per_class"]["HE"]["aupr"] == pytest.approx(1.0)
    assert result["per_class"]["MA"]["aupr"] < 0.7


def test_image_path_finds_the_one_file(tmp_path):
    (tmp_path / "images").mkdir()
    (tmp_path / "images" / "x.png").write_bytes(b"")
    assert io.image_path(tmp_path, "x").name == "x.png"
    with pytest.raises(FileNotFoundError):
        io.image_path(tmp_path, "y")


@_needs("dataset/TJDR")
def test_frozen_tjdr_lists_regenerate_from_the_release():
    from bench.data.tjdr import EXPECTED_FIELD, open_tjdr

    drawn = open_tjdr(REPO).make_splits()
    for name, lists in drawn.items():
        assert {k: len(v) for k, v in lists.items()} == EXPECTED_FIELD[name]
        assert splits.read_frozen(name) == lists


@_needs("dataset/FGADR/FGADR-Seg-set_Release.zip")
def test_frozen_fgadr_lists_regenerate_from_the_release():
    lists = fgadr.open_fgadr(REPO).make_splits()
    assert {k: len(v) for k, v in lists.items()} == {"train": 1096, "val": 193, "test": 553}
    assert splits.read_frozen("FGADR") == lists


@_needs("dataset/DiaRetDB1 V2.1")
def test_diaretdb1_counts_differ_from_the_paper_only_where_recorded():
    data = diaretdb.open_diaretdb1(REPO)
    assert diaretdb.count_problems(diaretdb.positive_image_counts(data)) == []
    sense = diaretdb.ellipse_sense(data)
    assert sense["inside_positive_sense"] == sense["ellipses"] == 763


def _retlesion_readable():
    try:
        retlesion.open_retlesion(REPO).grades()
    except (FileNotFoundError, RuntimeError):
        return False
    return True


@pytest.mark.skipif(not _retlesion_readable(), reason="Retinal-Lesions not unpacked and no password")
def test_frozen_retinal_lesions_lists_keep_patients_together():
    lists = retlesion.open_retlesion(REPO).make_splits()
    assert splits.read_frozen("Retinal-Lesions") == lists
    owner = {retlesion.patient(i): s for s, ids in lists.items() for i in ids}
    for split, ids in lists.items():
        assert all(owner[retlesion.patient(i)] == split for i in ids)
    assert json.loads(json.dumps(lists)) == lists
