from pathlib import Path

import pytest

from bench.data.catalog import catalog
from bench.data.ddr import EXPECTED_GRADING, EXPECTED_SEGMENTATION, open_ddr
from bench.data.diaretdb import open_diaretdb0, open_diaretdb1
from bench.data.idrid import VAL_IDS, open_idrid
from bench.data.tjdr import EXPECTED, open_tjdr

REPO = Path(__file__).resolve().parents[2]


def _needs(*paths):
    """The raw releases stay on the workstation; the server has only dataset/prepared and iDRID."""
    missing = [path for path in paths if not (REPO / path).exists()]
    return pytest.mark.skipif(bool(missing), reason=f"raw data not on this machine: {missing}")


@_needs("dataset/DDR", "dataset/TJDR", "dataset/DIARETDB1")
def test_catalog_covers_local_and_pending():
    names = [r.name for r in catalog(REPO)]
    assert names[:5] == ["IDRiD", "DDR-grading", "DDR-lesion", "DIARETDB1", "DIARETDB0"]
    by_name = {r.name: r for r in catalog(REPO)}
    assert by_name["IDRiD"].status == "ready"
    assert by_name["DDR-lesion"].status == "ready_split_zip"
    assert by_name["DDR-grading"].status == "not_for_lesion_segmentation"
    assert by_name["TJDR"].status == "ready_zip"
    assert (REPO / by_name["IDRiD"].path).is_dir()
    assert (REPO / by_name["DDR-grading"].path).is_file()


@_needs("dataset/iDRID")
def test_idrid_official_split_and_held_out_validation():
    data = open_idrid(REPO)
    splits = data.splits()
    assert len(splits["train_official"]) == 54
    assert len(splits["test"]) == 27
    assert splits["test"][0] == "IDRiD_55" and splits["test"][-1] == "IDRiD_81"
    assert len(splits["val"]) == 10
    assert set(splits["val"]) == set(VAL_IDS)
    assert set(splits["train"]).isdisjoint(splits["val"])
    assert set(splits["train"]) | set(splits["val"]) == set(splits["train_official"])
    assert data.image_path("IDRiD_01").is_file()
    assert data.image_path("IDRiD_81").is_file()


@_needs("dataset/iDRID")
def test_idrid_missing_lesion_file_is_a_negative_mask():
    data = open_idrid(REPO)
    present = data.read_mask("IDRiD_53", "MA")
    absent = data.read_mask("IDRiD_43", "HE")
    assert present.shape == absent.shape == (2848, 4288)
    assert present.any()
    assert not absent.any()
    assert not data.mask_path("IDRiD_43", "HE").is_file()
    exudate = data.read_mask("IDRiD_81", "EX")
    assert 0 < exudate.mean() < 0.1


@_needs("dataset/DDR")
def test_official_ddr_counts_and_validation_label_folder():
    data = open_ddr(REPO)
    assert len(data.parts) == 10
    for split, count in EXPECTED_SEGMENTATION.items():
        assert len(data.segmentation_ids(split)) == count
    for split, count in EXPECTED_GRADING.items():
        assert data.grading_count(split) == count
    assert data.label_dir("valid") == "segmentation label"
    assert data.label_dir("train") == "label"
    assert "007-6219-300" in data.segmentation_ids("test")
    assert "007-6325-400" in data.segmentation_ids("test")
    assert data.read_mask("train", "007-1774-100", "MA").any()
    assert not data.read_mask("train", "007-1774-100", "EX").any()


@_needs("dataset/DIARETDB1", "dataset/DiaRetDB V2.1")
def test_diaretdb1_is_v21_and_diaretdb0_is_in_the_other_folder():
    db1 = open_diaretdb1(REPO)
    splits = db1.splits()
    assert len(splits["train"]) == 28
    assert len(splits["test"]) == 61
    assert db1.image_path(splits["train"][0]).is_file()
    markings = db1.markings(splits["train"][0])
    assert {m["expert"] for m in markings} == {"01", "02", "03", "04"}
    assert any(m["lesion"] == "EX" for m in markings)
    db0 = open_diaretdb0(REPO)
    assert len(db0.image_ids()) == 130
    assert "DIARETDB1" in db0.stored_under


@_needs("dataset/TJDR")
def test_tjdr_split_and_class_values():
    data = open_tjdr(REPO)
    splits = data.splits()
    assert len(splits["train"]) == EXPECTED["train"]
    assert len(splits["test"]) == EXPECTED["test"]
    assert set(splits["train"]).isdisjoint(splits["test"])
    assert data.field("TJDR_test_089") == "standard"
    assert data.field("TJDR_train_140") == "ultrawide"
    mask = data.read_mask("TJDR_train_318", "SE")
    assert mask.shape == (2048, 2048)
    assert mask.any()
    assert not data.read_mask("TJDR_train_391", "SE").any()
