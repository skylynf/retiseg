import json
from pathlib import Path

import yaml

from bench.eval import cli
from bench.models.base import FAMILIES, REPRO_LEVELS, TRACKS
from bench.tests.test_evaluate import _write_preds, toy_dataset  # noqa: F401

REPO = Path(__file__).resolve().parents[2]


def _declared_repos():
    """Submodule paths and ignored clones; a fresh checkout may not have them on disk."""
    declared = set()
    for name in (".gitmodules", ".gitignore"):
        path = REPO / name
        if not path.is_file():
            continue
        for line in path.read_text().splitlines():
            line = line.strip()
            if line.startswith("path ="):
                declared.add(line.split("=", 1)[1].strip())
            elif line.startswith("official_code/"):
                declared.add(line.rstrip("/"))
    return declared


def test_model_registry_is_consistent():
    reg = yaml.safe_load((REPO / "bench/configs/models.yaml").read_text())
    declared = _declared_repos()
    assert set(reg) <= set(FAMILIES)
    names = []
    for family, entries in reg.items():
        for e in entries:
            names.append(e["name"])
            assert e["repro_level"] in REPRO_LEVELS, e
            assert set(e["tracks"]) <= set(TRACKS), e
            assert e["tier"] in (1, 2, 3), e
            if e["repo"]:
                assert (REPO / e["repo"]).is_dir() or e["repo"].rstrip("/") in declared, e
            if "C" in e["tracks"]:
                assert e["tier"] == 2, e
    assert len(names) == len(set(names))


def test_cli_evaluate_then_compare(toy_dataset, tmp_path, capsys):  # noqa: F811
    root, truth = toy_dataset
    _write_preds(tmp_path / "a", truth, noise=0.1, seed=1)
    _write_preds(tmp_path / "b", truth, noise=0.4, seed=2)
    for name in ("a", "b"):
        cli.main(["evaluate", "--data", str(root), "--pred", str(tmp_path / name), "--out", str(tmp_path / f"{name}_test"), "--no-lesion"])
    capsys.readouterr()
    cli.main(["compare", "--a", str(tmp_path / "a_test.npz"), "--b", str(tmp_path / "b_test.npz")])
    out = json.loads(capsys.readouterr().out)
    assert set(out) == {"MA", "HE", "EX"}
    assert out["MA"]["aupr_paired_bootstrap"]["delta"] > 0
    assert "p_holm" in out["MA"]["per_image_dice_wilcoxon"]
