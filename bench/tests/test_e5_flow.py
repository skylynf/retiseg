"""The documented E5 inference flow, end to end, with fake models (docs/ops/21 open issue 27).

17 section 6 says: b. smoke --pool-only --limit 2; c. the full pool runs; d. the two selections
written back to e5_frozen.yaml; e. --test-only --drop-maps FGADR,Retinal-Lesions; f. collect.
This test walks exactly that order. The launcher plans the items and checks their gates as the
queue would; each item's command line is then executed in this process with the fake generator,
decoder and scorer of test_e5_infer on toy prepared sets (canvas 20, window 8). Review A2, A3 and
A5 were seams between the launcher, the runners and the evaluation that no unit test crossed;
this one does. When 17 section 6 changes, change this test with it.
"""
import json
import sys
import types
from pathlib import Path

import yaml

from bench.e5 import amg
from bench.e5 import frozen as fz
from bench.e5 import geometry as geo
from bench.e5 import outputs as out
from bench.e5 import prompted, sam3_text
from bench.scripts import launch_e5_infer as launch
from bench.tests.test_e5_infer import _FakeDecoder, _FakeGenerator, _FakeScorer, _toy_e5_source

REPO = Path(__file__).resolve().parents[2]
FROZEN = REPO / "bench/configs/e5_frozen.yaml"
LONG_SIDE, WINDOW = 20, 8
SETS = ("IDRiD", "DDR", "FGADR", "TJDR_uwf")  # the two pool sets, a drop-maps set outside the k curves, the set with a canvas variant
DROP = ("FGADR", "Retinal-Lesions")


def _flag(argv, name, default=None):
    return argv[argv.index(name) + 1] if name in argv else default


def _execute(item, frozen, drop_maps=(), limit=None):
    """Run one queue item in this process: the argv the launcher would pass, with the fake models injected."""
    argv = list(item.argv)
    if item.phase == launch.TEST and item.owns_drop and item.set_name in set(drop_maps):
        argv += ["--drop-maps", item.set_name]
    drop = [_flag(argv, "--drop-maps")] if "--drop-maps" in argv else []
    module = argv[argv.index("-m") + 1]
    set_name, split, variant = _flag(argv, "--set"), _flag(argv, "--split", "test"), _flag(argv, "--variant", geo.MAIN)
    common = dict(frozen=frozen, long_side=LONG_SIDE, window=WINDOW, drop_maps=drop, limit=limit)
    if module == "bench.e5.amg":
        return amg.run(_flag(argv, "--model"), int(_flag(argv, "--density")), set_name, split, geo.G1, variant, None, generator=_FakeGenerator(), **common)
    if module == "bench.e5.prompted":
        return prompted.run(_flag(argv, "--model"), _flag(argv, "--kind"), set_name, split, variant, geo.G1, None, decoder=_FakeDecoder(), **common)
    if module == "bench.e5.sam3_text":
        if "--assemble" in argv:
            return sam3_text.assemble_selected(set_name, split, variant, frozen=frozen, drop_maps=drop, limit=limit)
        return sam3_text.run(int(_flag(argv, "--phrase")), set_name, split, variant, geo.G1, None, scorer=_FakeScorer(fz.phrase_ids(frozen)),
                             assemble="--no-assemble" not in argv, **common)
    raise AssertionError(f"unexpected item {item.label}: {argv}")


def _run_plan(items, frozen, drop_maps=(), limit=None):
    """Execute a plan in queue order, asserting every gate is open when its item's turn comes."""
    producers = launch.producers_of(items)
    for item in items:
        gate = launch.gate_for(item, producers, limit)
        assert gate is None or gate() == "open", f"{item.label}: gate {gate()} when its turn came"
        _execute(item, frozen, drop_maps, limit)
        assert item.done(limit), item.label


def _collect_runs_module(monkeypatch):
    """collect_runs loaded privately, with a pandas stand-in for this test only where pandas is missing: e5_rows needs none of it."""
    import importlib.util

    try:
        import pandas  # noqa: F401
    except ImportError:
        monkeypatch.setitem(sys.modules, "pandas", types.ModuleType("pandas"))  # removed again when the test ends
    path = REPO / "docs" / "paper" / "retiseg_paper" / "paper" / "scripts" / "collect_runs.py"
    spec = importlib.util.spec_from_file_location("e5_flow_collect_runs", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_documented_flow_from_smoke_to_collect(tmp_path, monkeypatch):
    prepared = tmp_path / "prepared"
    monkeypatch.setattr(geo, "test_sets", lambda: list(SETS))
    for name in SETS:
        splits = {"val": ["v0", "v1", "v2"], "test": ["t0", "t1"]} if name in ("IDRiD", "DDR") else {"test": ["t0", "t1"]}
        _toy_e5_source(prepared / name, name, splits)
    monkeypatch.setattr(out, "OUT_ROOT", tmp_path / "runs" / "E5")
    monkeypatch.setattr(launch, "_ROOT", tmp_path)   # no B-std / nnU-Net maps here: the same-information references stay out of the flow
    # plan() leaves MedSAM out until this file exists; the documented 22-item flow includes it
    (tmp_path / "pretrained").mkdir()
    (tmp_path / "pretrained" / "medsam_vit_b.pth").write_bytes(b"")
    monkeypatch.setattr(fz, "PROMPTS_DIR", tmp_path / "prompts")
    monkeypatch.setattr(fz, "FROZEN_PATH", tmp_path / "e5_frozen.yaml")
    monkeypatch.setattr(geo, "data_dir", lambda name: prepared / name)
    monkeypatch.setattr(geo, "long_side_for", lambda set_name, variant, frozen: LONG_SIDE)
    monkeypatch.setattr(geo, "window_size", lambda frozen, model: WINDOW)
    monkeypatch.setattr(fz, "pool_sets", lambda frozen: [("IDRiD", prepared / "IDRiD", "val"), ("DDR", prepared / "DDR", "val")])
    (tmp_path / "e5_frozen.yaml").write_text(FROZEN.read_text())
    frozen = fz.load_frozen()
    runs = tmp_path / "runs" / "E5"

    # b. smoke: --pool-only --limit 2 (the launcher writes the prompt files first, as --submit does)
    plan = launch.plan(frozen, pool_only=True, limit=2)
    assert len(plan) == 22 and launch.absent_sets(plan) == []
    launch.write_prompt_files(frozen, launch.missing_prompt_files(frozen, plan))
    _run_plan(plan, frozen, limit=2)
    smoke = json.loads((runs / "a" / "a_sam_vith_p64" / "DDR-val" / "done.json").read_text())
    assert smoke["limit"] == 2 and smoke["n_images"] == 2 and smoke["thresholds"] == "fixed only"

    # c. the full pool runs: the smoke directories count as unfinished and are redone (A5)
    plan = launch.plan(frozen, pool_only=True)
    assert len(plan) == 22
    _run_plan(plan, frozen)
    full = json.loads((runs / "a" / "a_sam_vith_p64" / "DDR-val" / "done.json").read_text())
    assert full["limit"] is None and full["n_images"] == 3 and full["excluded"] == []
    assert launch.plan(frozen, pool_only=True) == []  # nothing left on the pool
    assert all(it.phase == launch.TEST and it.label.startswith("b_") for it in launch.plan(frozen))  # no selection yet: the oracle tier only

    # d. the two selections, written back to the protocol as 17 section 6 d says
    chosen, table = amg.select_grid(frozen)
    phrases, _ = sam3_text.select_phrases(frozen)
    doc = yaml.safe_load((tmp_path / "e5_frozen.yaml").read_text())
    doc["zero_shot"]["grid"].update(yaml.safe_load(amg.format_selected_yaml(chosen, table)))
    doc["zero_shot"]["sam3"].update(yaml.safe_load(sam3_text.format_selected_yaml(phrases, _)))
    (tmp_path / "e5_frozen.yaml").write_text(yaml.safe_dump(doc))
    frozen = fz.load_frozen()
    assert set(fz.selected_pairs(frozen)) == {"SAM", "SAM 2"} and set(fz.selected_phrases(frozen)) == {"MA", "HE", "EX", "SE"}
    record = out.write_selection_record("selection_grid", amg.selection_payload(chosen, table, frozen), frozen)
    assert json.loads(record.read_text())["pool_images"] == {"IDRiD": 3, "DDR": 3}

    # e. --test-only --drop-maps FGADR,Retinal-Lesions: the pool assemblies come first, every gate opens in order
    plan = launch.plan(frozen, test_only=True)
    assert all(it.prerequisite for it in plan[:2]) and {it.label for it in plan[:2]} == {"a_sam3_selected:IDRiD:val:main", "a_sam3_selected:DDR:val:main"}
    launch.write_prompt_files(frozen, launch.missing_prompt_files(frozen, plan))  # --submit writes the plan's missing prompt files first
    _run_plan(plan, frozen, drop_maps=DROP)
    assert launch.plan(frozen) == [] and launch.plan(frozen, test_only=True) == []
    test_dones = [json.loads(p.read_text()) for p in runs.rglob("done.json")]
    test_dones = [d for d in test_dones if d.get("split") == "test"]
    # a: 2 models x 5 set-variants x 3 kinds; SAM 3: 3 phrases + 1 assembly per set-variant; b: 4 sources x (2 curve sets x 4 k + 3 others x 1 k)
    assert len(test_dones) == 2 * 5 * 3 + 4 * 5 + 4 * (2 * 4 + 3 * 1)
    for done in test_dones:
        assert done["thresholds"] == "fixed and pool val_opt", (done["source"], done["set"])
        assert done["maps_dropped"] is (done["set"] in DROP), (done["source"], done["set"])
    for set_name in (s for s in DROP if s in SETS):
        selected = json.loads((runs / "a" / "a_sam3_selected" / set_name / "done.json").read_text())
        assert selected["kind"] == "text_selected" and selected["maps_dropped"] is True  # A2: the main SAM 3 result exists on a drop-maps set
        assert not (runs / "a" / "a_sam3_ph0" / set_name / "prob_test").exists()
    assert (runs / "a" / "a_sam3_selected" / "IDRiD" / "prob_test").is_dir() and (runs / "a" / "a_sam3_ph1" / "IDRiD-val" / "prob_val").is_dir()
    assert sorted(p.name for p in (runs / "b" / "b_sam_vitb_box").iterdir() if p.name.startswith("FGADR")) == ["FGADR__kall"]  # D6
    assert sorted(p.name for p in (runs / "b" / "b_sam_vitb_box").iterdir() if p.name.startswith("IDRiD__")) == ["IDRiD__k1", "IDRiD__k20", "IDRiD__k5", "IDRiD__kall"]
    assert len(list((runs / "a" / f"a_sam_vith_p{fz.selected_pair(frozen, 'SAM')[0]}").glob("TJDR_uwf*"))) == 6  # main and fov, three kinds each

    # f. collect reads every finished directory of the main canvas variant
    collect_runs = _collect_runs_module(monkeypatch)
    rows, seen = collect_runs.e5_rows(tmp_path / "runs")
    keys = {(r["model"], r["variant"], r["test_set"]) for r in rows}
    for test in ("IDRiD", "DDR", "TJDR-uwf", "FGADR"):
        assert ("SAM", "zero_shot:auto_any_lesion", test) in keys and ("SAM 2", "zero_shot:auto_oracle_class", test) in keys
        assert ("SAM 3", "text:selected", test) in keys and ("MedSAM", "oracle:box:all", test) in keys and ("SAM 2", "oracle:point:all", test) in keys
    assert ("SAM", "oracle:box:1", "IDRiD") in keys and ("SAM", "oracle:box:1", "FGADR") not in keys  # the k curves on the curve sets only
    assert any(r["metric"] == "aupr_any_lesion" and r["model"] == "SAM" and r["test_set"] == "FGADR" for r in rows)
    assert any(r["metric"] == "canvas_long_side" and r["test_set"] == "TJDR-uwf" for r in rows)
    assert not any("fov" in r["test_set"] for r in rows) and all(r["test_set"].endswith("-val") or r["seed"] == -1 for r in rows)
