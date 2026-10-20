"""E3a: finished checkpoints on the test splits of sources they were not trained on.

The protocol is bench/configs/e3a_frozen.yaml. A source run is a scored B-std
or B1 run directory, an nnU-Net run (bench/scripts/nnunet_e3a.py, same
helpers) or a released author checkpoint. For each test set other than the
training source, this writes runs/E3a/<source>/<set>[__<variant>]/prob_test
with bench.common.io.write_prob, then evaluates it with bench.eval under the
frozen protocol. Dice thresholds come from the training source's validation
split (the run's prob_val), never from the test source. A set is finished when
its done.json exists; a rerun skips it and redoes any set that was cut short.

    python -m bench.e3a plan --run runs/BS_hrnet_idrid_seed0
    python -m bench.e3a run --run runs/BS_hrnet_idrid_seed0 [--sets TJDR_std,FGADR]
    python -m bench.e3a author --model M2MRF --train IDRiD      # also HRDecoder (retiseg-hacdr environment)
    python -m bench.e3a evaluate --source BS_hrnet_idrid_seed0 [--sets ...]

The canvas of a test image follows the training scale: field-of-view size
matched for standard-field sources, pixels per degree for TJDR_uwf (its
field-of-view-matched version is the variant "fov").
"""

import argparse
import json
import shutil
import socket
import statistics
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import yaml

REPO = _ROOT
FROZEN_PATH = REPO / "bench" / "configs" / "e3a_frozen.yaml"
LEAK_EXCLUSIONS = REPO / "bench" / "data" / "splits" / "leak_exclusions.json"
OUT_ROOT = REPO / "runs" / "E3a"
MAIN = "main"


def load_frozen(path=FROZEN_PATH):
    return yaml.safe_load(Path(path).read_text())


PREPARED_TO_SET = {"RetinalLesions": "Retinal-Lesions"}  # prepared directory name -> test-set name


def train_source_of(config):
    """IDRiD or DDR, from a cell config's dataset path (B-std and B1 runs)."""
    name = Path(config["dataset"]).name
    if name not in ("IDRiD", "DDR"):
        raise ValueError(f"E3a source runs are trained on IDRiD or DDR, got {name}")
    return name


def train_sources_of(config):
    """Every source a run was trained on, as test-set names: one for BS/B1, the cell's train list for BF."""
    if config.get("experiment") == "BF":
        names = [Path(item["dataset"]).name for item in config["train"]]
        return [PREPARED_TO_SET.get(name, name) for name in names]
    return [train_source_of(config)]


def scale_source_of(config):
    """The training source whose field of view sets the per-degree scale; None when it is not known (BF on TJDR_std, FGADR)."""
    if config.get("experiment") != "BF":
        return train_source_of(config)
    frozen = load_frozen()
    names = train_sources_of(config)
    known = [name for name in names if name in frozen["train_field_degrees"]]
    return known[0] if len(known) == len(names) and len(set(frozen["train_field_degrees"][n] for n in known)) == 1 else None


def targets(train_names, frozen, sets=None):
    """(test set, variant) pairs for a model trained on ``train_names``.

    Every test set except the training sources themselves; a per-degree set
    adds its comparison variant. ``sets`` restricts to those names.
    """
    trained = set(train_names)
    out = []
    for name in frozen["test_sets"]:
        if name in trained or (sets is not None and name not in sets):
            continue
        out.append((name, MAIN))
        if name in frozen.get("comparison_variants", {}):
            out.append((name, frozen["comparison_variants"][name]))
    if sets is not None:
        unknown = sorted(set(sets) - set(frozen["test_sets"]))
        if unknown:
            raise ValueError(f"unknown test sets {unknown}; known {sorted(frozen['test_sets'])}")
    return out


def target_long_side(test_name, variant, train_name, reference, frozen):
    """Long side of the field-of-view box on the network canvas.

    ``reference`` is the training scale: 1440 for B-std and B1, the median
    original field-of-view long side of the training split for nnU-Net.
    """
    if test_name in frozen.get("per_degree_sets", ()) and variant == MAIN:
        degrees = float(frozen["test_sets"][test_name]["field_degrees"])
        return int(round(float(reference) * degrees / float(frozen["train_field_degrees"][train_name])))
    if variant not in (MAIN, frozen.get("comparison_variants", {}).get(test_name)):
        raise ValueError(f"{test_name} has no variant {variant!r}")
    return int(round(float(reference)))


def set_dirname(test_name, variant):
    return test_name if variant == MAIN else f"{test_name}__{variant}"


def out_dir(source, test_name, variant):
    return OUT_ROOT / source / set_dirname(test_name, variant)


def test_data_dir(test_name, frozen):
    return REPO / frozen["test_sets"][test_name]["data"]


def leak_exclusions(train_names, test_name, path=None):
    """Test images showing an eye one of ``train_names`` has seen (bench/data/splits/leak_exclusions.json)."""
    path = Path(path) if path is not None else LEAK_EXCLUSIONS
    if not path.is_file():
        return set()
    table = json.loads(path.read_text())["by_training_source"]
    out = set()
    for name in train_names:
        out.update((table.get(name) or {}).get(test_name) or ())
    return out


def test_ids(test_name, frozen, train_names=()):
    """Test ids of a set, without the frozen exclusions and the leak exclusions of the training sources."""
    from bench.common.io import load_split

    ids = load_split(test_data_dir(test_name, frozen), frozen["split"])
    exclude = set((frozen.get("exclude") or {}).get(test_name) or ()) | leak_exclusions(train_names, test_name)
    return [image_id for image_id in ids if image_id not in exclude], sorted(exclude)


def is_done(source, test_name, variant):
    return (out_dir(source, test_name, variant) / "done.json").is_file()


def inference_geometry(config, card, long_side, frozen):
    """What bench.predict.predict_image needs for this canvas.

    B-std keeps its window rule (bench.bstd.inference_for). B1 resizes to
    card.forward_size, or runs the whole canvas. A fully convolutional model
    whose canvas exceeds large_canvas_window in either side runs overlapping
    windows of that size instead.
    """
    from bench.bstd import inference_for

    inference = inference_for(config, card)
    window = [int(v) for v in frozen["large_canvas_window"]]
    if card.forward_size is None and int(long_side) > max(window):
        # One large window per forward pass; eight 1440 windows do not fit a 40 GB card.
        return {"window": tuple(window), "overlap": float(frozen["window_overlap"]), "batch": 1}
    return inference


def _median_fov_long_side(train_dir, split="train"):
    """Median long side of the original field-of-view box over a split (nnU-Net's training scale)."""
    from bench.common.io import load_split, read_fov
    from bench.data.fov import crop_box

    sides = []
    for image_id in load_split(train_dir, split):
        y0, y1, x0, x1 = crop_box(read_fov(train_dir, image_id))
        sides.append(max(y1 - y0, x1 - x0))
    return float(statistics.median(sides))


def environment():
    import torch

    from bench.runtime import git_commit

    return {
        "host": socket.gethostname(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "torch": torch.__version__,
        "git": git_commit(),
    }


def evaluate_set(source, test_name, variant, frozen, train_dir=None, val_pred_dir=None, record=None, drop_maps=()):
    """bench.eval on one finished prediction directory; writes metrics.json/.npz and done.json.

    A test set named in ``drop_maps`` loses its prob_test after the metrics are
    written (metrics.npz keeps the per-image histograms for AUPR, bootstrap and
    paired tests); done.json records it.
    """
    from bench.eval.evaluate import evaluate, load_protocol

    directory = out_dir(source, test_name, variant)
    pred_dir = directory / "prob_test"
    if not (directory / "predicted.json").is_file():
        raise FileNotFoundError(f"{directory} has no finished prediction (predicted.json)")
    proto = load_protocol(REPO / frozen["eval_protocol"])
    predicted = json.loads((directory / "predicted.json").read_text())
    # Multi-source runs store the joined label in train_source ("DDR+Retinal-Lesions") and the
    # real names in train_sources. Exclusions are per source; the joined label matches none.
    sources = predicted.get("train_sources") or [predicted["train_source"]]
    _ids, excluded = test_ids(test_name, frozen, sources)
    use_val = val_pred_dir is not None and Path(val_pred_dir).is_dir()
    started = time.perf_counter()
    result = evaluate(
        test_data_dir(test_name, frozen),
        frozen["split"],
        pred_dir,
        proto,
        val_pred_dir=val_pred_dir if use_val else None,
        val_split="val",
        out_prefix=directory / "metrics",
        val_dataset_dir=train_dir if use_val else None,
        exclude=excluded,
    )
    done = dict(predicted)
    done.update(record or {})
    done.update(
        {
            "thresholds": "fixed and training-source val_opt" if use_val else "fixed only",
            "eval_seconds": round(time.perf_counter() - started, 1),
            "summary": result["summary"],
            "maps_dropped": test_name in set(drop_maps or ()),
        }
    )
    (directory / "done.json").write_text(json.dumps(done, indent=1) + "\n")
    if done["maps_dropped"]:
        shutil.rmtree(pred_dir)
    print(f"{source} {set_dirname(test_name, variant)} mAUPR {result['summary']['mAUPR']:.4f}", flush=True)
    return result


def _write_predicted(directory, payload):
    (directory / "predicted.json").write_text(json.dumps(payload, indent=1) + "\n")


# ---- B-std and B1 run directories ------------------------------------------------


def primary_checkpoint(run_dir):
    roles = Path(run_dir) / "checkpoint_roles.json"
    if not roles.is_file():
        raise FileNotFoundError(f"{run_dir} has no checkpoint_roles.json; score the run first")
    name = json.loads(roles.read_text())["primary"]
    path = Path(run_dir) / name
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_run_model(config, checkpoint, device):
    """The run's wrapper with the checkpoint loaded. M2MRF keeps the training source's normalization."""
    from bench.runtime import load_trusted

    name = config["model"]
    if name == "M2MRF":
        from bench.models.m2mrf import M2MRF, bind_normalization

        model = M2MRF()
        bind_normalization(model, config.get("budget_reference") or train_source_of(config))
        if model.network is None:
            raise RuntimeError(model.import_error or "M2MRF network was not imported")
    else:
        from bench.models.registry import build_model

        model = build_model(name)
        if config.get("experiment") == "BF":
            from bench.factor import apply_cell_geometry

            apply_cell_geometry(model, config)   # the E7 case study's patch key (26 N1c)
        if name == "HACDR-Net":
            from bench.scripts.predict_hacdr_b1 import _check_channels

            if model.net is None:
                raise ImportError(model.import_error or "HACDR-Net network was not imported")
            _check_channels(model)
    from bench.runtime import apply_matmul_precision

    apply_matmul_precision(model, config)
    state = load_trusted(checkpoint, map_location=device)
    if "model_name" in state and state["model_name"] != model.card.name:
        raise ValueError(f"checkpoint model {state['model_name']!r} != config model {model.card.name!r}")
    model.load_state_dict(state["model"])
    model.to(device)
    model.eval()
    return model


def dropped_maps(source, test_name, variant):
    """A finished set whose prob_test was removed after scoring (--drop-maps); its done.json says so."""
    directory = out_dir(source, test_name, variant)
    if not is_done(source, test_name, variant) or (directory / "prob_test").is_dir():
        return False
    return bool(json.loads((directory / "done.json").read_text()).get("maps_dropped"))


def run_source(run_dir, sets=None, device=None, predict_only=False, drop_maps=(), regenerate_maps=False):
    """``regenerate_maps`` re-predicts the sets whose maps were dropped after scoring (for the ensembles of
    bench/scripts/ensemble.py, docs/paper/14 M15): the maps are written back under the finished directory,
    predicted.json, metrics and done.json stay as they are, nothing is re-scored."""
    import torch

    from bench.predict import predict_image

    frozen = load_frozen()
    run_dir = Path(run_dir)
    config = yaml.safe_load((run_dir / "config.yaml").read_text())
    if config.get("experiment") not in ("BS", "B1", "BF"):
        raise ValueError(f"{run_dir} is not a B-std, B1 or BF run")
    if int(config["fov_diameter"]) != int(frozen["canvas_reference"]):
        raise ValueError(f"{run_dir} canvas {config['fov_diameter']} != {frozen['canvas_reference']} (E7 runs have no independent test)")
    train_names = train_sources_of(config)
    train_name = scale_source_of(config)  # None: per-degree sets are skipped, only their fov variant runs
    source = run_dir.name
    if regenerate_maps:
        pending = [(t, v) for t, v in targets(train_names, frozen, sets) if v == MAIN and dropped_maps(source, t, v)]
        if not pending:
            print(f"{source}: no finished set with dropped maps among the requested sets", flush=True)
            return
    else:
        pending = [(t, v) for t, v in targets(train_names, frozen, sets) if not is_done(source, t, v)]
    if train_name is None:
        pending = [(t, v) for t, v in pending if not (t in frozen.get("per_degree_sets", ()) and v == MAIN)]
    if not pending:
        print(f"{source}: every E3a set is done", flush=True)
        return
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = primary_checkpoint(run_dir)
    model = None
    env = environment()
    for test_name, variant in pending:
        directory = out_dir(source, test_name, variant)
        if regenerate_maps or not (directory / "predicted.json").is_file():
            if model is None:
                model = load_run_model(config, checkpoint, device)
            long_side = target_long_side(test_name, variant, train_name, frozen["canvas_reference"], frozen)
            inference = inference_geometry(config, model.card, long_side, frozen)
            ids, excluded = test_ids(test_name, frozen, train_names)
            data_dir = test_data_dir(test_name, frozen)
            started = time.perf_counter()
            with torch.no_grad():
                for index, image_id in enumerate(ids):
                    predict_image(model, data_dir, image_id, long_side, directory / "prob_test", device, inference)
                    if index % 50 == 0:
                        print(f"{source} {set_dirname(test_name, variant)} {index + 1}/{len(ids)}", flush=True)
            if regenerate_maps:
                print(f"{source} {set_dirname(test_name, variant)}: maps regenerated ({len(ids)} images); metrics unchanged", flush=True)
                continue
            _write_predicted(
                directory,
                {
                    "source": source,
                    "kind": config["experiment"],
                    "model": config["model"],
                    "seed": int(config["seed"]),
                    "train_source": train_name if config.get("experiment") != "BF" else "+".join(train_names),
                    "train_sources": train_names,
                    "checkpoint": str(checkpoint.relative_to(REPO) if checkpoint.is_absolute() else checkpoint),
                    "test_set": test_name,
                    "variant": variant,
                    "canvas_long_side": long_side,
                    "inference": None
                    if inference is None
                    else {
                        "window": None if inference["window"] is None else list(inference["window"]),
                        "overlap": inference["overlap"],
                        "batch": inference.get("batch", 8),
                    },
                    "forward_size": None if model.card.forward_size is None else list(model.card.forward_size),
                    "n_images": len(ids),
                    "excluded": excluded,
                    "predict_seconds": round(time.perf_counter() - started, 1),
                    **env,
                },
            )
        if not predict_only:
            evaluate_set(
                source,
                test_name,
                variant,
                frozen,
                train_dir=REPO / config["dataset"],
                val_pred_dir=run_dir / "prob_val",
                drop_maps=drop_maps,
            )


# ---- released author checkpoints (E1) ---------------------------------------------


def author_source(model, train_name):
    return f"E1_{model.lower()}_{train_name.lower()}"


def author_pipeline(model_name, train_name, checkpoint, device):
    """(predict(image path) -> 4 maps in author class order, class names, pipeline record) of a released checkpoint.

    M2MRF: bench/predict_m2mrf.py (keep-ratio resize, whole image). HRDecoder: bench/predict_hrdecoder.py
    (resize without keeping the ratio, slide for the IDRiD config). Both forks read BGR and normalise in RGB.
    """
    if model_name == "M2MRF":
        import numpy as np
        import torch

        from bench import predict_m2mrf as author

        spec = author.DATASETS[train_name.lower()]
        network, _meta = author.load_model(checkpoint, device)

        def predict(path):
            tensor, img_meta = author.preprocess(path, spec["mean"], spec["std"], spec["image_scale"])
            with torch.no_grad():
                prob = network.simple_test(tensor.to(device), img_meta, rescale=True)[0][0]
            prob = np.asarray(prob)
            if prob.shape[0] != 4:
                raise RuntimeError(f"expected 4 channels, got {prob.shape}")
            return prob

        record = {"image_scale": list(spec["image_scale"]), "keep_ratio": True, "mean": spec["mean"], "std": spec["std"],
                  "mode": "whole"}
        return predict, author.AUTHOR_CLASSES, record
    if model_name == "HRDecoder":
        from bench import predict_hrdecoder as author

        spec = author.DATASETS[train_name.lower()]
        network, info = author.load_model(checkpoint, device, spec["config_name"])

        def predict(path):
            tensor, img_meta = author.preprocess(path, spec["mean"], spec["std"], spec["image_scale"])
            return author.predict(network, tensor, img_meta, device)

        record = {"image_scale": list(spec["image_scale"]), "keep_ratio": False, "mean": spec["mean"], "std": spec["std"],
                  "hr_scale": info.get("hr_scale"), "test_cfg": info.get("test_cfg")}
        return predict, author.AUTHOR_CLASSES, record
    raise ValueError(f"no author pipeline wired for {model_name}")


def run_author(model_name, train_name, sets=None, device=None, predict_only=False, drop_maps=()):
    """The released checkpoint through the author's own test pipeline (author_pipeline)."""
    import numpy as np
    import torch

    from bench.common.io import image_path, write_prob

    frozen = load_frozen()
    spec_frozen = frozen["author_weights"][model_name][train_name]
    source = author_source(model_name, train_name)
    pending = [
        (t, v)
        for t, v in targets(spec_frozen["trained_on"], frozen, sets)
        if v == MAIN and not is_done(source, t, v)
    ]
    if not pending:
        print(f"{source}: every E3a set is done", flush=True)
        return
    checkpoint = REPO / spec_frozen["checkpoint"]
    if not checkpoint.is_file():
        raise FileNotFoundError(f"{checkpoint} (author_weights.{model_name}.{train_name} in e3a_frozen.yaml)")
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    predict = classes = record = None
    env = environment()
    for test_name, variant in pending:
        directory = out_dir(source, test_name, variant)
        if not (directory / "predicted.json").is_file():
            if predict is None:
                predict, classes, record = author_pipeline(model_name, train_name, checkpoint, device)
            ids, excluded = test_ids(test_name, frozen, spec_frozen["trained_on"])
            data_dir = test_data_dir(test_name, frozen)
            started = time.perf_counter()
            for index, image_id in enumerate(ids):
                prob = predict(image_path(data_dir, image_id))
                for channel, cls in enumerate(classes):
                    write_prob(directory / "prob_test", cls, image_id, np.clip(prob[channel], 0, 1))
                if index % 50 == 0:
                    print(f"{source} {test_name} {index + 1}/{len(ids)}", flush=True)
            _write_predicted(
                directory,
                {
                    "source": source,
                    "kind": "E1",
                    "model": model_name,
                    "train_source": train_name,
                    "checkpoint": spec_frozen["checkpoint"],
                    "test_set": test_name,
                    "variant": variant,
                    "author_pipeline": record,
                    "n_images": len(ids),
                    "excluded": excluded,
                    "predict_seconds": round(time.perf_counter() - started, 1),
                    **env,
                },
            )
        if not predict_only:
            evaluate_set(source, test_name, variant, frozen, drop_maps=drop_maps)


# ---- command line ---------------------------------------------------------------


def _sets(text):
    return None if not text else [item.strip() for item in text.split(",") if item.strip()]


def _plan(args):
    frozen = load_frozen()
    run_dir = Path(args.run)
    config = yaml.safe_load((run_dir / "config.yaml").read_text())
    train_name = train_source_of(config)
    for test_name, variant in targets([train_name], frozen, _sets(args.sets)):
        long_side = target_long_side(test_name, variant, train_name, frozen["canvas_reference"], frozen)
        state = "done" if is_done(run_dir.name, test_name, variant) else "todo"
        print(f"{run_dir.name} {set_dirname(test_name, variant)} canvas {long_side} {state}")


def _evaluate(args):
    frozen = load_frozen()
    base = OUT_ROOT / args.source
    for directory in sorted(base.iterdir()) if base.is_dir() else []:
        predicted = directory / "predicted.json"
        if not predicted.is_file():
            continue
        info = json.loads(predicted.read_text())
        if _sets(args.sets) and info["test_set"] not in _sets(args.sets):
            continue
        if (directory / "done.json").is_file() and not args.force:
            continue
        run_dir = REPO / "runs" / args.source
        train_dir = REPO / frozen["test_sets"][info["train_source"]]["data"]
        val = run_dir / "prob_val" if (run_dir / "prob_val").is_dir() and info.get("kind") != "E1" else None
        evaluate_set(args.source, info["test_set"], info["variant"], frozen, train_dir=train_dir, val_pred_dir=val)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="bench.e3a")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--run", required=True)
    p.add_argument("--sets")
    r = sub.add_parser("run")
    r.add_argument("--run", required=True, help="a scored B-std or B1 run directory")
    r.add_argument("--sets")
    r.add_argument("--device")
    r.add_argument("--predict-only", action="store_true")
    r.add_argument("--drop-maps", help="comma-separated test sets whose maps are removed after evaluation")
    r.add_argument("--regenerate-maps", action="store_true", help="re-predict finished sets whose maps were dropped; no re-scoring (ensembles)")
    a = sub.add_parser("author")
    a.add_argument("--model", required=True)
    a.add_argument("--train", required=True, choices=["IDRiD", "DDR"])
    a.add_argument("--sets")
    a.add_argument("--device")
    a.add_argument("--predict-only", action="store_true")
    a.add_argument("--drop-maps")
    e = sub.add_parser("evaluate", help="evaluate finished predictions again or for the first time (CPU)")
    e.add_argument("--source", required=True, help="directory name under runs/E3a")
    e.add_argument("--sets")
    e.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "plan":
        _plan(args)
    elif args.command == "run":
        run_source(args.run, _sets(args.sets), args.device, args.predict_only, _sets(args.drop_maps) or (), regenerate_maps=args.regenerate_maps)
    elif args.command == "author":
        run_author(args.model, args.train, _sets(args.sets), args.device, args.predict_only, _sets(args.drop_maps) or ())
    else:
        _evaluate(args)


if __name__ == "__main__":
    main()
