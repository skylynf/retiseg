"""Run directories, records and evaluation of the E5 inference tiers (the layout of bench/e3a.py).

    runs/E5/<tier>/<source>/<set dir>/
        prob_test/  (prob_val/ for a pool run)   16-bit maps, bench.common.io.write_prob
        predicted.json      what ran: tier, model, checkpoint file and sha256, geometry, settings,
                            n_images, seconds, host, gpu, torch, git
        metrics.json, .npz  bench.eval under bench/eval/protocol.yaml
        done.json           predicted.json plus the summary, the thresholds used, maps_dropped

<tier> is a (zero-shot) or b (oracle prompts). <source> names the model and its
setting, e.g. a_sam_vith_p64, a_sam3_ma1, b_sam_vitb_box. <set dir> is the test
set plus variants, e.g. DDR, TJDR_uwf__fov, IDRiD__oracle_class, DDR__cap0.02,
IDRiD__k5. A pool run (D2) is the val split of IDRiD or DDR under <set>-val.

Thresholds (design/15 section 4 under D2): fixed 0.5 always; "val_opt" per
class is the Dice-optimal threshold of the pooled pixel histograms of the
source's two pool runs, computed once both exist and handed to
bench.eval.evaluate as val_thresholds, so the metric names stay those of every
other model. FGADR and Retinal-Lesions maps are removed after evaluation when
asked (drop_maps), as in E3a; a pool run never drops its maps.
"""

import hashlib
import json
import re
import shutil
import time
from functools import lru_cache
from pathlib import Path

from bench import e3a
from bench.e5 import frozen as fz
from bench.e5 import geometry as geo
from bench.eval import pixel
from bench.eval.evaluate import evaluate, load_protocol, pooled_histograms

REPO = fz.REPO
OUT_ROOT = REPO / "runs" / "E5"
TIERS = fz.TIERS
POOL_SUFFIX = "-val"
SLUG = {"SAM": "sam", "SAM 2": "sam2", "SAM 3": "sam3", "MedSAM": "medsam", "M2MRF": "m2mrf", "nnU-Net": "nnunet"}
_SOURCE_TOKEN = re.compile(r"[a-z0-9][a-z0-9.\-]*")  # no underscore: "_" joins the source name
_VARIANT_TOKEN = re.compile(r"[a-z0-9][a-z0-9._\-]*")  # single underscores allowed (oracle_class); "__" separates variants


def slug(model):
    if model not in SLUG:
        raise ValueError(f"no source slug for {model!r}; known {list(SLUG)}")
    return SLUG[model]


def source_name(tier, model, *tokens):
    """<tier>_<model slug>_<token>...; tokens are lower-case, e.g. ("vith", "p64") or ("box",)."""
    if tier not in TIERS:
        raise ValueError(f"tier must be one of {TIERS}, got {tier!r}")
    for token in tokens:
        if not _SOURCE_TOKEN.fullmatch(str(token)):
            raise ValueError(f"source token {token!r} must be lower-case letters, digits, '.' or '-'")
    return "_".join([tier, slug(model), *map(str, tokens)])


def set_dirname(set_name, variants=()):
    """<set>[__<variant>]...; the main result has no variant."""
    for variant in variants:
        if not _VARIANT_TOKEN.fullmatch(str(variant)) or "__" in str(variant):
            raise ValueError(f"variant {variant!r} must be a plain token")
    return set_name + "".join(f"__{variant}" for variant in variants)


def pool_set_dirname(set_name):
    return f"{set_name}{POOL_SUFFIX}"


def out_dir(tier, source, set_dir):
    if tier not in TIERS:
        raise ValueError(f"tier must be one of {TIERS}, got {tier!r}")
    return OUT_ROOT / tier / source / set_dir


def pred_dir(directory, split):
    return Path(directory) / ("prob_val" if fz.is_pool_split(split) else "prob_test")


def is_done(directory, limit=None):
    """True when the directory holds a finished evaluation that answers this request.

    A full request (``limit`` None) is answered only by a full run: the done.json
    of a smoke run (--limit) does not count, so the launcher queues the item again
    and the run evaluates it again (review fix 2, A5). A smoke request is answered
    by any finished evaluation.
    """
    path = Path(directory) / "done.json"
    if not path.is_file():
        return False
    return limit is not None or json.loads(path.read_text()).get("limit") is None


def needs_prediction(directory, limit=None):
    """True without predicted.json, or when a smoke run (limit set) is to be redone by a full run."""
    directory = Path(directory)
    if not (directory / "predicted.json").is_file():
        return True
    return limit is None and read_predicted(directory).get("limit") is not None


def environment():
    """host, gpu, torch, git of this process (bench.e3a.environment)."""
    return e3a.environment()


@lru_cache(maxsize=None)
def _sha256(path, size, mtime_ns):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_record(path):
    """{"file", "sha256", "bytes"} of a weights file (e5_frozen.yaml asks for the sha256 of every checkpoint)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    stat = path.stat()
    return {"file": path.name, "sha256": _sha256(str(path.resolve()), stat.st_size, stat.st_mtime_ns), "bytes": int(stat.st_size)}


EVALUATION_FILES = ("done.json", "metrics.json", "metrics.npz")


def write_predicted(directory, payload):
    """Record a finished prediction; an evaluation of earlier maps in the directory goes with it."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name in EVALUATION_FILES:  # new maps make the old metrics stale (a smoke run redone in full)
        (directory / name).unlink(missing_ok=True)
    (directory / "predicted.json").write_text(json.dumps(payload, indent=1) + "\n")


def read_predicted(directory):
    path = Path(directory) / "predicted.json"
    if not path.is_file():
        raise FileNotFoundError(f"{Path(directory)} has no finished prediction (predicted.json)")
    return json.loads(path.read_text())


def _pool_totals(tier, source, frozen, proto, pool=None, variants=()):
    """{cls: [pos, neg]} pixel histograms pooled over the source's pool runs, or None until all exist.

    A pool run made with --limit does not count as existing: a selection or a
    threshold never comes from the first N pool images.
    """
    pool = fz.pool_sets(frozen) if pool is None else list(pool)
    totals = None
    for name, data_dir, split in pool:
        directory = out_dir(tier, source, set_dirname(pool_set_dirname(name), variants))
        maps = pred_dir(directory, split)
        if not (directory / "predicted.json").is_file() or not maps.is_dir():
            return None
        if read_predicted(directory).get("limit") is not None:
            return None
        hist = pooled_histograms(data_dir, split, maps, proto)
        if totals is None:
            totals = {cls: [pos.copy(), neg.copy()] for cls, (pos, neg) in hist.items()}
        else:
            for cls, (pos, neg) in hist.items():
                totals[cls][0] += pos
                totals[cls][1] += neg
    return totals


def pool_thresholds(tier, source, frozen, proto, pool=None, variants=()):
    """{class: Dice-optimal threshold} on the pooled histograms of the source's pool runs, or None until all exist.

    ``variants`` selects the pool directories that match the test directory
    (e.g. ("cap0.02",) or ("oracle_class",)); ``pool`` overrides
    fz.pool_sets(frozen) as [(name, prepared dir, split)] (tests).
    """
    totals = _pool_totals(tier, source, frozen, proto, pool, variants)
    if totals is None:
        return None
    return {cls: float(pixel.best_dice_threshold(pos, neg, proto["prob_levels"])) for cls, (pos, neg) in totals.items()}


def thresholds_for_test(tier, source, frozen, proto, pool=None, variants=(), limit=None):
    """The pool thresholds a test-split evaluation must have; raises instead of falling back to fixed 0.5 only.

    ``limit`` is the prediction's own limit (predicted.json). A full prediction on a
    test split is never evaluated without the pool: the maps are already on disk
    when this raises, so the same command, repeated once the source's pool runs are
    finished, evaluates only. A smoke prediction (limit set) is the exception: it is
    redone in full later, so it takes what exists and done.json says "fixed only".
    (review fix 3, A3)
    """
    thresholds = pool_thresholds(tier, source, frozen, proto, pool, variants)
    if thresholds is None and limit is None:
        where = set_dirname("<set>" + POOL_SUFFIX, variants)
        raise RuntimeError(
            f"{source}: no finished full pool run under {out_dir(tier, source, where)}; a test split is evaluated with the "
            "pool's Dice thresholds (e5_frozen.yaml validation_pool), so run the pool first (launch_e5_infer.py --pool-only) "
            "and repeat this command, which then evaluates only"
        )
    return thresholds


def evaluation_scope(directory, all_ids):
    """(limit, excluded ids) of the prediction in ``directory``: the evaluation covers what was predicted.

    The scope comes from predicted.json, not from the request, so a smoke request
    on a fully predicted directory evaluates every image, and a full request on a
    smoke directory does not happen (needs_prediction redoes the prediction first).
    """
    limit = read_predicted(directory).get("limit")
    ids = list(all_ids) if limit is None else list(all_ids)[: int(limit)]
    return limit, list(all_ids)[len(ids):]


def pool_class_aupr(tier, source, frozen, proto, pool=None, variants=()):
    """{class: exact AUPR} on the pooled pixel histograms of the source's pool runs, or None until all exist."""
    totals = _pool_totals(tier, source, frozen, proto, pool, variants)
    if totals is None:
        return None
    return {cls: float(pixel.average_precision(pos, neg)) for cls, (pos, neg) in totals.items()}


def evaluate_dir(directory, set_name, split="test", data_dir=None, proto=None, val_thresholds=None, drop_maps=(), record=None, exclude=()):
    """bench.eval on one finished prediction directory; writes metrics.json, metrics.npz and done.json.

    ``val_thresholds`` are the pool thresholds (None: fixed 0.5 only). A test
    set named in ``drop_maps`` loses its maps after the metrics are written;
    a pool run keeps them. ``exclude`` lists split ids left out (a smoke run
    that predicted only the first images).
    """
    directory = Path(directory)
    predicted = read_predicted(directory)
    proto = proto if proto is not None else load_protocol(REPO / "bench" / "eval" / "protocol.yaml")
    data_dir = Path(data_dir) if data_dir is not None else geo.data_dir(set_name)
    maps = pred_dir(directory, split)
    started = time.perf_counter()
    result = evaluate(data_dir, split, maps, proto, out_prefix=directory / "metrics", val_thresholds=val_thresholds, exclude=exclude)
    drop = (set_name in set(drop_maps or ())) and not fz.is_pool_split(split)
    done = dict(predicted)
    done.update(record or {})
    done.update(
        {
            "thresholds": "fixed and pool val_opt" if val_thresholds is not None else "fixed only",
            "val_thresholds": None if val_thresholds is None else {cls: float(v) for cls, v in val_thresholds.items()},
            "eval_seconds": round(time.perf_counter() - started, 1),
            "excluded": sorted(exclude or ()),
            "summary": result["summary"],
            "maps_dropped": drop,
        }
    )
    (directory / "done.json").write_text(json.dumps(done, indent=1) + "\n")
    if drop:
        shutil.rmtree(maps)
    any_aupr = result["summary"].get("AUPR_any_lesion")
    print(f"{directory.parent.name} {directory.name} mAUPR {result['summary']['mAUPR']:.4f} any {any_aupr if any_aupr is None else round(any_aupr, 4)}", flush=True)
    return result
