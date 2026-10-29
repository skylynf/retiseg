"""Shared helpers of the clinical error review (docs/clinical/28).

The review reads the benchmark's own files and writes nothing back into them:
    prepared dataset   dataset/prepared/<SET>/{images,masks,fov,ignore,splits.json}   (bench/common/io.py)
    probability maps   <pred_dir>/<CLS>/<id>.png, 16-bit, probability = value / 65535 (bench/common/io.py)
    thresholds         a metrics json of bench.eval (per_class[cls].thresholds.val_opt, else .fixed),
                       a release thresholds.json ({"thresholds": {cls: t}}), or a plain {cls: t} mapping
    lesion size edges  bench/eval/protocol.yaml (lesion_size_edges_rel_diameter, frozen 2026-10-04)

Random numbers come from numpy Generator(PCG64(seed)) as in docs/paper/14 M3.
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from bench.common import io as bio  # noqa: E402

CLASSES = tuple(bio.LESION_CLASSES)          # ("MA", "HE", "EX", "SE")
PROB_LEVELS = bio.PROB_LEVELS
PROTOCOL = REPO / "bench" / "eval" / "protocol.yaml"


def rng(seed):
    return np.random.Generator(np.random.PCG64(int(seed)))


def size_edges():
    """(edges, names) of the frozen lesion-size bins; edges are relative equivalent diameters."""
    import yaml

    proto = yaml.safe_load(PROTOCOL.read_text())
    edges = [float(e) for e in proto["lesion_size_edges_rel_diameter"]]
    return edges, list(proto["lesion_size_names"])


def size_bin(rel_diameter, edges, names):
    for i, name in enumerate(names):
        if edges[i] <= rel_diameter < edges[i + 1]:
            return name
    return names[-1]


def equivalent_diameter(area):
    return float(2.0 * np.sqrt(float(area) / np.pi))


def load_thresholds(spec):
    """{cls: threshold} and a description of where it came from.

    spec is a path to a bench.eval metrics json (val_opt threshold if present, else the fixed one), a release
    thresholds.json ({"thresholds": {...}}), a json file with a plain {cls: t} mapping, or a float written as text.
    """
    try:
        t = float(spec)
        return {c: t for c in CLASSES}, f"fixed {t}"
    except (TypeError, ValueError):
        pass
    path = Path(spec)
    payload = json.loads(path.read_text())
    if isinstance(payload, dict) and "per_class" in payload:
        out, kinds = {}, set()
        for c in CLASSES:
            th = payload["per_class"][c]["thresholds"]
            kind = "val_opt" if "val_opt" in th else "fixed"
            kinds.add(kind)
            out[c] = float(th[kind]["threshold"])
        return out, f"{path}: {'/'.join(sorted(kinds))} thresholds of bench.eval"
    if isinstance(payload, dict) and "thresholds" in payload:
        return {c: float(payload["thresholds"][c]) for c in CLASSES}, payload.get("source", str(path))
    if isinstance(payload, dict) and all(c in payload for c in CLASSES):
        return {c: float(payload[c]) for c in CLASSES}, str(path)
    raise ValueError(f"{spec}: no thresholds found")


def read_prob(pred_dir, cls, image_id):
    """Probability map in [0, 1] as float32 (bench.common.io quantization)."""
    return (bio.read_prob_quantized(pred_dir, cls, image_id).astype(np.float32) / PROB_LEVELS)


def read_case(data_dir, image_id):
    """FOV, ground truth and ignore masks of one image of a prepared source; all boolean arrays of one shape."""
    fov = bio.read_fov(data_dir, image_id)
    gt = {c: bio.read_mask(data_dir, c, image_id, shape=fov.shape) for c in CLASSES}
    ignore = {c: bio.read_ignore(data_dir, c, image_id, fov.shape) for c in CLASSES}
    for c in CLASSES:
        if gt[c].shape != fov.shape:
            raise ValueError(f"{data_dir} {image_id} {c}: mask {gt[c].shape} vs fov {fov.shape}")
    return fov, gt, ignore


def load_grades(path, set_name, data_dir=None):
    """{image id: grade} from runs/CLIN/grades.json ({set: {id: grade}}) or a flat {id: grade} json.

    path "auto": runs/CLIN/grades.json if it exists, otherwise the grades the paper's stratification reads
    (bench/scripts/clinical_strata.py): DDR from its release (DR_grading/<split>.txt), FGADR, Retinal-Lesions and
    MAPLES-DR from export_cases.jsonl of the prepared directory; IDRiD and TJDR carry no per-image grade here.
    """
    if not path:
        return {}
    if path == "auto":
        cached = REPO / "runs" / "CLIN" / "grades.json"
        if cached.is_file():
            return load_grades(str(cached), set_name)
        from bench.scripts import clinical_strata

        if set_name == "DDR":
            return {str(k): int(v) for k, v in clinical_strata.grades_ddr(REPO).items()}
        if data_dir is not None and (Path(data_dir) / "export_cases.jsonl").is_file():
            return {str(k): int(v) for k, v in clinical_strata.grades_from_prepared(data_dir).items()}
        return {}
    payload = json.loads(Path(path).read_text())
    if set_name in payload and isinstance(payload[set_name], dict):
        payload = payload[set_name]
    return {str(k): (int(v) if v is not None else None) for k, v in payload.items()}


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def grade_band(grade):
    if grade is None:
        return "ungraded"
    return "0-1" if int(grade) <= 1 else "2-4"
