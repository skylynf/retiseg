"""RetiSeg-Ref, the reference model (docs/paper/26 section 5 item 2; docs/paper/14 note 58), on new images.

The reference model is the pixel mean of two factor cells trained on IDRiD + DDR + TJDR_std merged with
the DDR recipe, canvas and pixel budget, validated on DDR: runs/BF_ref_retfound_fine_seed0 (RETFound +
UPerNet) and runs/BF_ref_samed_fine_seed0 (SAMed). Its scores on the eight test sets are
runs/ENS/REF/reference/<set> (bench/scripts/ensemble.py reference). This module runs the model on fundus
images that are not in a prepared source, and packs the release.

    python -m bench.reference predict --images DIR_OR_FILES... --out DIR [--release release/retiseg_ref | --members a,b]
    python -m bench.reference bundle --out release/retiseg_ref [--members a,b] [--scores runs/ENS/REF/reference]

predict: the field of view by the project rule (bench.data.fov: luminance above the border, largest region,
holes filled, the threshold raised above a flat grey border), the field-of-view crop resized to the
1,440-pixel canvas, each member's B-std sliding-window inference (bench.predict.predict_image), the two
probability maps averaged and written at the original resolution as 16-bit PNG (out/prob/<cls>/<id>.png,
bench.common.io.write_prob), binary masks at the release thresholds (out/masks/<cls>/<id>.png; the
thresholds were chosen on the DDR validation split of the ensemble, never on a test set), and
out/summary.json with each image's maximum probability inside the field of view (the image-level score of
the protocol) and the thresholds used. No ground truth is read and nothing is scored; the member maps stay
in out/members/<member>/ so a user can check each one.

bundle: the release directory -- each member's primary checkpoint (checkpoint_roles.json: best.pt, the
validation-selected one) as weights.pt with its config.yaml, recipe.json and environment.json; thresholds.json
(from the reference ensemble's DDR metrics; 0.5 for every class when no ensemble has been scored, and
thresholds.json says which); scores.json (the ensemble's and the members' mAUPR on every scored set);
members.json (the registration of 14 note 58); README.md; sha256sums.txt. Weights of DDR- and TJDR-trained
models are redistributed only after their licences allow it (docs/paper/14 M14).
"""

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import yaml
from PIL import Image

from bench.common import io
from bench.common.io import LESION_CLASSES, write_prob

MEMBERS = ("runs/BF_ref_retfound_fine_seed0", "runs/BF_ref_samed_fine_seed0")
SCORES = "runs/ENS/REF/reference"
NAME = "RetiSeg-Ref"
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
FIXED_THRESHOLD = 0.5


# ---- images that are not in a prepared source -------------------------------------


def list_images(items):
    """Image files from directories (sorted, one level) and file paths, in that order; duplicates by stem refused."""
    paths = []
    for item in items:
        item = Path(item)
        if item.is_dir():
            paths.extend(sorted(p for p in item.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES and p.is_file()))
        elif item.is_file():
            paths.append(item)
        else:
            raise FileNotFoundError(item)
    if not paths:
        raise FileNotFoundError("no image files among the given paths")
    stems = [p.stem for p in paths]
    repeated = sorted({s for s in stems if stems.count(s) > 1})
    if repeated:
        raise ValueError(f"image names must be unique without their suffix; repeated: {repeated}")
    return paths


def prepare_images(paths, work_dir):
    """A prepared-source directory (images/, fov/, splits.json {"all": ids}) for the files, with the project's
    field-of-view rule. Returns the ids. The image bytes are copied, not re-encoded."""
    from bench.data.fov import border_threshold, field_of_view

    work_dir = Path(work_dir)
    (work_dir / "images").mkdir(parents=True, exist_ok=True)
    (work_dir / "fov").mkdir(parents=True, exist_ok=True)
    ids, thresholds = [], {}
    for path in paths:
        image = np.asarray(Image.open(path).convert("RGB"))
        threshold = border_threshold(image)
        fov = field_of_view(image, threshold)
        shutil.copyfile(path, work_dir / "images" / f"{path.stem}{path.suffix.lower()}")
        Image.fromarray(np.asarray(fov, dtype=np.uint8) * 255).save(work_dir / "fov" / f"{path.stem}.png")
        ids.append(path.stem)
        thresholds[path.stem] = float(threshold)
    (work_dir / "splits.json").write_text(json.dumps({"all": ids}, indent=1) + "\n")
    (work_dir / "fov_thresholds.json").write_text(json.dumps(thresholds, indent=1) + "\n")
    return ids


def member_name(run_dir):
    return Path(run_dir).name


def member_config(run_dir, work_dir):
    """The member's cell config pointed at the prepared directory of new images (split "all", no training list
    is read at inference; the canvas, geometry and normalization come from the cell)."""
    run_dir = Path(run_dir)
    config = yaml.safe_load((run_dir / "config.yaml").read_text())
    config["dataset"] = str(work_dir)
    config["split_test"] = "all"
    config["split_val"] = "all"
    return config


def primary_checkpoint(run_dir):
    run_dir = Path(run_dir)
    roles = run_dir / "checkpoint_roles.json"
    name = json.loads(roles.read_text())["primary"] if roles.is_file() else "best.pt"
    path = run_dir / name
    if not path.is_file():
        path = run_dir / "weights.pt"   # a bundled member
    if not path.is_file():
        raise FileNotFoundError(f"{run_dir}: no {name} and no weights.pt")
    return path


def predict_member(run_dir, work_dir, ids, out_dir, device=None):
    """The member's maps for every id, at the original resolution, by its own B-std inference geometry."""
    import torch

    from bench.bstd import inference_for
    from bench.e3a import load_run_model
    from bench.predict import predict_image

    run_dir = Path(run_dir)
    config = member_config(run_dir, work_dir)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_run_model(config, primary_checkpoint(run_dir), device)
    inference = inference_for(config, model.card)
    diameter = int(config["fov_diameter"])
    started = time.perf_counter()
    with torch.no_grad():
        for index, image_id in enumerate(ids, start=1):
            predict_image(model, Path(work_dir), image_id, diameter, Path(out_dir), device, inference)
            print(f"{member_name(run_dir)} {index}/{len(ids)} {image_id}", flush=True)
    return time.perf_counter() - started


# ---- averaging, thresholds, summary ------------------------------------------------


def average_members(member_dirs, ids, out_dir):
    from bench.scripts.ensemble import average_maps

    return average_maps([Path(d) for d in member_dirs], ids, LESION_CLASSES, Path(out_dir))


def load_thresholds(path_or_dir):
    """{cls: threshold} from a release thresholds.json, or from the reference ensemble's DDR metrics.json
    (per_class[cls].thresholds.val_opt.threshold). Returns (thresholds, source)."""
    path = Path(path_or_dir)
    if path.is_dir():
        for candidate in (path / "thresholds.json", path / "DDR" / "metrics.json"):
            if candidate.is_file():
                path = candidate
                break
    if not path.is_file():
        return {cls: FIXED_THRESHOLD for cls in LESION_CLASSES}, "fixed 0.5 (no scored reference ensemble)"
    payload = json.loads(path.read_text())
    if "thresholds" in payload and all(cls in payload["thresholds"] for cls in LESION_CLASSES):
        return {cls: float(payload["thresholds"][cls]) for cls in LESION_CLASSES}, payload.get("source", str(path))
    if "per_class" in payload:
        out = {cls: float(payload["per_class"][cls]["thresholds"]["val_opt"]["threshold"]) for cls in LESION_CLASSES}
        return out, f"{path}: validation-optimal Dice thresholds of the reference ensemble on the DDR validation split"
    raise ValueError(f"{path} holds neither a thresholds block nor per_class metrics")


def binarize_and_summarize(prob_dir, work_dir, ids, thresholds, masks_dir):
    """Binary masks at the thresholds, and per image the maximum probability inside the field of view and the
    positive pixel count per class. ``work_dir`` is the prepared directory (its fov/<id>.png is read)."""
    prob_dir, work_dir, masks_dir = Path(prob_dir), Path(work_dir), Path(masks_dir)
    images = {}
    for image_id in ids:
        fov = io.read_fov(work_dir, image_id)
        entry = {}
        for cls in LESION_CLASSES:
            prob = io.read_prob_quantized(prob_dir, cls, image_id).astype(np.float64) / io.PROB_LEVELS
            inside = prob[fov]
            mask = (prob >= thresholds[cls]) & fov
            (masks_dir / cls).mkdir(parents=True, exist_ok=True)
            Image.fromarray(mask.astype(np.uint8) * 255).save(masks_dir / cls / f"{image_id}.png")
            entry[cls] = {
                "max_probability_in_fov": float(inside.max()) if inside.size else 0.0,
                "positive_pixels": int(mask.sum()),
                "positive_fraction_of_fov": float(mask.sum() / max(int(fov.sum()), 1)),
            }
        images[image_id] = entry
    return images


def predict(images, out_dir, members=MEMBERS, release=None, device=None, keep_work=False):
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"{out_dir} is not empty")
    members = [Path(m) for m in (members or MEMBERS)]
    if release is not None:
        release = Path(release)
        members = [release / member_name(m) for m in members]
    for run_dir in members:
        if not (run_dir / "config.yaml").is_file():
            raise FileNotFoundError(f"{run_dir}/config.yaml")
    paths = list_images(images)
    work_dir = out_dir / "prepared"
    ids = prepare_images(paths, work_dir)
    seconds = {}
    member_dirs = []
    for run_dir in members:
        directory = out_dir / "members" / member_name(run_dir)
        seconds[member_name(run_dir)] = predict_member(run_dir, work_dir, ids, directory, device)
        member_dirs.append(directory)
    average_members(member_dirs, ids, out_dir / "prob")
    thresholds, threshold_source = load_thresholds(release if release is not None else _ROOT / SCORES)
    per_image = binarize_and_summarize(out_dir / "prob", work_dir, ids, thresholds, out_dir / "masks")
    summary = {
        "model": NAME,
        "members": [str(m) for m in members],
        "checkpoints": [str(primary_checkpoint(m)) for m in members],
        "canvas_long_side": int(member_config(members[0], work_dir)["fov_diameter"]),
        "classes": list(LESION_CLASSES),
        "thresholds": thresholds,
        "thresholds_source": threshold_source,
        "field_of_view": "bench.data.fov (luminance threshold raised above a flat grey border; largest region; holes filled)",
        "maps": "prob/<class>/<id>.png, 16-bit, probability x 65535, original resolution, 0 outside the field of view; "
                "masks/<class>/<id>.png at the thresholds; members/<member>/ the two members' maps",
        "n_images": len(ids),
        "predict_seconds": seconds,
        "images": per_image,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    if not keep_work:
        shutil.rmtree(work_dir / "images", ignore_errors=True)
    print(f"{NAME}: {len(ids)} images -> {out_dir}", flush=True)
    return summary


# ---- the release directory ------------------------------------------------------------


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_scores(scores_root, members):
    """mAUPR of the reference ensemble and of each member on every set with a metrics.json: ensemble from
    <scores_root>/<set>/metrics.json, members from their in-source metrics (metrics_test.json for the DDR
    test split, insource_<set>/metrics.json for IDRiD and TJDR_std) and runs/E3a/<member>/<set>[__fov]/metrics.json."""
    scores_root = Path(scores_root)
    out = {"ensemble": {}, "members": {}}
    if scores_root.is_dir():
        for setdir in sorted(scores_root.iterdir()):
            path = setdir / "metrics.json"
            if path.is_file():
                out["ensemble"][setdir.name] = _summary(path)
    for run_dir in members:
        run_dir = Path(run_dir)
        name = member_name(run_dir)
        rows = {}
        if (run_dir / "metrics_test.json").is_file():
            rows[Path(yaml.safe_load((run_dir / "config.yaml").read_text())["dataset"]).name] = _summary(run_dir / "metrics_test.json")
        for insource in sorted(run_dir.glob("insource_*")):
            path = insource / "metrics.json"
            if path.is_file():
                rows[insource.name[len("insource_"):]] = _summary(path)
        e3a_root = _ROOT / "runs" / "E3a" / name
        if e3a_root.is_dir():
            for setdir in sorted(e3a_root.iterdir()):
                path = setdir / "metrics.json"
                if path.is_file():
                    rows[setdir.name] = _summary(path)
        out["members"][name] = rows
    return out


def _summary(path):
    summary = json.loads(Path(path).read_text()).get("summary", {})
    return {key: summary[key] for key in ("mAUPR", "mDice_pooled_val_opt", "AUPR_any_lesion", "mAUC_image_max") if key in summary}


def bundle(out_dir, members=MEMBERS, scores_root=SCORES):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    members = [Path(m) for m in (members or MEMBERS)]
    copied = []
    for run_dir in members:
        checkpoint = primary_checkpoint(run_dir)
        dest = out_dir / member_name(run_dir)
        dest.mkdir(exist_ok=True)
        shutil.copyfile(checkpoint, dest / "weights.pt")
        for name in ("config.yaml", "recipe.json", "environment.json", "checkpoint_roles.json"):
            if (run_dir / name).is_file():
                shutil.copyfile(run_dir / name, dest / name)
        copied.append({"member": member_name(run_dir), "model": yaml.safe_load((run_dir / "config.yaml").read_text())["model"],
                       "checkpoint": checkpoint.name, "weights": f"{member_name(run_dir)}/weights.pt"})
    thresholds, source = load_thresholds(_ROOT / scores_root if not Path(scores_root).is_absolute() else Path(scores_root))
    (out_dir / "thresholds.json").write_text(json.dumps({"source": source, "thresholds": thresholds}, indent=1) + "\n")
    scores = collect_scores(_ROOT / scores_root if not Path(scores_root).is_absolute() else Path(scores_root), members)
    (out_dir / "scores.json").write_text(json.dumps(scores, indent=1) + "\n")
    (out_dir / "members.json").write_text(json.dumps({
        "model": NAME,
        "rule": "the two main-table models with the highest DDR validation mAUPR (RETFound + UPerNet 0.6432, SAMed 0.6319; "
                "docs/paper/14 note 58, registered 2026-10-23 before any merged run), each trained once on IDRiD + DDR + "
                "TJDR_std merged (599 images) with its DDR recipe, canvas (1,440) and pixel budget, validated on DDR; "
                "the probability maps are averaged pixel by pixel",
        "members": copied,
        "combination": "pixel mean of the two members' probability maps",
        "thresholds": "thresholds.json (validation-optimal Dice thresholds on the DDR validation split of the ensemble)",
        "usage": "python -m bench.reference predict --images DIR --out OUT --release <this directory>",
    }, indent=1) + "\n")
    (out_dir / "README.md").write_text(_readme(copied, thresholds, source, scores))
    sums = []
    for path in sorted(out_dir.rglob("*")):
        if path.is_file() and path.name != "sha256sums.txt":
            sums.append(f"{_sha256(path)}  {path.relative_to(out_dir)}")
    (out_dir / "sha256sums.txt").write_text("\n".join(sums) + "\n")
    print(f"{NAME} release in {out_dir}: {len(copied)} members, {len(scores['ensemble'])} scored sets", flush=True)
    return out_dir


def _readme(copied, thresholds, source, scores):
    lines = [
        f"# {NAME}",
        "",
        "The reference model of the RetiSeg assessment (docs/paper/14 note 58): two models trained once on IDRiD + DDR +",
        "TJDR_std merged with their DDR recipe, canvas and pixel budget, their probability maps averaged pixel by pixel.",
        "",
        "## Members",
        "",
    ]
    for item in copied:
        lines.append(f"- `{item['member']}` ({item['model']}): `{item['weights']}`, the {item['checkpoint']} checkpoint of the run; "
                     f"`config.yaml`, `recipe.json`, `environment.json` beside it.")
    lines += [
        "",
        "## Run it",
        "",
        "```",
        "python -m bench.reference predict --images DIR_OR_FILES --out OUT --release <this directory>",
        "```",
        "",
        "Writes `OUT/prob/<class>/<id>.png` (16-bit probability maps at the original resolution, 0 outside the field of view),",
        "`OUT/masks/<class>/<id>.png` (binary masks at the thresholds below), `OUT/members/` (each member's maps) and",
        "`OUT/summary.json` (per image, the maximum probability inside the field of view and the positive area). Classes:",
        "MA, HE, EX, SE. The inference geometry is each member's B-std one (field-of-view crop resized to a 1,440-pixel",
        "canvas, sliding windows). Needs the `retiseg` environment (bench/requirements-train.txt).",
        "",
        "## Thresholds",
        "",
        f"{source}.",
        "",
    ]
    lines += [f"- {cls}: {thresholds[cls]:.4f}" for cls in LESION_CLASSES]
    lines += ["", "## Scores (mAUPR, exact AUPR over field-of-view pixels, mean of four classes)", ""]
    if scores["ensemble"]:
        lines.append("| test set | ensemble | " + " | ".join(item["member"] for item in copied) + " |")
        lines.append("|---|---|" + "---|" * len(copied))
        for name, entry in scores["ensemble"].items():
            row = [name, f"{entry.get('mAUPR', float('nan')):.3f}"]
            for item in copied:
                member = scores["members"].get(item["member"], {})
                value = member.get(name, member.get(f"{name}__fov", {})).get("mAUPR")
                row.append("" if value is None else f"{value:.3f}")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
        lines.append("IDRiD, DDR and TJDR_std are internal tests (their training splits were used); the other sets are independent.")
    else:
        lines.append("No scored reference ensemble was found when this release was packed (runs/ENS/REF/reference).")
    lines += ["", "## Files", "", "`sha256sums.txt` lists every file. `members.json` records the registration and the combination rule.", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("predict", help="run the reference model on images")
    p.add_argument("--images", nargs="+", required=True, help="directories and/or image files")
    p.add_argument("--out", required=True)
    p.add_argument("--members", default=",".join(MEMBERS), help="comma-separated run directories")
    p.add_argument("--release", default=None, help="a bundled release directory; the members are read from it")
    p.add_argument("--device", default=None)
    p.add_argument("--keep-work", action="store_true", help="keep the prepared copies of the images")
    b = sub.add_parser("bundle", help="pack the release directory")
    b.add_argument("--out", required=True)
    b.add_argument("--members", default=",".join(MEMBERS))
    b.add_argument("--scores", default=SCORES, help="the reference ensemble's directory")
    args = parser.parse_args(argv)
    members = [m.strip() for m in args.members.split(",") if m.strip()]
    if args.command == "predict":
        predict(args.images, args.out, members=members, release=args.release, device=args.device, keep_work=args.keep_work)
    else:
        bundle(args.out, members=members, scores_root=args.scores)


if __name__ == "__main__":
    main()
