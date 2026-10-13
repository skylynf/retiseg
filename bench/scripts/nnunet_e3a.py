"""nnU-Net checkpoints on the E3a test sets (bench/configs/e3a_frozen.yaml).

nnU-Net trained at original resolution, so its training scale is the median
field-of-view long side of its training split (IDRiD about 3413, DDR about
1658). Each test image is resized as a whole so that its field-of-view box
has that long side (pixels per degree for TJDR_uwf, see bench.e3a), passed
through nnU-Net's own predictor with the nnUNetv2_predict defaults (step 0.5,
Gaussian weighting, mirroring), and the softmax probabilities are resized
back to the original image, set to 0 outside the field of view and written
with write_prob. checkpoint_final.pth, fold 0, as for the internal test.
Evaluation and done.json are bench.e3a's.

    CUDA_VISIBLE_DEVICES=4 python bench/scripts/nnunet_e3a.py --train IDRiD --plan default [--sets TJDR_std]
    (--train TJDR_std or FGADR for the E3b sources; their same-eye exclusions come from leak_exclusions.json)

Runs in the retiseg-nnunet environment.
"""

import argparse
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
from PIL import Image

from bench import e3a
from bench.common.io import LESION_CLASSES, image_path, read_fov, write_prob
from bench.data.fov import crop_box
from bench.scripts.nnunet_bridge import DATASET_IDS, LABELS

PLANS = {"default": "nnUNetPlans", "resencm": "nnUNetResEncUNetMPlans"}
# nnUNetv2_predict defaults (nnunetv2/inference/predict_from_raw_data.py, predict_entry_point).
PREDICTOR = {"tile_step_size": 0.5, "use_gaussian": True, "use_mirroring": True, "perform_everything_on_device": True}
# NaturalImage2DIO.read_images returns this spacing for every PNG.
SPACING = (999, 1, 1)


def model_folder(train, plan):
    return _ROOT / "runs" / "nnunet" / f"Dataset{DATASET_IDS[train]:03d}_{train}" / f"nnUNetTrainer__{PLANS[plan]}__2d"


def source_name(train, plan):
    return f"NN_{plan}_{train.lower()}_fold0"


def scaled_image(data_dir, image_id, long_side):
    """Whole RGB image resized so its field-of-view box has ``long_side``; returns array, fov, factor."""
    rgb = Image.open(image_path(data_dir, image_id)).convert("RGB")
    fov = read_fov(data_dir, image_id)
    y0, y1, x0, x1 = crop_box(fov)
    factor = float(long_side) / max(y1 - y0, x1 - x0)
    width, height = rgb.size
    size = (max(1, int(round(width * factor))), max(1, int(round(height * factor))))
    if size != (width, height):
        rgb = rgb.resize(size, Image.Resampling.BILINEAR)
    return np.asarray(rgb), fov, factor


def nnunet_input(rgb):
    """(3, 1, H, W) float32, the array NaturalImage2DIO gives for an RGB PNG."""
    return np.ascontiguousarray(rgb.transpose(2, 0, 1)[:, None]).astype(np.float32)


def resize_plane(plane, height, width):
    plane = np.ascontiguousarray(plane, dtype=np.float32)
    if plane.shape == (height, width):
        return plane
    return np.asarray(Image.fromarray(plane, mode="F").resize((width, height), Image.Resampling.BILINEAR))


def lesion_planes(probabilities, height, width, fov):
    """Softmax channels (labels from nnunet_bridge.LABELS) to four original-size planes."""
    probabilities = np.asarray(probabilities)
    if probabilities.ndim == 4:
        if probabilities.shape[1] != 1:
            raise ValueError(f"expected one 2-D slice, got {probabilities.shape}")
        probabilities = probabilities[:, 0]
    if probabilities.shape[0] != len(LABELS):
        raise ValueError(f"{probabilities.shape[0]} channels, labels are {LABELS}")
    out = {}
    for cls in LESION_CLASSES:
        plane = np.clip(resize_plane(probabilities[LABELS[cls]], height, width).astype(np.float64), 0.0, 1.0)
        plane[~fov] = 0.0
        out[cls] = plane
    return out


def build_predictor(train, plan, device, settings=None):
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    if device.type == "cuda":
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    settings = dict(PREDICTOR if settings is None else settings)
    predictor = nnUNetPredictor(device=device, verbose=False, verbose_preprocessing=False, allow_tqdm=False, **settings)
    predictor.initialize_from_trained_model_folder(str(model_folder(train, plan)), (0,), "checkpoint_final.pth")
    return predictor


def run(train, plan, sets=None, device_name=None, predict_only=False, on_device=True, drop_maps=()):
    import torch

    frozen = e3a.load_frozen()
    source = source_name(train, plan)
    run_dir = _ROOT / "runs" / source
    if not (model_folder(train, plan) / "fold_0" / "checkpoint_final.pth").is_file():
        raise FileNotFoundError(f"{model_folder(train, plan)}/fold_0/checkpoint_final.pth")
    # a training source without a frozen field of view (TJDR_std, FGADR; E3b) has no per-degree scale for the
    # ultra-widefield set: only its field-of-view variant runs, as bench.e3a does for the BF cells
    no_degrees = train not in frozen["train_field_degrees"]
    pending = [(t, v) for t, v in e3a.targets([train], frozen, sets)
               if not e3a.is_done(source, t, v) and not (no_degrees and t in frozen.get("per_degree_sets", ()) and v == e3a.MAIN)]
    if not pending:
        print(f"{source}: every E3a set is done", flush=True)
        return
    train_dir = _ROOT / frozen["test_sets"][train]["data"]
    reference = e3a._median_fov_long_side(train_dir, "train")
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    predictor = None
    # perform_everything_on_device only moves the sliding-window sums to the CPU (nnUNetv2_predict --not_on_device).
    settings = dict(PREDICTOR, perform_everything_on_device=bool(on_device))
    env = e3a.environment()
    for test_name, variant in pending:
        directory = e3a.out_dir(source, test_name, variant)
        if not (directory / "predicted.json").is_file():
            if predictor is None:
                predictor = build_predictor(train, plan, device, settings)
            long_side = e3a.target_long_side(test_name, variant, train, reference, frozen)
            ids, excluded = e3a.test_ids(test_name, frozen, [train])
            data_dir = e3a.test_data_dir(test_name, frozen)
            factors = []
            started = time.perf_counter()
            for index, image_id in enumerate(ids):
                rgb, fov, factor = scaled_image(data_dir, image_id, long_side)
                factors.append(factor)
                _seg, probabilities = predictor.predict_single_npy_array(
                    nnunet_input(rgb), {"spacing": SPACING}, None, None, True
                )
                for cls, plane in lesion_planes(probabilities, fov.shape[0], fov.shape[1], fov).items():
                    write_prob(directory / "prob_test", cls, image_id, plane)
                if index % 50 == 0:
                    print(f"{source} {e3a.set_dirname(test_name, variant)} {index + 1}/{len(ids)}", flush=True)
            e3a._write_predicted(
                directory,
                {
                    "source": source,
                    "kind": "nnU-Net",
                    "model": f"nnU-Net {plan}",
                    "train_source": train,
                    "checkpoint": str(model_folder(train, plan).relative_to(_ROOT) / "fold_0" / "checkpoint_final.pth"),
                    "test_set": test_name,
                    "variant": variant,
                    "reference_fov_long_side": reference,
                    "target_fov_long_side": long_side,
                    "scale_factor_median": float(np.median(factors)) if factors else None,
                    "predictor": settings,
                    "n_images": len(ids),
                    "excluded": excluded,
                    "predict_seconds": round(time.perf_counter() - started, 1),
                    **env,
                },
            )
        if not predict_only:
            e3a.evaluate_set(
                source, test_name, variant, frozen, train_dir=train_dir, val_pred_dir=run_dir / "prob_val",
                drop_maps=drop_maps,
            )


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, choices=sorted(DATASET_IDS))
    parser.add_argument("--plan", required=True, choices=sorted(PLANS))
    parser.add_argument("--sets")
    parser.add_argument("--device")
    parser.add_argument("--predict-only", action="store_true")
    parser.add_argument("--not-on-device", action="store_true", help="sum the windows on the CPU, as nnUNetv2_predict --not_on_device")
    parser.add_argument("--drop-maps", help="comma-separated test sets whose maps are removed after evaluation")
    args = parser.parse_args(argv)
    run(
        args.train, args.plan, e3a._sets(args.sets), args.device, args.predict_only,
        on_device=not args.not_on_device, drop_maps=e3a._sets(args.drop_maps) or (),
    )


if __name__ == "__main__":
    main()
