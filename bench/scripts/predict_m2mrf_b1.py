"""Write original-resolution 16-bit probability maps for one B1 M2MRF checkpoint.

The network sees the field-of-view crop whose longer side is the configured
diameter. Four sigmoid channels are MA, HE, EX, SE. Probabilities are
bilinearly resized back onto that crop. Pixels outside the field of view are
0. Files are written only through bench.common.io.write_prob.

    python bench/scripts/predict_m2mrf_b1.py --config CONFIG --checkpoint CKPT --pred-dir DIR --split test
"""

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
import yaml
from PIL import Image

from bench.bstd import inference_for
from bench.common.io import LESION_CLASSES, load_split, write_prob
from bench.data.b1_input import load_example, pad_to_multiple
from bench.models.m2mrf import M2MRF, bind_normalization, refuse_author_runs
from bench.runtime import lesion_probabilities


def _resize_plane(plane, height, width):
    plane = np.ascontiguousarray(plane, dtype=np.float32)
    if plane.shape == (height, width):
        return np.clip(plane, 0.0, 1.0)
    resized = Image.fromarray(plane).resize((width, height), Image.Resampling.BILINEAR)
    return np.clip(np.asarray(resized, dtype=np.float32), 0.0, 1.0)


def write_image(model, dataset_dir, image_id, diameter, pred_dir, device):
    """One prepared image in, four original-resolution maps out."""
    if model.network is None:
        load_example(
            dataset_dir,
            image_id,
            diameter,
            model.card.normalization["mean"],
            model.card.normalization["std"],
            model.card.forward_size,
            masks=False,
        )
        raise RuntimeError(model.import_error or "M2MRF network was not imported")
    model.eval()
    card = model.card
    canvas = load_example(
        dataset_dir,
        image_id,
        diameter,
        card.normalization["mean"],
        card.normalization["std"],
        card.forward_size,
        masks=False,
    )
    content_h, content_w = canvas["tensor"].shape[-2:]
    tensor = pad_to_multiple(canvas["tensor"], card.pad_multiple).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)[:, :, :content_h, :content_w]
        prob = lesion_probabilities(logits, card)[0].detach().cpu().numpy()
    canvas_h, canvas_w = canvas["canvas_hw"]
    if (content_h, content_w) != (canvas_h, canvas_w):
        prob = np.stack([_resize_plane(plane, canvas_h, canvas_w) for plane in prob])
    y0, y1, x0, x1 = canvas["crop"]
    crop_h, crop_w = y1 - y0, x1 - x0
    placed = np.zeros((len(LESION_CLASSES),) + canvas["original_hw"], dtype=np.float32)
    for index, plane in enumerate(prob):
        placed[index, y0:y1, x0:x1] = _resize_plane(plane, crop_h, crop_w)
    placed[:, ~canvas["fov"]] = 0.0
    for cls, plane in zip(LESION_CLASSES, placed):
        write_prob(pred_dir, cls, image_id, plane)


def predict(config_path, checkpoint, pred_dir, split):
    config = yaml.safe_load(Path(config_path).read_text())
    pred_dir = Path(pred_dir)
    refuse_author_runs(pred_dir)
    if tuple(LESION_CLASSES) != ("MA", "HE", "EX", "SE"):
        raise RuntimeError("lesion order must stay MA, HE, EX, SE")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = M2MRF().to(device)
    bind_normalization(model, Path(config["dataset"]).name)
    root = Path(config["dataset"])
    diameter = int(config["fov_diameter"])
    inference = inference_for(config, model.card)
    if inference is not None and inference["window"] is not None:
        raise RuntimeError("M2MRF is fully convolutional; a B-std window would need bench.infer here")
    image_ids = load_split(root, split)
    if model.network is None:
        load_example(
            root,
            image_ids[0],
            diameter,
            model.card.normalization["mean"],
            model.card.normalization["std"],
            model.card.forward_size,
            masks=False,
        )
        raise RuntimeError(model.import_error or "M2MRF network was not imported")
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    if "model_name" in state and state["model_name"] != model.card.name:
        raise ValueError(f"checkpoint model {state['model_name']!r} != config model {model.card.name!r}")
    model.load_state_dict(state["model"])
    model.eval()
    for image_id in image_ids:
        write_image(model, root, image_id, diameter, pred_dir, device)
        print(image_id, flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--split", default="test")
    args = parser.parse_args(argv)
    predict(args.config, args.checkpoint, args.pred_dir, args.split)


if __name__ == "__main__":
    main()
