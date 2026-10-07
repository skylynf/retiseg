"""Write original-resolution 16-bit probability maps for one checkpoint.

The network sees the field-of-view crop resized so its longer side is the
configured diameter, then card.forward_size when that is set. A B-std cell
keeps the diameter canvas and uses card.forward_size as a sliding window
instead (bench.infer). Probabilities
are bilinearly resized back onto the crop. Pixels outside the field of view
are 0. Files are written only through bench.common.io.write_prob.

Sigmoid cards apply sigmoid to the four lesion channels. Softmax cards
softmax over every channel, including background, then take card.class_index.
Those four planes are not renormalized.

    python -m bench.predict --config CONFIG --checkpoint CKPT --pred-dir DIR --split test
    python bench/predict.py --config CONFIG --checkpoint CKPT --pred-dir DIR --split test
"""

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
import yaml
from PIL import Image

from bench.common.io import LESION_CLASSES, load_split, write_prob
from bench.bstd import inference_for
from bench.data.b1_input import load_example, pad_to_multiple
from bench.infer import canvas_logits
from bench.models.registry import build_model
from bench.runtime import assert_not_finished_run, lesion_probabilities


def _resize_plane(plane, height, width):
    plane = np.ascontiguousarray(plane, dtype=np.float32)
    if plane.shape == (height, width):
        return np.clip(plane, 0.0, 1.0)
    resized = Image.fromarray(plane).resize((width, height), Image.Resampling.BILINEAR)
    return np.clip(np.asarray(resized, dtype=np.float32), 0.0, 1.0)


def predict_image(model, dataset_dir, image_id, diameter, pred_dir, device, inference=None):
    """``inference`` None is B1: the canvas, resized to card.forward_size when set.

    A dict with ``window`` and ``overlap`` is B-std: the diameter canvas,
    whole when window is None, otherwise overlapping windows of that size.
    """
    model.eval()
    card = model.card
    canvas = load_example(
        dataset_dir,
        image_id,
        diameter,
        card.normalization["mean"],
        card.normalization["std"],
        card.forward_size if inference is None else None,
        masks=False,
    )
    content_h, content_w = canvas["tensor"].shape[-2:]
    with torch.no_grad():
        if inference is None:
            tensor = pad_to_multiple(canvas["tensor"], card.pad_multiple).unsqueeze(0).to(device)
            logits = model(tensor)[:, :, :content_h, :content_w]
        else:
            logits = canvas_logits(
                model, canvas["tensor"].to(device), card, inference["window"], inference["overlap"]
            ).unsqueeze(0)
        prob = lesion_probabilities(logits, card)[0].detach().cpu().numpy()
    canvas_h, canvas_w = canvas["canvas_hw"]
    if (content_h, content_w) != (canvas_h, canvas_w):
        prob = np.stack([_resize_plane(plane, canvas_h, canvas_w) for plane in prob])
    y0, y1, x0, x1 = canvas["crop"]
    crop_h, crop_w = y1 - y0, x1 - x0
    placed = np.zeros((len(LESION_CLASSES),) + canvas["original_hw"], dtype=np.float32)
    for index, plane in enumerate(prob):
        placed[index, y0:y1, x0:x1] = _resize_plane(plane, crop_h, crop_w)
    outside = ~canvas["fov"]
    placed[:, outside] = 0.0
    for cls, plane in zip(LESION_CLASSES, placed):
        write_prob(pred_dir, cls, image_id, plane)


def predict(config_path, checkpoint, pred_dir, split):
    config = yaml.safe_load(Path(config_path).read_text())
    pred_dir = Path(pred_dir)
    assert_not_finished_run(pred_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config["model"]).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    if "model_name" in state and state["model_name"] != model.card.name:
        raise ValueError(f"checkpoint model {state['model_name']!r} != config model {model.card.name!r}")
    model.load_state_dict(state["model"])
    model.eval()
    root = Path(config["dataset"])
    diameter = int(config["fov_diameter"])
    inference = inference_for(config, model.card)
    with torch.no_grad():
        for image_id in load_split(root, split):
            predict_image(model, root, image_id, diameter, pred_dir, device, inference)
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
