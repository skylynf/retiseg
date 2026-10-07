"""Write HACDR-Net probability maps at the original image size.

The head stays the author's 5-class softmax. FGADRDataset.CLASSES is
background, EX, MA, SE, HE with reduce_zero_label=False, so the maps are
taken from channels 2, 4, 1, 3 and written MA, HE, EX, SE through write_prob.
Pixels outside the field of view are 0. One process, one GPU.

    python bench/scripts/predict_hacdr_b1.py --config CONFIG --checkpoint CKPT --pred-dir DIR --split test
"""

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
import yaml

from bench.bstd import inference_for
from bench.common.io import LESION_CLASSES, load_split
from bench.models.hacdr import FGADR_CLASSES, lesion_class_index
from bench.models.registry import build_model
from bench.predict import predict_image
from bench.runtime import assert_not_finished_run, refuse_distributed

CLASS_INDEX = (2, 4, 1, 3)


def _check_channels(model):
    if FGADR_CLASSES != ("background", "EX", "MA", "SE", "HE"):
        raise RuntimeError("FGADRDataset class list changed; lesion channels are not guessed")
    if tuple(model.card.class_index) != CLASS_INDEX or lesion_class_index() != CLASS_INDEX:
        raise RuntimeError(
            f"HACDR-Net class_index is {tuple(model.card.class_index)}, not MA, HE, EX, SE = {CLASS_INDEX}"
        )
    if model.card.output_activation != "softmax":
        raise RuntimeError("HACDR-Net head is exclusive softmax; output_activation must stay softmax")
    if tuple(model.classes) != tuple(LESION_CLASSES):
        raise RuntimeError(f"output order {model.classes} is not MA, HE, EX, SE")


def predict(config_path, checkpoint, pred_dir, split):
    refuse_distributed()
    config = yaml.safe_load(Path(config_path).read_text())
    if config.get("model") != "HACDR-Net":
        raise ValueError(f"this script predicts HACDR-Net, got {config.get('model')!r}")
    pred_dir = Path(pred_dir)
    assert_not_finished_run(pred_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config["model"]).to(device)
    if model.net is None:
        raise ImportError(model.import_error or "HACDR-Net network was not imported")
    _check_channels(model)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    if "model_name" in state and state["model_name"] != model.card.name:
        raise ValueError(f"checkpoint model {state['model_name']!r} != config model {model.card.name!r}")
    model.load_state_dict(state["model"])
    model.eval()
    root = Path(config["dataset"])
    diameter = int(config["fov_diameter"])
    inference = inference_for(config, model.card)
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
