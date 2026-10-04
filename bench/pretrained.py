"""Local ImageNet files named by the author recipes.

Nothing here is fetched. Training raises when a required file is absent, so a
run cannot start from random weights and still be described as pretrained.
"""

from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
DIR = REPO / "pretrained"

HRNET_W48_NAME = "hrnetv2_w48-d2186c55.pth"
SWIN_T_NAME = "swin_tiny_patch4_window7_224.pth"
HRNET_W48_URL = "https://download.openmmlab.com/pretrain/third_party/hrnetv2_w48-d2186c55.pth"
SWIN_T_URL = "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth"


def _existing(name, extra, url):
    candidates = [DIR / name, *extra]
    for path in candidates:
        if path.is_file():
            return path
    places = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"missing {name}. Looked in {places}. Download {url} and save it as {DIR / name}.")


def hrnet_w48():
    """open-mmlab://msra/hrnetv2_w48, the file E1r already loaded."""
    return _existing(
        HRNET_W48_NAME,
        [REPO / "runs" / "E1_m2mrf" / HRNET_W48_NAME],
        HRNET_W48_URL,
    )


def swin_tiny():
    return _existing(SWIN_T_NAME, [], SWIN_T_URL)


def load_matching(module, path, minimum=50):
    """Copy tensors whose names and shapes already agree. Leave the rest."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(blob, dict):
        for key in ("state_dict", "model"):
            if key in blob and isinstance(blob[key], dict):
                blob = blob[key]
                break
    cleaned = {}
    for key, value in blob.items():
        name = key[7:] if key.startswith("module.") else key
        cleaned[name] = value
    model_dict = module.state_dict()
    matched = {
        key: value
        for key, value in cleaned.items()
        if key in model_dict and tuple(value.shape) == tuple(model_dict[key].shape)
    }
    if len(matched) < int(minimum):
        raise RuntimeError(f"{path} matched {len(matched)} tensors in {type(module).__name__}; refusing to train")
    model_dict.update(matched)
    module.load_state_dict(model_dict)
    return len(matched)
