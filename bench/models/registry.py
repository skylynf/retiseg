"""Find a B1 wrapper class by the ``name`` field in bench/configs/models.yaml.

The map is explicit. Official repositories are not imported or scanned.
Registered names are U-Net, DeepLabv3, Swin-Unet, FCT, HRNet, M2MRF and HACDR-Net. Other names in the yaml fail until
their wrapper is added here. ``build_model`` calls the class with no arguments, so
a wrapper's ``__init__`` takes none. Weights and dataset-specific settings
belong on the card or in ``author_recipe``.
"""

from pathlib import Path

import yaml

from bench.models.deeplabv3plus import DeepLabV3Plus
from bench.models.fct import FCT
from bench.models.hacdr import HACDRNet
from bench.models.hrnet import HRNet
from bench.models.m2mrf import M2MRF
from bench.models.swin_unet import SwinUnet
from bench.models.unet import UNet

WRAPPERS = {
    "U-Net": UNet,
    "DeepLabv3": DeepLabV3Plus,
    "Swin-Unet": SwinUnet,
    "FCT": FCT,
    "HRNet": HRNet,
    # H2Former is out of the first batch: at 960 one stage-0 Swin block's forward
    # holds several 12 GB attention maps, which a 40 GB A100 cannot fit even with
    # the backward recomputation in bench/models/h2former.py.
    "M2MRF": M2MRF,
    "HACDR-Net": HACDRNet,
}


def model_names():
    path = Path(__file__).resolve().parents[1] / "configs" / "models.yaml"
    document = yaml.safe_load(path.read_text())
    names = []
    for entries in document.values():
        for entry in entries:
            names.append(entry["name"])
    return names


def build_model(name):
    known = model_names()
    if name not in known:
        raise KeyError(f"{name!r} is not a name in bench/configs/models.yaml")
    if name not in WRAPPERS:
        raise KeyError(f"{name!r} has no B1 wrapper registered")
    return WRAPPERS[name]()
