"""Find a wrapper class by the ``name`` field in bench/configs/models.yaml.

The map is explicit. Official repositories are not imported or scanned; the
E5 wrappers import theirs only when a model is built.
Registered names are U-Net, DeepLabv3, Swin-Unet, FCT, HRNet, M2MRF and HACDR-Net,
and the E5 training settings SAMed, SAM-Adapter, SAM2-UNet, MedSAM and RETFound
(design/15). Other names in the yaml fail until their wrapper is added here.
EXPERIMENT_ONLY holds the encoders of the E5 encoder comparison: experimental
conditions, not registered methods, so they are not in models.yaml.
``build_model`` calls the class with no arguments, so a wrapper's
``__init__`` takes none. Weights and dataset-specific settings belong on the
card or in ``author_recipe``.
"""

from pathlib import Path

import yaml

from bench.models.deeplabv3plus import DeepLabV3Plus
from bench.models.fct import FCT
from bench.models.hacdr import HACDRNet
from bench.models.hrnet import HRNet
from bench.models.m2mrf import M2MRF
from bench.models.medsam import MedSAM
from bench.models.sam2_unet import SAM2UNet
from bench.models.sam_adapter import SAMAdapter
from bench.models.samed import SAMed
from bench.models.swin_unet import SwinUnet
from bench.models.unet import UNet
from bench.models.vit_upernet import EncoderAugreg, EncoderDINOv2, EncoderMAE, EncoderRETFoundDINOv2, ViTUPerNet

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
    # E5 training settings (design/15 sections 7-8): B-std main table, seed 0 first.
    "SAMed": SAMed,
    "SAM-Adapter": SAMAdapter,
    "SAM2-UNet": SAM2UNet,
    "MedSAM": MedSAM,
    "RETFound": ViTUPerNet,
}
E5_MODELS = ("SAMed", "SAM-Adapter", "SAM2-UNet", "MedSAM", "RETFound")
# E5 encoder comparison: the RETFound row's UPerNet head and recipe on other ViT-L encoders, DDR only.
EXPERIMENT_ONLY = {
    "ViT-UPerNet MAE-IN1k": EncoderMAE,
    "ViT-UPerNet RETFound-DINOv2": EncoderRETFoundDINOv2,
    "ViT-UPerNet DINOv2": EncoderDINOv2,
    "ViT-UPerNet augreg": EncoderAugreg,
}


def model_names():
    path = Path(__file__).resolve().parents[1] / "configs" / "models.yaml"
    document = yaml.safe_load(path.read_text())
    names = []
    for entries in document.values():
        for entry in entries:
            names.append(entry["name"])
    return names


def wrapper_class(name):
    if name in EXPERIMENT_ONLY:
        return EXPERIMENT_ONLY[name]
    known = model_names()
    if name not in known:
        raise KeyError(f"{name!r} is not a name in bench/configs/models.yaml")
    if name not in WRAPPERS:
        raise KeyError(f"{name!r} has no B1 wrapper registered")
    return WRAPPERS[name]


def build_model(name):
    return wrapper_class(name)()


def recipe_and_card(name, dataset):
    """(author_recipe, card) of a model; classes with a recipe_for classmethod are not built.

    The queue launcher reads recipes for its seed-0 gate; building SAM-Adapter's
    ViT-H for that alone would take 2.5 GB.
    """
    cls = wrapper_class(name)
    if hasattr(cls, "recipe_for"):
        return cls.recipe_for(dataset), cls.card
    model = cls()
    return model.author_recipe(dataset), model.card
