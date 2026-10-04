"""B1 wrapper for M2MRF-C.

The network class stays in official_code/M2MRF. This module reads that
repository's IDRiD and DDR configs and exposes them as an AuthorRecipe.
It does not import mmseg until a network is actually built, so the recipe
can be read on PyTorch 2.x before mmcv is available.

B1 feeds multilabel masks in the order MA, HE, EX, SE. The author head is
four sigmoids in the order EX, HE, SE, MA. forward permutes those channels.
It does not overwrite overlapping lesions.
"""

import sys
from pathlib import Path

import torch
from torch import nn

from bench.models.base import AuthorRecipe, ModelCard, SegmentationWrapper

REPO = Path(__file__).resolve().parents[2]
OFFICIAL = REPO / "official_code" / "M2MRF"
CONFIGS = {
    "IDRiD": {
        "variant": OFFICIAL / "configs" / "m2mrf" / "fcn_hr48-M2MRF-C_40k_idrid_bdice.py",
        "schedule": OFFICIAL / "configs" / "_base_" / "schedules" / "schedule_40k_idrid.py",
        "dataset": OFFICIAL / "configs" / "_base_" / "datasets" / "idrid.py",
    },
    "DDR": {
        "variant": OFFICIAL / "configs" / "m2mrf" / "fcn_hr48-M2MRF-C_60k_ddr_bdice.py",
        "schedule": OFFICIAL / "configs" / "_base_" / "schedules" / "schedule_60k_ddr.py",
        "dataset": OFFICIAL / "configs" / "_base_" / "datasets" / "ddr.py",
    },
}
HR18 = OFFICIAL / "configs" / "_base_" / "models" / "fcn_hr18.py"
HR48 = OFFICIAL / "configs" / "_base_" / "models" / "fcn_hr48.py"
README = OFFICIAL / "README.md"

CLASSES = ("MA", "HE", "EX", "SE")
# Author classes are bg, EX, HE, SE, MA. The four sigmoid channels follow the
# lesion order, without the background channel.
AUTHOR_LESIONS = ("EX", "HE", "SE", "MA")
CHANNEL_INDEX = tuple(AUTHOR_LESIONS.index(name) for name in CLASSES)


def _exec_config(path):
    path = Path(path)
    namespace = {}
    exec(compile(path.read_text(), str(path), "exec"), namespace)
    return namespace


def _deep_merge(base, update):
    merged = dict(base)
    for key, value in update.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _line(path, needle):
    for number, line in enumerate(Path(path).read_text().splitlines(), start=1):
        if needle in line:
            return number, line.strip()
    raise RuntimeError(f"{needle!r} not found in {path}")


def _cite(path, needle, label):
    number, line = _line(path, needle)
    relative = Path(path).resolve().relative_to(REPO)
    return f"{label}: {relative}:{number}: {line}"


def _readme_gpus():
    number, line = _line(README, "dist_train.sh")
    count = int(line.split()[-1])
    relative = README.resolve().relative_to(REPO)
    return count, f"gpus: {relative}:{number}: {line}"


def merged_model_dict(dataset):
    """Author EncoderDecoder dict for M2MRF-C, including the pretrained URL."""
    if dataset not in CONFIGS:
        raise ValueError(f"B1 dataset must be IDRiD or DDR, got {dataset!r}")
    cfg = _deep_merge(_exec_config(HR18)["model"], _exec_config(HR48)["model"])
    variant = _exec_config(CONFIGS[dataset]["variant"])
    cfg = _deep_merge(cfg, variant["model"])
    return cfg


def author_rgb_norm(dataset):
    """RGB mean and std as written in the author dataset config, on 0-255."""
    namespace = _exec_config(CONFIGS[dataset]["dataset"])
    norm = namespace["img_norm_cfg"]
    if not norm.get("to_rgb", False):
        raise RuntimeError(f"{dataset} img_norm_cfg is not marked to_rgb")
    return list(norm["mean"]), list(norm["std"])


def b1_normalization(dataset):
    """Same affine map as the author, in the units b1_input.normalize expects.

    b1_input divides the JPEG by 255 before subtracting the mean. Dividing the
    author's 0-255 RGB stats by 255 leaves (pixel - mean) / std unchanged.
    Prepared images are already RGB, so the author's BGR-to-RGB flag is not
    applied a second time.
    """
    mean, std = author_rgb_norm(dataset)
    return [float(v) / 255.0 for v in mean], [float(v) / 255.0 for v in std]


def _pipeline_step(pipeline, step_type):
    found = [step for step in pipeline if step.get("type") == step_type]
    if len(found) != 1:
        raise RuntimeError(f"expected one {step_type} in the train pipeline, found {len(found)}")
    return found[0]


def author_recipe(dataset: str) -> AuthorRecipe:
    if dataset not in CONFIGS:
        raise ValueError(f"B1 dataset must be IDRiD or DDR, got {dataset!r}")
    paths = CONFIGS[dataset]
    schedule = _exec_config(paths["schedule"])
    data_ns = _exec_config(paths["dataset"])
    variant = _exec_config(paths["variant"])
    hr48 = _exec_config(HR48)
    optimizer = schedule["optimizer"]
    lr_config = schedule["lr_config"]
    iterations = int(schedule["runner"]["max_iters"])
    samples = int(data_ns["data"]["samples_per_gpu"])
    gpus, gpu_cite = _readme_gpus()
    loss = variant["model"]["decode_head"]["loss_decode"]
    classes = tuple(data_ns["classes"])
    if classes != ("bg",) + AUTHOR_LESIONS:
        raise RuntimeError(f"{dataset} classes {classes} are not bg, EX, HE, SE, MA")
    if loss.get("type") != "BinaryLoss" or loss.get("loss_type") != "dice":
        raise RuntimeError(f"{dataset} loss is not binary dice: {loss}")
    flip = _pipeline_step(data_ns["train_pipeline"], "RandomFlip")
    if "PhotoMetricDistortion" not in {step.get("type") for step in data_ns["train_pipeline"]}:
        raise RuntimeError(f"{dataset} train pipeline has no PhotoMetricDistortion")
    pretrained = hr48["model"]["pretrained"]
    mean, std = author_rgb_norm(dataset)
    image_scale = tuple(data_ns["image_scale"])
    crop = tuple(int(v) for v in data_ns["crop_size"])
    smooth = float(loss["smooth"])
    citations = [
        _cite(paths["variant"], "loss_type='dice'", "binary dice"),
        _cite(paths["schedule"], "type='SGD'", "SGD"),
        _cite(paths["schedule"], "policy='poly'", "poly"),
        _cite(paths["schedule"], "max_iters=", "iterations"),
        _cite(paths["dataset"], "samples_per_gpu=", "samples_per_gpu"),
        gpu_cite,
        _cite(HR48, "hrnetv2_w48", "ImageNet HRNet-W48"),
        _cite(paths["dataset"], "image_scale =", "image_scale"),
        _cite(paths["dataset"], "crop_size =", "crop_size"),
        _cite(paths["dataset"], "flip_ratio=", "flip_ratio"),
        _cite(paths["dataset"], "PhotoMetricDistortion", "photometric"),
        _cite(paths["dataset"], "to_rgb=True", "normalization"),
    ]
    notes = (
        "B1 deviations from the author train pipeline: "
        f"images are not resized to image_scale {image_scale} and are not randomly cropped to crop_size {crop}. "
        "PhotoMetricDistortion is not used. "
        "The field-of-view crop keeps its aspect ratio and its longer side is 1440. "
        f"Horizontal flip probability is 0.5; the author RandomFlip flip_ratio is {flip['flip_ratio']}. "
        "Targets are multilabel with channels MA, HE, EX, SE. "
        f"The author classes are {list(classes)}; overlapping pixels are not overwritten. "
        "Network channels are permuted from EX, HE, SE, MA to MA, HE, EX, SE. "
        f"samples_per_gpu is {samples}. The only dist_train.sh line launches {gpus} GPUs on the IDRiD config; "
        "DDR has no separate launch line and uses the same per-GPU batch, "
        f"so the author effective batch is {samples * gpus}. "
        f"The B1 loader batch is {samples * gpus} on one GPU, so BatchNorm sees {samples * gpus} images, "
        "the same count the author's SyncBN would synchronise. "
        "The learning rate is not multiplied by the GPU count. "
        f"Author RGB mean {mean} and std {std} are 0-255. "
        "b1_input.normalize divides by 255, so those stats are divided by 255 and the affine map is the same. "
        "Prepared images are already RGB, so the author's BGR-to-RGB conversion is not applied again."
    )
    return AuthorRecipe(
        loss="dice",
        optimizer=str(optimizer["type"]).lower(),
        lr=float(optimizer["lr"]),
        weight_decay=float(optimizer["weight_decay"]),
        schedule=str(lr_config["policy"]).lower(),
        iterations=iterations,
        batch_size=samples * gpus,
        effective_batch_size=samples * gpus,
        crop_size=crop,
        pretrained=str(pretrained),
        epochs=0,
        loss_params={"smooth": smooth},
        optimizer_params={
            "momentum": float(optimizer["momentum"]),
            "power": float(lr_config["power"]),
            "min_lr": float(lr_config["min_lr"]),
        },
        notes=notes,
        source_of_settings="\n".join(citations),
    )


class _Cfg(dict):
    """dict with attribute access, which the author's test_cfg expects."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def _install_import_path():
    compat = str(REPO / "bench" / "compat")
    official = str(OFFICIAL)
    for entry in (official, compat):
        while entry in sys.path:
            sys.path.remove(entry)
    sys.path.insert(0, compat)
    sys.path.insert(0, official)
    import numpy as np

    np.float = np.float64


def package_versions():
    info = {"torch": torch.__version__, "mmcv": None, "mmcv_file": None}
    try:
        import mmcv
    except Exception as exc:
        info["mmcv_error"] = f"{type(exc).__name__}: {exc}"
        return info
    info["mmcv"] = getattr(mmcv, "__version__", None)
    info["mmcv_file"] = getattr(mmcv, "__file__", None)
    return info


def refuse_author_runs(path):
    """Leave the author-weight run and the E1r retrains untouched."""
    from bench.runtime import assert_not_finished_run

    for candidate in (Path(path), Path(path).resolve()):
        parts = candidate.parts
        if "runs" not in parts:
            continue
        index = parts.index("runs")
        if index + 1 >= len(parts):
            continue
        name = parts[index + 1]
        if name == "E1_m2mrf" or name.startswith("E1r"):
            raise RuntimeError(f"refusing to write {path}: E1 and E1r runs stay as they are")
    assert_not_finished_run(path)


def build_network(dataset="IDRiD"):
    """Construct HRNet_M2MRF_C without loading the ImageNet checkpoint."""
    _install_import_path()
    from mmseg.models import build_segmentor

    cfg = merged_model_dict(dataset)
    cfg["pretrained"] = None
    test_cfg = _exec_config(CONFIGS[dataset]["variant"]).get("test_cfg") or {"mode": "whole"}
    network = build_segmentor(cfg, train_cfg=_Cfg(), test_cfg=_Cfg(test_cfg))
    return network


def bind_normalization(model, dataset):
    mean, std = b1_normalization(dataset)
    model.card = ModelCard(
        name=model.card.name,
        family=model.card.family,
        track=model.card.track,
        repro_level=model.card.repro_level,
        source=model.card.source,
        output_activation=model.card.output_activation,
        in_channels=model.card.in_channels,
        normalization={"mean": mean, "std": std},
        pad_multiple=model.card.pad_multiple,
        forward_size=model.card.forward_size,
        class_index=model.card.class_index,
    )
    return model


class M2MRF(SegmentationWrapper, nn.Module):
    classes = CLASSES
    card = ModelCard(
        name="M2MRF",
        family="fundus_multiscale",
        track="B",
        repro_level="R4",
        source="official_code/M2MRF",
        output_activation="sigmoid",
        in_channels=3,
        normalization={
            "mean": [116.513 / 255.0, 56.437 / 255.0, 16.309 / 255.0],
            "std": [80.206 / 255.0, 41.232 / 255.0, 13.293 / 255.0],
        },
        # Stem and the three stage transitions each stride by 2, so the lowest
        # HRNet branch is the input divided by 32. M2MRF pads its own patches.
        pad_multiple=32,
        forward_size=None,
        class_index=(),
    )

    def __init__(self, *, construct=True):
        super().__init__()
        self.network = None
        self.import_error = None
        self.versions = {"torch": torch.__version__, "mmcv": None, "mmcv_file": None}
        if not construct:
            return
        try:
            self.network = build_network()
        except Exception as exc:
            self.import_error = f"{type(exc).__name__}: {exc}"
        self.versions = package_versions()

    def author_recipe(self, dataset: str) -> AuthorRecipe:
        return author_recipe(dataset)

    def load_imagenet(self, path):
        """Load the local ImageNet HRNet-W48 file named by fcn_hr48.py."""
        if self.network is None:
            raise RuntimeError(self.import_error or "M2MRF network was not imported")
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        before = {name: tensor.detach().clone() for name, tensor in self.network.backbone.named_parameters()}
        self.network.init_weights(pretrained=str(path))
        copied = 0
        for name, tensor in self.network.backbone.named_parameters():
            if not torch.equal(before[name], tensor):
                copied += 1
        if copied == 0:
            raise RuntimeError(f"no backbone parameter changed when loading {path}")
        return copied

    def forward(self, x):
        if self.network is None:
            detail = self.import_error or "M2MRF network was not imported"
            raise RuntimeError(detail)
        logits = self.network.encode_decode(x, None)
        if logits.shape[1] != len(AUTHOR_LESIONS):
            raise RuntimeError(f"expected {len(AUTHOR_LESIONS)} author channels, got {logits.shape[1]}")
        return logits[:, CHANNEL_INDEX]


# The class card repeats the IDRiD stats so ModelCard does not have to wait
# for bind_normalization. Import fails if those literals drift from the config.
def _check_card_norm():
    mean, std = b1_normalization("IDRiD")
    card_mean = M2MRF.card.normalization["mean"]
    card_std = M2MRF.card.normalization["std"]
    for got, expected in zip(card_mean, mean):
        if abs(got - expected) > 1e-12:
            raise RuntimeError("M2MRF card mean drifted from the IDRiD config")
    for got, expected in zip(card_std, std):
        if abs(got - expected) > 1e-12:
            raise RuntimeError("M2MRF card std drifted from the IDRiD config")


_check_card_norm()

