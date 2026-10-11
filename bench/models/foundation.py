"""Shared pieces of the E5 wrappers (design/15): official imports, weights, recomputation, schedules.

The official repositories ship conflicting copies of the same top-level
packages: SAMed and MedSAM each fork ``segment_anything``, SAM-Adapter has a
generic ``models`` package, SAM2-UNet bundles its own ``sam2`` that registers
a Hydra config module on import. ``official`` imports one repository at a
time: it drops the named packages from sys.modules, puts the repository
first on sys.path, and takes it off again afterwards. The classes it returns
keep working after another repository is imported, because their modules hold
their own globals. Nothing under official_code is changed.

Weights are read from pretrained/ only. A missing file raises before the first
step, with the download source in the message, so a run never starts from
random weights while being described as pretrained.
"""

import contextlib
import math
import sys
import types
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
OFFICIAL = REPO / "official_code"
PRETRAINED = REPO / "pretrained"
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# file name -> where it comes from (design/15 section 12).
WEIGHTS = {
    "sam_vit_b_01ec64.pth": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    "sam_vit_h_4b8939.pth": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "sam2_hiera_large.pt": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt",
    "sam2.1_hiera_large.pt": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
    "medsam_vit_b.pth": "https://drive.google.com/drive/folders/1ETWmi4AiniJeWOt6HAsYgTjYv_fkgzoN (MedSAM README)",
    "RETFound_mae_natureCFP.pth": "https://huggingface.co/YukunZhou/RETFound_mae_natureCFP (gated)",
    "RETFound_dinov2_meh.pth": "https://huggingface.co/YukunZhou/RETFound_dinov2_meh (gated)",
    "mae_pretrain_vit_large.pth": "https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_large.pth",
}


@contextlib.contextmanager
def official(repo, purge=(), extra_paths=()):
    """Import from official_code/<repo> with the packages in ``purge`` loaded fresh.

    ``extra_paths`` go on sys.path in front of the repository for the import,
    e.g. bench/compat for the mmcv subset SAM-Adapter's package imports.
    """
    root = str(OFFICIAL / repo)
    if not Path(root).is_dir():
        raise FileNotFoundError(f"{root} is missing; git submodule update --init official_code/{repo}")
    for name in list(sys.modules):
        if any(name == prefix or name.startswith(prefix + ".") for prefix in purge):
            del sys.modules[name]
    added = [root] + [str(path) for path in extra_paths]
    for path in added:
        sys.path.insert(0, path)
    try:
        yield
    finally:
        for path in added:
            try:
                sys.path.remove(path)
            except ValueError:
                pass


def weights(name):
    """Path of a weights file under pretrained/, or a FileNotFoundError naming its source."""
    path = PRETRAINED / name
    if not path.is_file():
        raise FileNotFoundError(f"missing pretrained/{name}; download it from {WEIGHTS.get(name, 'its release page')}")
    return path


def recompute(modules):
    """Recompute each module's forward in backward (torch.utils.checkpoint), state dict unchanged.

    The bound forward is wrapped on the instance, so parameter names and
    checkpoints stay the same; outside training it runs as before. Used only
    when a cell asks for it (activation_checkpointing), and recorded.
    """
    from torch.utils.checkpoint import checkpoint

    for module in modules:
        plain = module.forward

        def forward(self, *args, _plain=plain, **kwargs):
            if self.training and torch.is_grad_enabled():
                return checkpoint(_plain, *args, use_reentrant=False, **kwargs)
            return _plain(*args, **kwargs)

        module.forward = types.MethodType(forward, module)


def stepped_cosine(step, total, base, eta_min, segments):
    """CosineAnnealingLR(T_max=segments) stepped once per segment, stretched over ``total`` updates.

    Update ``step`` (0-based) belongs to segment floor(segments * step / total),
    which has the rate the author's scheduler gives that epoch.
    """
    segment = min(int(segments) - 1, int(segments) * int(step) // max(1, int(total)))
    return eta_min + (base - eta_min) * (1.0 + math.cos(math.pi * segment / int(segments))) / 2.0


def set_lr(optimizer, lr):
    """Write ``lr`` to every group, times the group's lr_scale when it has one."""
    for group in optimizer.param_groups:
        group["lr"] = lr * group.get("lr_scale", 1.0)


def trainable(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
