"""Contract every model wrapper implements, including Track C re-implementations.

A wrapper subclasses both torch.nn.Module and SegmentationWrapper. It imports the
network from official_code/<repo> (or defines it, for Track C) and exposes:

    forward(x)  x: float tensor (B, in_channels, H, W), already normalized
                with card.normalization and padded with zeros to card.pad_multiple.
                Returns logits (B, C, H, W) at that same spatial size.

    author_recipe(dataset)  the authors' training settings for "IDRiD" or "DDR".
                Setting B1 uses them for the loss, the optimizer and the step
                count. The shared runner owns the data split, the field-of-view
                canvas and the evaluation.

The runner calls build_loss, build_optimizer and adjust_lr. A wrapper overrides
one of them only when the default named by the recipe cannot express the authors'
objective. Training, probability maps and evaluation stay outside the wrapper.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

FAMILIES = ("general", "fundus_multiscale", "relational_transformer", "semi_supervised", "foundation")
TRACKS = ("A", "B", "C")
REPRO_LEVELS = ("R0", "R1", "R2", "R3", "R4")


@dataclass(frozen=True)
class ModelCard:
    name: str
    family: str
    track: str
    repro_level: str
    source: str
    output_activation: str = "sigmoid"  # sigmoid: one channel per lesion; softmax: channel 0 is background
    in_channels: int = 3
    normalization: dict = field(default_factory=lambda: {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]})
    # After normalization the runner pads the bottom and right with 0 until both
    # sides are multiples of this. 16 is the U-Net downsampling depth.
    pad_multiple: int = 16
    # None: the network sees the FOV crop whose longer side is fov_diameter.
    # (H, W): that canvas is resized to this size, and predictions are mapped
    # back onto the original image. This is not a random crop.
    forward_size: tuple | None = None
    # Empty: sigmoid, and channels 0-3 are MA, HE, EX, SE.
    # Four ints: softmax over the network channels (background included); these
    # are the channel ids of MA, HE, EX, SE. Those four planes are not
    # renormalized after they are taken out.
    class_index: tuple = ()

    def __post_init__(self):
        if self.family not in FAMILIES:
            raise ValueError(f"family must be one of {FAMILIES}")
        if self.track not in TRACKS:
            raise ValueError(f"track must be one of {TRACKS}")
        if self.repro_level not in REPRO_LEVELS:
            raise ValueError(f"repro_level must be one of {REPRO_LEVELS}")
        if self.output_activation not in ("sigmoid", "softmax"):
            raise ValueError("output_activation must be 'sigmoid' or 'softmax'")
        if not isinstance(self.pad_multiple, int) or isinstance(self.pad_multiple, bool) or self.pad_multiple < 1:
            raise ValueError("pad_multiple must be a positive integer")
        if self.forward_size is not None:
            if len(self.forward_size) != 2 or any(int(v) < 1 for v in self.forward_size):
                raise ValueError("forward_size must be (height, width) with positive sides")
            object.__setattr__(self, "forward_size", (int(self.forward_size[0]), int(self.forward_size[1])))
        object.__setattr__(self, "class_index", tuple(int(i) for i in self.class_index))
        if any(i < 0 for i in self.class_index):
            raise ValueError("class_index channels must be >= 0")
        if self.output_activation == "sigmoid" and len(self.class_index) != 0:
            raise ValueError("sigmoid output keeps class_index empty; channels 0-3 are MA, HE, EX, SE")
        if self.output_activation == "softmax" and len(self.class_index) != 4:
            raise ValueError("softmax output needs class_index with the MA, HE, EX, SE channel ids")


@dataclass
class AuthorRecipe:
    """Authors' optimization settings. Spatial preprocessing is not in here.

    ``crop_size`` records what the authors cropped to. The B1 runner does not
    crop to it: every model sees the field-of-view canvas at ``fov_diameter``,
    then ``card.forward_size`` when that is set.

    ``iterations`` 0 together with ``epochs`` 0 means the cell config supplies
    the epoch count. The runner then sets
    iterations = epochs * ceil(training images / effective_batch_size).
    A positive ``iterations`` is the author's step count and wins over the
    cell config. ``effective_batch_size`` defaults to ``batch_size``. When the
    cell's batch is smaller, the runner accumulates gradients up to this value
    and does not start a second process.
    """

    loss: str
    optimizer: str
    lr: float
    weight_decay: float
    schedule: str
    iterations: int
    batch_size: int
    crop_size: tuple
    pretrained: str
    notes: str = ""
    source_of_settings: str = ""  # file and line, or paper section, each value was taken from
    epochs: int = 0
    effective_batch_size: int | None = None
    loss_params: dict = field(default_factory=dict)
    optimizer_params: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.effective_batch_size is None:
            self.effective_batch_size = self.batch_size
        if not isinstance(self.loss_params, dict) or not isinstance(self.optimizer_params, dict):
            raise ValueError("loss_params and optimizer_params must be dicts")
        if int(self.batch_size) < 1 or int(self.effective_batch_size) < 1:
            raise ValueError("batch_size and effective_batch_size must be positive")
        if int(self.iterations) < 0 or int(self.epochs) < 0:
            raise ValueError("iterations and epochs must be >= 0")
        self.batch_size = int(self.batch_size)
        self.effective_batch_size = int(self.effective_batch_size)
        self.iterations = int(self.iterations)
        self.epochs = int(self.epochs)


class SegmentationWrapper(ABC):
    card: ModelCard
    classes: tuple

    @abstractmethod
    def author_recipe(self, dataset: str) -> AuthorRecipe: ...

    def n_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def build_loss(self, recipe):
        from bench.runtime import build_loss

        return build_loss(recipe, self.card.output_activation)

    def build_optimizer(self, params, recipe):
        from bench.runtime import build_optimizer

        return build_optimizer(params, recipe)

    def adjust_lr(self, optimizer, step, total_steps, recipe):
        from bench.runtime import adjust_lr

        return adjust_lr(optimizer, step, total_steps, recipe)
