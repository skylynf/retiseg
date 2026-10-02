"""Contract every model wrapper implements, including Track C re-implementations.

A wrapper subclasses both torch.nn.Module and SegmentationWrapper, imports the network
from official_code/<repo> (or defines it, for Track C) and exposes:

    forward(x)       x: float tensor (B, in_channels, H, W), normalized as in `card.normalization`
                     returns logits (B, num_classes, H, W) at the input resolution,
                     class order equal to `classes`
    author_recipe()  the authors' training settings, used verbatim in setting B1

Training, prediction (resize back to original resolution, 16-bit PNG) and evaluation
are shared and must not be re-implemented inside a wrapper.
"""

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

    def __post_init__(self):
        if self.family not in FAMILIES:
            raise ValueError(f"family must be one of {FAMILIES}")
        if self.track not in TRACKS:
            raise ValueError(f"track must be one of {TRACKS}")
        if self.repro_level not in REPRO_LEVELS:
            raise ValueError(f"repro_level must be one of {REPRO_LEVELS}")
        if self.output_activation not in ("sigmoid", "softmax"):
            raise ValueError("output_activation must be 'sigmoid' or 'softmax'")


@dataclass(frozen=True)
class AuthorRecipe:
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


class SegmentationWrapper(ABC):
    card: ModelCard
    classes: tuple

    @abstractmethod
    def author_recipe(self) -> AuthorRecipe: ...

    def n_parameters(self):
        return sum(p.numel() for p in self.parameters())
