"""TC-Net reproduction from Zhang et al., Computers in Biology and Medicine 2023.

The network takes an RGB image and no extra input. See DESIGN.md.
"""

from tcnet.assumptions import EXTRA_INPUTS, PAPER, SEED
from tcnet.loss import DynamicCyclicalFocalLoss, cyclical_indicator
from tcnet.model import TCNet
from tcnet.seed import set_seed

__all__ = [
    "EXTRA_INPUTS",
    "PAPER",
    "SEED",
    "TCNet",
    "DynamicCyclicalFocalLoss",
    "cyclical_indicator",
    "set_seed",
]
