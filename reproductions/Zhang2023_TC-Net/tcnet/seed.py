"""Public random seed for the TC-Net reproduction."""

from __future__ import annotations

import os
import random

from tcnet.assumptions import SEED


def set_seed(seed: int = SEED) -> int:
    """Seed Python and PyTorch. Returns the seed that was used."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np
    except ImportError:
        np = None
    else:
        np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed
