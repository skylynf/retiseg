"""Contract tests of the three real model builds of the E5 inference tiers (docs/ops/21 open issue 28).

They need the weights under pretrained/ and are skipped without them, so the suite stays
runnable on a machine that only holds the code; the automatic-mask generators also need a
CUDA device (64 points per side on a CPU takes minutes). server_run.sh check runs them in
retiseg (SAM, SAM 2, MedSAM) and retiseg-sam3 (SAM 3). Each test builds the official model
exactly as the runs do, feeds one synthetic window and asserts the output's shape, dtype
and range and the model's device: review A4 (the SAM 3 model left on the CPU) is the kind
of failure they catch before a smoke does.
"""
from pathlib import Path

import numpy as np
import pytest

from bench.e5 import frozen as fz
from bench.models import foundation

REPO = Path(__file__).resolve().parents[2]


def _have(*names):
    return all((foundation.PRETRAINED / name).is_file() for name in names)


def _device():
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _window(size, seed=0):
    """A fundus-like window: dark red background, a bright disc, a few small bright and dark spots."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size]
    image = np.zeros((size, size, 3), np.uint8)
    image[..., 0], image[..., 1], image[..., 2] = 120, 40, 20
    disc = (yy - size * 0.5) ** 2 + (xx - size * 0.6) ** 2 < (size * 0.08) ** 2
    image[disc] = (235, 200, 120)
    for _ in range(12):
        cy, cx, r = rng.integers(40, size - 40), rng.integers(40, size - 40), rng.integers(3, 9)
        spot = (yy - cy) ** 2 + (xx - cx) ** 2 < r**2
        image[spot] = (250, 230, 150) if rng.random() < 0.5 else (60, 15, 10)
    return image


def _params_device(module):
    return {parameter.device.type for parameter in module.parameters()}


@pytest.mark.skipif(not _have("sam_vit_h_4b8939.pth"), reason="pretrained/sam_vit_h_4b8939.pth missing")
@pytest.mark.skipif(_device() != "cuda", reason="the automatic-mask generator at 64 points per side needs a CUDA device")
def test_sam_generator_contract():
    from bench.e5 import amg

    generator, record = amg.build_generator("SAM", 64, fz.load_frozen(), _device())
    assert _params_device(generator.predictor.model) == {_device()} and record["device"] == _device() and record["checkpoint"]["sha256"]
    records = generator.generate(_window(1024))
    assert isinstance(records, list) and records
    for item in records[:20]:
        assert item["segmentation"].shape == (1024, 1024) and item["segmentation"].dtype == bool
        assert 0.0 <= float(item["predicted_iou"]) <= 1.5 and 0.0 <= float(item["stability_score"]) <= 1.0  # SAM's IoU head can exceed 1 slightly


@pytest.mark.skipif(not _have("sam2.1_hiera_large.pt"), reason="pretrained/sam2.1_hiera_large.pt missing")
@pytest.mark.skipif(_device() != "cuda", reason="the automatic-mask generator at 64 points per side needs a CUDA device")
def test_sam2_generator_contract():
    from bench.e5 import amg

    generator, record = amg.build_generator("SAM 2", 64, fz.load_frozen(), _device())
    assert _params_device(generator.predictor.model) == {_device()} and record["sam2_postprocessing"]["apply_postprocessing"] is True
    records = generator.generate(_window(1024))
    assert isinstance(records, list) and records
    for item in records[:20]:
        assert item["segmentation"].shape == (1024, 1024) and item["segmentation"].dtype == bool
        assert 0.0 <= float(item["predicted_iou"]) <= 1.5 and 0.0 <= float(item["stability_score"]) <= 1.0


@pytest.mark.parametrize("model, weight, kind", [("SAM", "sam_vit_b_01ec64.pth", "boxes"), ("SAM 2", "sam2.1_hiera_large.pt", "points"), ("MedSAM", "medsam_vit_b.pth", "boxes")])
def test_prompt_decoder_contract(model, weight, kind):
    if not _have(weight):
        pytest.skip(f"pretrained/{weight} missing")
    from bench.e5 import prompted

    decoder, record = prompted.build_decoder(model, fz.load_frozen(), _device())
    network = decoder.model if model == "MedSAM" else decoder.predictor.model
    assert _params_device(network) == {_device()} and record["checkpoint"]["sha256"]
    decoder.set_image(_window(1024))
    prob = decoder.box([600, 480, 680, 560]) if kind == "boxes" else decoder.point([614, 512])
    prob = np.asarray(prob)
    assert prob.shape == (1024, 1024) and np.issubdtype(prob.dtype, np.floating)
    assert np.isfinite(prob).all() and prob.min() >= 0.0 and prob.max() <= 1.0 and prob.max() > 0.05  # the bright disc is segmentable


@pytest.mark.skipif(not _have("sam3.pt"), reason="pretrained/sam3.pt missing")
def test_sam3_scorer_contract():
    from PIL import Image

    from bench.e5 import sam3_text

    try:
        with foundation.official("sam3", purge=("sam3",)):
            import sam3  # noqa: F401
    except ImportError:
        pytest.skip("sam3 is not importable here (retiseg-sam3 environment)")
    scorer, record = sam3_text.build_scorer(fz.load_frozen(), _device(), 1008)  # check_placed raises when the model stayed on the CPU
    assert record["builder_device"] == _device() and record["resolution"] == 1008 and record["confidence_threshold"] == 0.0
    scorer.set_image(Image.fromarray(_window(1008)))
    prob = np.asarray(scorer.score("hard exudate"))
    assert prob.shape == (1008, 1008) and prob.dtype == np.float32 and np.isfinite(prob).all() and 0.0 <= prob.min() and prob.max() <= 1.0
