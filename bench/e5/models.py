"""Promptable models for the E5 inference tiers, one small adapter each.

Every adapter offers:
    set_image(rgb)                       encode one window (uint8 RGB, H x W x 3)
    predict_box(box)                     one box [x0, y0, x1, y1] in window pixels -> (prob float32 H x W in [0, 1], score)
    predict_point(xy)                    one positive point -> the multimask output with the highest predicted IoU
    auto_masks(rgb, params)              SAM / SAM 2 automatic masks: list of {segmentation, predicted_iou, stability_score, area}
    window                               native window size

Prompted outputs are continuous (design/15 section 6.4): the predictor's
return_logits output through a sigmoid; MedSAM follows MedSAM_Inference.py
(sigmoid of the 256 low-resolution output, then bilinear to the window).

The official repositories are imported through bench.models.foundation.official
(official_code/<repo>, weights from pretrained/). SAM and SAM 2 follow their
released predictor and automatic-mask-generator APIs; MedSAM is the SAM ViT-B
architecture with the MedSAM checkpoint and its own 0-1 min-max normalization
(MedSAM_Inference.py); SAM 3 is in bench/e5/sam3_text.py.

Line references for the server-side check (the repositories are not on the
workstation): segment_anything/predictor.py (SamPredictor.set_image, predict),
segment_anything/automatic_mask_generator.py (SamAutomaticMaskGenerator.generate),
sam2/sam2_image_predictor.py (SAM2ImagePredictor), sam2/automatic_mask_generator.py.
"""

import numpy as np
import torch

from bench.models.foundation import official, weights
from bench.e5.common import WINDOW


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SamAdapter:
    """segment-anything (SAM 1), ViT-H or ViT-B; also the base of MedSAM."""

    name = "sam"
    window = WINDOW["sam"]

    def __init__(self, variant="vit_h", checkpoint=None, repo="segment-anything", medsam=False):
        self.variant = variant
        self.medsam = medsam
        file = checkpoint or {"vit_h": "sam_vit_h_4b8939.pth", "vit_b": "sam_vit_b_01ec64.pth"}[variant]
        with official(repo, purge=("segment_anything",)):
            from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry

            self._amg_cls, self._predictor_cls = SamAutomaticMaskGenerator, SamPredictor
            self.model = sam_model_registry[variant](checkpoint=str(weights(file))).to(_device()).eval()
        self.predictor = self._predictor_cls(self.model)
        self._image = None

    def set_image(self, rgb):
        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
        if self.medsam:
            # MedSAM_Inference.py: per-image min-max to [0, 1] on the 1024 resize, no mean/std.
            # SamPredictor applies SAM's pixel mean/std; feeding the min-max image scaled back to 0-255 and
            # bypassing that normalization is done by replacing the model's statistics with 0 / 255 once.
            lo, hi = float(rgb.min()), float(rgb.max())
            rgb = ((rgb.astype(np.float32) - lo) / max(hi - lo, 1e-8) * 255.0).astype(np.uint8)
        self.predictor.set_image(rgb)
        self._image = rgb

    def predict_box(self, box):
        logits, scores, low_res = self.predictor.predict(box=np.asarray(box, dtype=np.float32), multimask_output=False,
                                                         return_logits=True)
        return self._prob(logits[0], low_res[0]), float(scores[0])

    def predict_point(self, xy):
        logits, scores, low_res = self.predictor.predict(
            point_coords=np.asarray([xy], dtype=np.float32), point_labels=np.asarray([1]), multimask_output=True,
            return_logits=True,
        )
        best = int(np.argmax(scores))
        return self._prob(logits[best], low_res[best]), float(scores[best])

    def _prob(self, logits, low_res):
        return _sigmoid(np.asarray(logits, dtype=np.float32))

    def auto_masks(self, rgb, params):
        generator = self._amg_cls(self.model, **params)
        return generator.generate(np.ascontiguousarray(rgb, dtype=np.uint8))


def _sigmoid(x):
    return (1.0 / (1.0 + np.exp(-x.astype(np.float64)))).astype(np.float32)


class MedSamAdapter(SamAdapter):
    """MedSAM: SAM ViT-B weights trained on medical images; box prompts only (e5_frozen oracle.medsam)."""

    name = "medsam"

    def __init__(self):
        super().__init__(variant="vit_b", checkpoint="medsam_vit_b.pth", repo="MedSAM", medsam=True)
        # MedSAM skips SAM's mean/std (MedSAM_Inference.py:121-136): neutralize them so set_image's 0-255
        # min-max image reaches the encoder as the 0-1 image MedSAM was trained on.
        with torch.no_grad():
            self.model.pixel_mean.zero_()
            self.model.pixel_std.fill_(255.0)

    def predict_point(self, xy):
        raise NotImplementedError("MedSAM is used with boxes only (e5_frozen.yaml oracle.models)")

    def _prob(self, logits, low_res):
        # MedSAM_Inference.py:59-71: sigmoid of the 256 x 256 output, bilinear to the input size.
        prob = torch.sigmoid(torch.as_tensor(np.asarray(low_res, dtype=np.float32))[None, None])
        prob = torch.nn.functional.interpolate(prob, size=self._image.shape[:2], mode="bilinear", align_corners=False)
        return prob[0, 0].numpy()

    def auto_masks(self, rgb, params):
        raise NotImplementedError("MedSAM has no automatic-mask generator in E5")


class Sam2Adapter:
    """SAM 2.1 Hiera-L image predictor and automatic mask generator (official_code/sam2)."""

    name = "sam2"
    window = WINDOW["sam2"]

    def __init__(self, checkpoint="sam2.1_hiera_large.pt", config="configs/sam2.1/sam2.1_hiera_l.yaml",
                 apply_postprocessing=True):
        # apply_postprocessing=False is what the "more_masks" generator setting asks for (e5_frozen.yaml,
        # notebooks/automatic_mask_generator_example.ipynb); it is a build_sam2 argument, not a generator one.
        self.apply_postprocessing = bool(apply_postprocessing)
        with official("sam2", purge=("sam2",)):
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            self._amg_cls = SAM2AutomaticMaskGenerator
            self.model = build_sam2(config, str(weights(checkpoint)), device=_device(),
                                    apply_postprocessing=self.apply_postprocessing)
            self.predictor = SAM2ImagePredictor(self.model)

    def set_image(self, rgb):
        self.predictor.set_image(np.ascontiguousarray(rgb, dtype=np.uint8))

    def predict_box(self, box):
        logits, scores, _ = self.predictor.predict(box=np.asarray(box, dtype=np.float32), multimask_output=False,
                                                   return_logits=True)
        return _sigmoid(np.asarray(logits[0], dtype=np.float32)), float(scores[0])

    def predict_point(self, xy):
        logits, scores, _ = self.predictor.predict(
            point_coords=np.asarray([xy], dtype=np.float32), point_labels=np.asarray([1]), multimask_output=True,
            return_logits=True,
        )
        best = int(np.argmax(scores))
        return _sigmoid(np.asarray(logits[best], dtype=np.float32)), float(scores[best])

    def auto_masks(self, rgb, params):
        generator = self._amg_cls(self.model, **params)
        return generator.generate(np.ascontiguousarray(rgb, dtype=np.uint8))


def build(name, variant="vit_h", **kwargs):
    """``kwargs`` are model-construction options such as SAM 2's apply_postprocessing."""
    if name == "sam":
        return SamAdapter(variant=variant, **kwargs)
    if name == "medsam":
        return MedSamAdapter(**kwargs)
    if name == "sam2":
        return Sam2Adapter(**kwargs)
    raise KeyError(f"unknown E5 model {name!r}; expected sam, medsam or sam2 (sam3 is bench/e5/sam3_text.py)")
