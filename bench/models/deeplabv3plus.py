"""DeepLabv3 baseline for B1, wrapped around torchvision.

The registered name is DeepLabv3. torchvision's ``deeplabv3_resnet101`` is
DeepLabv3, not the DeepLabv3+ decoder.

The callable is ``deeplabv3_resnet101``. In torchvision 0.21 that network is
DeepLabv3: ResNet-101, ASPP rates (12, 24, 36), and bilinear upsampling of the
logits to the input size. It does not add the DeepLabv3+ decoder. Its output
stride is 8 (``replace_stride_with_dilation=[False, True, True]``). The PASCAL
VOC recipe below still records the paper's training output stride of 16.

``forward`` returns logits. The card's sigmoid is applied by the runner.
Channel 0-3 are MA, HE, EX, SE, the same order as ``bench.common.io.LESION_CLASSES``.
The four-channel classifier is created with ``num_classes=4`` and is not loaded
from a VOC or COCO checkpoint. The backbone default is ImageNet-1K.

Normalization matches ``ResNet101_Weights.IMAGENET1K_V1.transforms()`` on
images already in ``[0, 1]``: mean ``[0.485, 0.456, 0.406]`` and std
``[0.229, 0.224, 0.225]``. Those are the 0-255 channel statistics divided by
255. The B1 runner divides by 255 before applying ``card.normalization``.
"""

from torch import nn

from bench.models.base import AuthorRecipe, ModelCard, SegmentationWrapper

# torchvision ResNet101_Weights.IMAGENET1K_V1.transforms(), 0-1 space.
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

_SOURCE = (
    "lr=0.007, schedule=poly, crop_size=513x513, fine-tune batch normalization when output stride=16: "
    "Chen et al., DeepLabv3+, ECCV 2018, Section 4, the paragraph before Section 4.1. "
    "power=0.9, iterations=30000, batch_size=16, output stride=16, and the same initial learning rate 0.007: "
    "Chen et al., DeepLabv3, arXiv:1706.05587, Section 4.1 Training Protocol, which DeepLabv3+ Section 4 says it follows. "
    "pretrained=ResNet101_Weights.IMAGENET1K_V1: DeepLabv3+ Section 3, ImageNet-1k pretrained ResNet-101. "
    "momentum=0.9: official_code/DeepLabv3plus/research/deeplab/train.py:116, the default of the momentum optimizer selected at train.py:84. "
    "weight_decay=0.0001 for ResNet: train.py:129-132. The flag default on line 131 is 0.00004 and the comment assigns that value to MobileNet-V2 and Xception. "
    "DeepLabv3+ Section 4.3 (Nesterov momentum 0.9, learning rate 0.05, weight decay 4e-5) is Xception ImageNet pretraining, not this recipe."
)

_NOTES = (
    "IDRiD and DDR use this same recipe. "
    "crop_size 513 is the PASCAL VOC crop from DeepLabv3+ Section 4 and is not applied; "
    "forward_size is empty, so B1 feeds the field-of-view canvas whose longer side is fov_diameter. "
    "torchvision deeplabv3_resnet101 has no hard spatial size: DeepLabV3.forward interpolates logits "
    "to the input height and width, so the canvas is not resized to 513. "
    "The wrapped network is DeepLabv3, not the DeepLabv3+ decoder, and its output stride is 8 "
    "(ASPP rates 12, 24, 36). The paper's training output stride is 16 (DeepLabv3+ Section 4 and Table 3). "
    "torchvision drops the ASPP projection with probability 0.5; official_code/DeepLabv3plus "
    "model.py uses keep_prob 0.9 on that projection. This wrapper leaves the torchvision dropout in place. "
    "DeepLabv3 Section 4.1 then freezes batch normalization, uses output stride 8, and runs another 30K "
    "iterations at learning rate 0.001 on VOC trainval. That second stage is not one learning rate, and "
    "B1 has no trainval stage, so iterations stay 30000 at learning rate 0.007. "
    "The paper does not name a loss; VOC mIOU is 21 mutually exclusive classes. Overlapping lesions use "
    "bce_with_logits and sigmoid. "
    "The runner optimizer is sgd with momentum 0.9, which is the PyTorch form of train.py's MomentumOptimizer. "
    "train.py:123-127 says fine_tune_batch_norm needs a batch larger than 12, and a smaller batch should set that flag false. "
    "Sixteen images at long side 1440 do not fit one 40GB GPU, so the loader batch is 1 and gradients accumulate toward 16. "
    "On 44 IDRiD training images one epoch submits steps of 16, 16 and 12 images. "
    "The nominal effective batch of 16 is only the full step; the tail of 12 is the mean over those 12 images. "
    "Every BatchNorm2d stays in eval and is not trained. "
    "Random scale augmentation in DeepLabv3+ Section 4 is not applied; the B1 runner owns augmentation."
)


class DeepLabV3Plus(SegmentationWrapper, nn.Module):
    classes = ("MA", "HE", "EX", "SE")
    card = ModelCard(
        name="DeepLabv3",
        family="general",
        track="B",
        repro_level="R3",
        source="torchvision.models.segmentation.deeplabv3_resnet101",
        output_activation="sigmoid",
        normalization={"mean": list(_IMAGENET_MEAN), "std": list(_IMAGENET_STD)},
        # Output stride of this torchvision network is 8.
        pad_multiple=8,
        forward_size=None,
        class_index=(),
    )

    def __init__(self, weights_backbone="IMAGENET1K_V1"):
        super().__init__()
        from torchvision.models.segmentation import deeplabv3_resnet101

        self.net = deeplabv3_resnet101(
            weights=None,
            weights_backbone=weights_backbone,
            num_classes=len(self.classes),
            aux_loss=False,
            progress=False,
        )
        last = self.net.classifier[-1]
        if not isinstance(last, nn.Conv2d) or last.out_channels != len(self.classes):
            raise RuntimeError(
                f"DeepLab classifier must be a 4-channel conv, got {type(last).__name__} "
                f"out_channels={getattr(last, 'out_channels', None)}"
            )
        if self.net.aux_classifier is not None:
            raise RuntimeError("aux classifier is not part of this B1 wrapper")

    def author_recipe(self, dataset: str) -> AuthorRecipe:
        if dataset not in ("IDRiD", "DDR"):
            raise ValueError(f"B1 dataset must be IDRiD or DDR, got {dataset!r}")
        return AuthorRecipe(
            loss="bce_with_logits",
            optimizer="sgd",
            lr=0.007,
            weight_decay=0.0001,
            schedule="poly",
            iterations=30000,
            epochs=0,
            batch_size=1,
            effective_batch_size=16,
            crop_size=(513, 513),
            pretrained="ResNet101_Weights.IMAGENET1K_V1",
            optimizer_params={"momentum": 0.9, "power": 0.9},
            notes=_NOTES,
            source_of_settings=_SOURCE,
        )

    def train(self, mode=True):
        super().train(mode)
        if mode:
            for module in self.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
                    for param in module.parameters():
                        param.requires_grad = False
        return self

    def forward(self, x):
        return self.net(x)["out"]
