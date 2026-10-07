"""Locked constants for the TC-Net reproduction.

Values under the paper-stated names are written in Zhang et al.,
Computers in Biology and Medicine 2023. Gap-filling choices are listed
in ASSUMPTIONS and explained in Chinese in DESIGN.md. POSTHOC holds the
published table numbers and is not read by the model or the loss.
"""

from __future__ import annotations

PAPER = {
    "id": "M-TCNET-2023",
    "title": (
        "TC-Net: A joint learning framework based on CNN and vision "
        "transformer for multi-lesion medical images segmentation"
    ),
    "authors": (
        "Zhongxiang Zhang, Guangmin Sun, Kun Zheng, Jin-Kui Yang, "
        "Xiao-rong Zhu, Yu Li"
    ),
    "year": 2023,
    "venue": "Computers in Biology and Medicine 161:106967",
    "doi": "10.1016/j.compbiomed.2023.106967",
    "pdf": "literature/papers/03_dr_methods/relation_transformer/Zhang2023_TC-Net.pdf",
}

# Section 4.2.
INPUT_SIZE = 512
BATCH_SIZE = 4
OPTIMIZER = "Adam"
LR = 5e-4
WEIGHT_DECAY = 1e-5

# Section 3.1. Cardinality K and radix R.
CARDINALITY = 2
RADIX = 2
# Printed equations for the two ResNeSt outputs. At a 512 input these are
# 128x128x256 and 64x64x512.
C2_CHANNELS = 256
C3_CHANNELS = 512
# Figure 2 label for C1. The text does not print this channel count. A1.
C1_CHANNELS = 128

# Section 3.1. MHSSA uses four heads.
NUM_HEADS = 4

# Section 4.2. DCFL coefficients.
CYCLICAL_FACTOR = 2.0
GAMMA_HIGH = 3.0
GAMMA_LOW = 2.0
# Equation (10). The offset inside the logarithm is written as 1.10.
BETA_LOG_OFFSET = 1.10

# Table 1 column order, with background made explicit. A22.
FUNDUS_CLASSES = ("background", "EX", "HE", "MA", "SE")
NUM_CLASSES = len(FUNDUS_CLASSES)

# The paper input is one RGB image. No vessel map, no text, no prior mask.
EXTRA_INPUTS: tuple[str, ...] = ()
INPUTS = ("RGB image",)

# Online publication date, 13 May 2023. Public seed. A20.
SEED = 20230513

# A15. en in equation (11). Figure 12 says the loss converged after 1000 epochs.
# The paper does not list an epoch hyperparameter separately.
EPOCHS = 1000

# A16. Adam (Kingma and Ba, 2015) defaults. The paper gives lr and weight decay only.
ADAM_BETAS = (0.9, 0.999)
ADAM_EPS = 1e-8

# A4 and A10. Transformer width. The paper never prints Ct or this width.
TF_CHANNELS = 128
# A3. Three halvings from C3=512 give 256, 128, then 64. Figure 2 prints the first two.
CNN_OUT_CHANNELS = 64

# A9. UTNet, the cited MHSSA source, constructs the model with reduce_size=8.
# TC-Net only says that K and V are bilinearly downsampled.
KV_REDUCE_SIZE = 8
ATTN_DROPOUT = 0.0

# A6 and A8.
MLP_RATIO = 4

# A2. Hidden width of the two fully connected layers inside split attention.
# ResNeSt uses reduction 4 and a floor of 32. TC-Net prints neither.
SPLIT_REDUCTION = 4
SPLIT_MIN_HIDDEN = 32

# A11. Equation (6) is evaluated on this grid. 64 is the C3 grid of a 512 input.
SPATIAL_ATTN_SIZE = 64

# A13. Hidden width of the fusion fully connected layers, as a fraction of Cc+Ct.
FUSION_HIDDEN_RATIO = 1

# A14. The 3x3 in the segmentation head keeps the LLCS channel count.
HEAD_MID_CHANNELS = CNN_OUT_CHANNELS + TF_CHANNELS

# A17.
PT_EPS = 1e-6

# A19. Section 4.2 names these transforms and does not give magnitudes.
# They are locked here and are not wired to a dataset in this step.
FLIP_PROBABILITY = 0.5
COLOR_JITTER = (0.2, 0.2, 0.2, 0.05)  # brightness, contrast, saturation, hue
AFFINE_DEGREES = 15.0
AFFINE_TRANSLATE = 0.1
AFFINE_SCALE = (0.9, 1.1)

# Table 6 and the "Ours" rows of Tables 2 and 3. Post-hoc only.
POSTHOC = {
    "params": 46.9e6,
    "gflops": 414.0,
    "inference_ms": 133.3,
    "gpu_memory_mb_batch2": 13011.0,
    "idrid_macc": 0.6985,
    "idrid_miou": 0.5696,
    "idrid_mkappa": 0.6387,
    "idrid_mdice": 0.6968,
    "ddr_macc": 0.5171,
    "ddr_miou": 0.3969,
    "ddr_mkappa": 0.4070,
    "ddr_mdice": 0.5092,
    "note": "Numbers reported by the paper. Not used to choose structure or hyperparameters.",
}


def adam_kwargs() -> dict:
    return {
        "lr": LR,
        "betas": ADAM_BETAS,
        "eps": ADAM_EPS,
        "weight_decay": WEIGHT_DECAY,
    }


ASSUMPTIONS = (
    {
        "id": "A1",
        "text": (
            "The first CNN layer is a 3x3 stride-1 convolution with 128 channels, "
            "then BN, ReLU, and 3x3 max-pooling with stride 2. That is C1 in Figure 2, "
            "256x256x128 at a 512 input. The text only says convolution, BN, ReLU, and that pool."
        ),
    },
    {
        "id": "A2",
        "text": (
            "Each split is 1x1, BN, ReLU, 3x3, BN, ReLU. The K cardinals do not share weights. "
            "The two fully connected layers in split attention have a ReLU between them and a "
            "hidden width of max(C/4, 32). Softmax is over the radix and is channel-wise. "
            "The 1x1 after concatenation has BN, and ReLU follows the residual add. K=2 and R=2 are printed."
        ),
    },
    {
        "id": "A3",
        "text": (
            "Each of the three CNN decoder stages bilinearly upsamples by 2 and halves channels. "
            "Stage 1 concatenates C2. Stage 2 concatenates C1. Stage 3 has no shallower skip. "
            "Cc is therefore 64. The text only writes the C2 concatenation; the C1 skip is the "
            "dashed link in Figure 2."
        ),
    },
    {
        "id": "A4",
        "text": (
            "The Transformer basic block is two stride-2 3x3 convolutions (BN, ReLU) and one "
            "stride-2 3x3 max-pool, with 128 channels. At a 512 input, X is 64x64x128. "
            "The text says a series of convolutions, BN, and 3x3 max-pooling."
        ),
    },
    {
        "id": "A5",
        "text": (
            "The four encoder Transformer blocks keep the basic-block resolution and emit "
            "X2, X3, X4, X5. No downsampling between those blocks is written."
        ),
    },
    {
        "id": "A6",
        "text": (
            "The residual block inside an encoder Transformer block is two 3x3 convolutions "
            "with BN, ReLU after the first, and ReLU after the identity add."
        ),
    },
    {
        "id": "A7",
        "text": (
            "Norm in the Transformer block of Figure 2 is LayerNorm over channels at each "
            "position. LN in the LLCS equation is the same."
        ),
    },
    {
        "id": "A8",
        "text": "The MLP is two 1x1 convolutions, hidden width 4C, GELU in between. The text only says MLP.",
    },
    {
        "id": "A9",
        "text": (
            "K and V are bilinearly reduced to 8x8, the UTNet constructor default. "
            "Figure 3 draws Dropout and gives no rate, so the rate is 0. "
            "Q, K, and V are three separate 3x3 convolutions with bias. "
            "d in sqrt(d) is the per-head channel count. B is the Swin relative-position bias, "
            "added after the 1/sqrt(d) scaling, as in equation (4). align_corners is False."
        ),
    },
    {
        "id": "A10",
        "text": (
            "Of the four decoder blocks, the first is the written cross-attention: Q from X4, "
            "K and V from X5. The next three match the encoder self-attention block. "
            "One bilinear resize back to the input resolution and one 3x3+ReLU produce Dt "
            "at 512x512x128. The overview figure draws a single 3x3 and ReLU."
        ),
    },
    {
        "id": "A11",
        "text": (
            "The spatial path implements equation (6), softmax over i. Equation (5) global "
            "pooling cannot be reshaped to N by Cc/2, so it is not implemented. Dc is adaptive-"
            "average-pooled to 64x64 before the NxN map, and Os is bilinearly returned to Dc. "
            "64 is the C3 grid. The prose says sigmoid; the numbered equation says softmax."
        ),
    },
    {
        "id": "A12",
        "text": (
            "In the channel path, D1t dot Sc is a Ct/2 vector. The 1x1 maps it to Ct, then "
            "LayerNorm and sigmoid, then a per-channel multiply with Dt."
        ),
    },
    {
        "id": "A13",
        "text": (
            "The fusion path's two fully connected layers have hidden width Cc+Ct, ReLU between "
            "them, and sigmoid after. The text only says two fully connected layers followed by sigmoid."
        ),
    },
    {
        "id": "A14",
        "text": (
            "The segmentation-head 3x3 keeps the 192 channels of O. The 1x1 emits class logits. "
            "Softmax sits inside DCFL."
        ),
    },
    {
        "id": "A15",
        "text": (
            "en in equation (11) is 1000, because Figure 12 says the loss converged after 1000 "
            "training epochs. No separate epoch hyperparameter is printed."
        ),
    },
    {
        "id": "A16",
        "text": (
            "Adam uses beta1=0.9, beta2=0.999, eps=1e-8. The learning rate stays at 5e-4. "
            "The text gives an initial learning rate and no schedule."
        ),
    },
    {
        "id": "A17",
        "text": (
            "beta uses the natural logarithm. The caller may pass pixel counts for the whole "
            "training set each epoch; otherwise the current target is used. The loss is the mean "
            "over pixels. log(pt) is floored at 1e-6. ei runs from 1 through en."
        ),
    },
    {
        "id": "A18",
        "text": (
            "Conv weights use Kaiming normal, fan_out, ReLU. BN scale is 1 and bias is 0. "
            "The paper says a pretrained ResNeSt block, but Figure 2's three widths and K=2 "
            "do not match ResNeSt-50 tensors, so those weights are not loaded and the width "
            "is not changed to chase the 46.9M parameter count."
        ),
    },
    {
        "id": "A19",
        "text": (
            "Horizontal flip probability 0.5. Color jitter amplitudes 0.2, 0.2, 0.2, 0.05. "
            "Affine rotation +/-15 degrees, translation 0.1, scale 0.9 to 1.1. No dataset is attached."
        ),
    },
    {
        "id": "A20",
        "text": "The random seed is 20230513, the article's online publication date.",
    },
    {
        "id": "A21",
        "text": "There is no extra input. Both the fundus and skin experiments take RGB only.",
    },
    {
        "id": "A22",
        "text": (
            "Fundus class order is background, EX, HE, MA, SE, matching Table 1. "
            "The skin experiment is the same network with num_classes=2 and the same RGB input."
        ),
    },
)
