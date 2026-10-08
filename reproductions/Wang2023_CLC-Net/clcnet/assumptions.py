"""Locked constants for the CLC-Net reproduction.

Paper-stated values stay here so training cannot drift from section 4.1.
Assumption values are the details the paper leaves open. They are fixed
in this file and explained in DESIGN.md. They are not tuned to Table 1–3.
"""

from __future__ import annotations

PAPER = {
    "id": "M-CLC-2023",
    "title": "CLC-Net: Contextual and local collaborative network for lesion segmentation in diabetic retinopathy images",
    "authors": "Xiyue Wang, Yuqi Fang, Sen Yang, Delong Zhu, Minghui Wang, Jing Zhang, Jun Zhang, Jun Cheng, Kai-yu Tong, Xiao Han",
    "year": 2023,
    "venue": "Neurocomputing 527:100-109",
    "doi": "10.1016/j.neucom.2023.01.013",
    "pdf": "literature/papers/03_dr_methods/cnn_multiscale/Wang2023_CLC-Net.pdf",
}

# Section 4.1 and Figure 2. These are written in the paper.
LOCAL_SIZE = 256
CONTEXT_SIZE = 512
PATCH_STRIDE = 128
BATCH_SIZE = 16
EPOCHS = 60
OPTIMIZER = "Adam"
LR_INITIAL = 3e-4
LR_GAMMA = 0.1
# 1-indexed epochs on which the new learning rate starts.
LR_MILESTONES = (20, 50)
SEGMENTATION_CLASSES = ("background", "HE", "MA", "EX", "SE")
# Empirical weights in section 3.1, in the class order above.
SEGMENTATION_WEIGHTS = (1.0, 2.0, 2.0, 2.0, 2.0)
# Section 3.2 worked example 1110 = MA, EX, HE present and SE absent.
CLASSIFICATION_CLASSES = ("MA", "EX", "HE", "SE")
NUM_LESIONS = 4
REPORTED_PARAMS = 92.19e6
REPORTED_FLOPS = 20.58e9
EXTRA_INPUTS: tuple[str, ...] = ()
INPUTS = ("RGB fundus photograph",)

SEED = 20230108

# A2. Decoder block channel rule. See DESIGN.md before changing.
BOTTLENECK_CHANNELS = 2048
SKIP_CHANNELS = (1024, 512, 256, 64)  # layer3, layer2, layer1, stem

# A3. SKNet citation defaults for the three kernels named in section 3.1.
SK_KERNELS = 3
SK_GROUPS = 32
SK_REDUCTION = 16
SK_MIN_CHANNELS = 32
SK_DILATIONS = (1, 2, 3)

# A8.
CLS_HIDDEN = 512

# A10.
DICE_EPS = 1.0

# A11. Adam paper defaults. Weight decay is unspecified, so it stays 0.
ADAM_BETAS = (0.9, 0.999)
ADAM_EPS = 1e-8
ADAM_WEIGHT_DECAY = 0.0

# A12.
FLIP_PROBABILITY = 0.5
ROTATION_DEGREES = (0.0, 360.0)
SHIFT_FRACTION = 0.1
SCALE_RANGE = (0.8, 1.2)

# A17.
EOPHTHA_N_TRAIN = 15
EOPHTHA_N_TEST = 6
EOPHTHA_EXPECTED = 21

# Overlap priority for the exclusive training label (A9), highest first.
OVERLAP_PRIORITY = ("MA", "HE", "EX", "SE")

PREPROCESS_ID = "otsu-luminance-bbox-v1"


def learning_rate(epoch: int) -> float:
    """Return the Adam learning rate for a 1-indexed epoch."""
    if epoch < 1 or epoch > EPOCHS:
        raise ValueError(f"epoch must be in 1..{EPOCHS}, got {epoch}")
    lr = LR_INITIAL
    for milestone in LR_MILESTONES:
        if epoch >= milestone:
            lr *= LR_GAMMA
    return lr


def decoder_schedule():
    """Channel counts implied by halving-then-concat.

    Returns (up_in_channels, feature_channels). feature_channels are the
    decoder maps after skip concatenation, at strides 16, 8, 4, 2.
    """
    current = BOTTLENECK_CHANNELS
    up_in = []
    feats = []
    for skip in SKIP_CHANNELS:
        up_in.append(current)
        mid = current // 2
        if mid * 2 != current:
            raise RuntimeError(f"channel count {current} is not even")
        current = mid + skip
        feats.append(current)
    return tuple(up_in), tuple(feats)


UP_IN_CHANNELS, DECODER_CHANNELS = decoder_schedule()


def class_index(name: str) -> int:
    return SEGMENTATION_CLASSES.index(name)


ASSUMPTIONS = (
    {
        "id": "A1",
        "text": "编码器是 SE-ResNeXt-50（32×4d），SE 缩减比 16，Kaiming 随机初始化，不加载 ImageNet 权重。BatchNorm 用 PyTorch 默认 momentum 0.1、eps 1e-5。",
    },
    {
        "id": "A2",
        "text": "解码块 Conv2d 为 1×1 且把通道减半，接着 BN 和 ReLU；SKConv 通道数不变；TransConv2d 为 2×2、stride 2、带偏置，只把空间尺寸加倍。",
    },
    {
        "id": "A3",
        "text": "SKConv 三个分支是膨胀率 1、2、3 的 3×3 分组卷积，G=32，r=16，L=32，按被引用的 SKNet 融合。",
    },
    {
        "id": "A4",
        "text": "四个解码尺度是输入的 1/16、1/8、1/4、1/2。跳跃连接来自 layer3、layer2、layer1 和 stride-2 的 stem。",
    },
    {
        "id": "A5",
        "text": "每个解码特征 1×1 投影到 5 类，双线性上采样到 256，拼接后再 1×1。分割损失只加在最终 logits 上。",
    },
    {
        "id": "A6",
        "text": "上下文 512 图像双线性缩到 256，align_corners=False。上下文标签按 2×2 最大池化，再用重叠优先级合成单标签。",
    },
    {
        "id": "A7",
        "text": "融合发生在跳跃拼接之后。SKM 作用在上下文特征上，与局部特征拼接，再经 1×1+BN+ReLU 压回原通道数。拼接消融跳过 SKM，保留这次投影。",
    },
    {
        "id": "A8",
        "text": "分类头是全局平均池化、全连接 2048→512、ReLU、全连接 512→4。顺序为 MA、EX、HE、SE。任一正像素记为 1。",
    },
    {
        "id": "A9",
        "text": "分割类别顺序是背景、HE、MA、EX、SE。重叠像素的优先级是 MA > HE > EX > SE。",
    },
    {
        "id": "A10",
        "text": "Dice 的 ε=1，加在分子和分母，对四个病灶类求和，在整个 mini-batch 上计算。wMCE 对像素取平均。分支损失系数都是 1。",
    },
    {
        "id": "A11",
        "text": "Adam 的 β1=0.9，β2=0.999，ε=1e-8，weight decay=0。第 1–19 轮 3e-4，第 20–49 轮 3e-5，第 50–60 轮 3e-6。报告第 60 轮。",
    },
    {
        "id": "A12",
        "text": "增强作用在 512 上下文补丁上，再取中心 256。翻转概率各 0.5，旋转 [0,360) 度，平移 ±10% 边长，缩放 [0.8,1.2]。",
    },
    {
        "id": "A13",
        "text": "Otsu 用亮度且不模糊。边缘多数为前景则反相。最大连通域、填洞、裁外接矩形。视野外图像像素置 0，标注不擦除。不能整除步长时补贴边窗口。无视野的局部窗口丢弃。越界补 0。",
    },
    {
        "id": "A14",
        "text": "像素除以 255。不做均值方差归一化。",
    },
    {
        "id": "A15",
        "text": "推理把局部分支 softmax 概率按重叠平均。主结果来自局部分支。上下文头评估时把 256 概率双线性放大到 512 再拼接。窗口外背景概率为 1。",
    },
    {
        "id": "A16",
        "text": "AUPR 把该划分全部像素放在一起计算。并列分数用稳定排序。Dice 和 IoU 用 argmax，对照原始二值掩膜。e-ophtha 只计算 MA 和 EX。",
    },
    {
        "id": "A17",
        "text": "e-ophtha 取同时有 MA 和 EX 的 21 张图。排序后用 NumPy PCG64(20230108) 打乱，前 15 训练、后 6 测试，并写入划分文件。IDRiD 与 DDR 用官方划分。",
    },
    {
        "id": "A18",
        "text": "种子 20230108 用于 Python、NumPy、PyTorch。每个增强样本使用 SeedSequence([种子, 轮次, 下标])。",
    },
    {
        "id": "A19",
        "text": "补零像素参与损失，其标签是背景。",
    },
    {
        "id": "A20",
        "text": "训练时丢掉不足 16 张的最后一个 batch。",
    },
    {
        "id": "A21",
        "text": "额外输入没有。模型只接收 RGB 眼底图像。",
    },
    {
        "id": "A22",
        "text": "保存第 20、50 和 60 轮的权重。对外结果使用第 60 轮，不用验证集挑选。",
    },
)


class Variant:
    def __init__(self, name, use_local, use_context, fusion, use_cls):
        if fusion not in ("none", "concat", "skm"):
            raise ValueError(fusion)
        if fusion != "none" and not (use_local and use_context):
            raise ValueError("fusion requires both branches")
        self.name = name
        self.use_local = use_local
        self.use_context = use_context
        self.fusion = fusion
        self.use_cls = use_cls

    def as_dict(self):
        return {
            "name": self.name,
            "use_local": self.use_local,
            "use_context": self.use_context,
            "fusion": self.fusion,
            "use_cls": self.use_cls,
        }


# Table 1 training settings. The two "Ours" rows share the full variant;
# evaluation chooses the local or the contextual head.
VARIANTS = {
    "loc": Variant("loc", True, False, "none", False),
    "loc_cl": Variant("loc_cl", True, False, "none", True),
    "cont": Variant("cont", False, True, "none", False),
    "cont_cl": Variant("cont_cl", False, True, "none", True),
    "loc_cont": Variant("loc_cont", True, True, "concat", False),
    "loc_cont_att": Variant("loc_cont_att", True, True, "skm", False),
    "full": Variant("full", True, True, "skm", True),
}
