# TC-Net 设计记录

Zhang 等，Computers in Biology and Medicine 161 (2023) 106967。doi:10.1016/j.compbiomed.2023.106967。

这次只固定网络、DCFL 和论文写明的训练超参数。没有数据划分，没有训练，也没有用表 2、表 3、表 6 的数字去选择未写明的宽度或阈值。原文数字记在 `POSTHOC` 里，只供以后对照。

代码在 `tcnet/`。引用的 ResNeSt 与 UTNet 克隆在 `official_code/ResNeSt` 和 `official_code/UTNet`，它们是文中引用的模块，不是 TC-Net 作者的仓库。全文没有 TC-Net 的代码链接。

## 输入

论文实验只有一种输入：RGB 图像。眼底和皮肤都是这样。没有血管、文本或先验掩膜。

`TCNet.extra_inputs` 是空元组。前向只接受 `(N, 3, H, W)`。结果里应写成 `input=RGB`。

第 4.2 节把图像缩放到 512×512。实现要求边长是 64 的倍数，这样各次 stride-2 和 8×8 的键值网格能整除。论文设定就是 512。

眼底类别顺序固定为 `background, EX, HE, MA, SE`，对应表 1 的 BK、EX、HE、MA、SE。默认 `num_classes=5`。皮肤实验是同一个网络、`num_classes=2`，输入仍然只有 RGB。

## 论文已经写明、代码直接采用的部分

- 混合编码器、混合解码器、LLCS、分割头。图 2。
- CNN 使用 ResNeSt block，K=2，R=2。C2 为 H/4×W/4×256，C3 为 H/8×W/8×512。
- 第一层是卷积、BN、ReLU，再加 3×3、stride 2 的 max pooling。
- 每个 split 两层卷积，各接 BN 和 ReLU。Cardinal 内做 split attention：先把各 split 相加并全局池化，两层全连接得到权重，再加权求和。公式 (1)(2)。
- 各 cardinal 拼接后做 1×1，并加上捷径 T。公式 (3)。
- Transformer 编码器：一个 basic block（卷积、BN、3×3 max pooling）和四个 Transformer block。输出记为 X2、X3、X4、X5。
- 每个 Transformer block 含 residual block 和 Transformer 模块。模块是 Norm、MHSSA、残差、Norm、MLP、残差。图 2(a)。
- MHSSA 用四个头。Q、K、V 由三路 3×3 卷积得到。K 和 V 双线性下采样。加上相对位置偏置 B。公式 (4)。图 3 在 softmax 之后有 Dropout。
- 解码器里的交叉注意力：Q 来自低层 X4，K 和 V 来自高层 X5，投影都是 3×3。
- CNN 解码器三档，每档双线性 ×2，卷积把通道减半。C3 上采样到 H/4 后与 C2 拼接，再两层 3×3、BN、ReLU。
- LLCS 三条路径。输出是 `Concat(Os, Oc) + Of`。公式 (8)。
- 分割头：3×3、ReLU、1×1，得到 H×W×c。第 3.4 节没有 softmax。
- DCFL：公式 (9)–(13)。r=2，γ1=3，γ2=2。
- 优化器 Adam，初始学习率 5×10^-4，weight decay 0.00001。batch size 4。
- 数据增强的种类：随机水平翻转、颜色变换、随机仿射。幅度没有写，见 A19。这次不接数据集。

## 结构

512 输入时的张量：

| 位置 | 形状 | 依据 |
|---|---|---|
| C1 | 256×256×128 | 图 2；通道与第一层卷积核是 A1 |
| C2 | 128×128×256 | 正文 |
| C3 | 64×64×512 | 正文 |
| CNN 解码 128² | 128×128×256 | 图 2，与 C2 拼接 |
| CNN 解码 256² | 256×256×128 | 图 2，与 C1 拼接，A3 |
| Dc | 512×512×64 | 第三档再减半，A3 |
| X 以及 X2…X5 | 64×64×128 | A4、A5 |
| Dt | 512×512×128 | A10 |
| LLCS 输出 O | 512×512×192 | 公式 (8) |
| logits | 512×512×5 | 第 3.4 节 |

CNN 与 Transformer 两条分支都从同一张 RGB 读入，不共享权重。

Transformer 解码的四个 block：第一块是 X4/X5 交叉注意力，后三块是自注意力。正文说解码器有四个 Transformer block 和一条跳连，公式只写了 X4 与 X5。后三块因此保持和编码器 block 相同，这是 A10。

## 损失

`pt` 是真值类别的 softmax 概率。

- `Fhc = -(1+pt)^γ1 log(pt)`，训练前段权重大。
- `Flc = -(1-pt)^γ2 log(pt)`，难样本权重大。
- `I(ei, en) = 1 - r·ei/en`，当 `r·ei ≤ en`；否则 `(r·ei/en - 1)/(r-1)`。
- `βk = 1 / log(1.10 + 类别 k 的像素比例)`。比例的分母是各类像素数之和，与公式 (10) 一致。
- `DCFL = I·Fhc + β·(1-I)·Flc`，再对像素平均。

r=2、en=1000 时，I 在第 1 个 epoch 接近 1，第 500 个 epoch 为 0，第 1000 个 epoch 回到 1。ei 从 1 计到 en。

β 按论文是每个 epoch、每个类别由真值算出。`set_epoch_class_counts` 接收整个训练集的计数。没有传入时，用当前 `target` 计算，并把 `last_beta_source` 标成 `batch`。

## 固定假设

未写明的细节都停在下面这些值上，不再为了靠近原文表格去改。

| 编号 | 固定内容 |
|---|---|
| A1 | 第一层 3×3、stride 1、128 通道，再 maxpool。得到图 2 的 C1 |
| A2 | split 为 1×1 加 3×3；两层全连接中间 ReLU，隐藏宽度 max(C/4, 32)；按通道做 radix softmax；相加后 ReLU |
| A3 | 第二档拼接 C1；第三档无跳连；Cc=64 |
| A4 | basic block：两层 stride-2 的 3×3 和一层 stride-2 maxpool，128 通道，512 输入下为 64×64 |
| A5 | 四个编码器 block 不再下采样 |
| A6 | residual block 为两层 3×3 |
| A7 | Norm 为通道上的 LayerNorm |
| A8 | MLP 隐藏宽度 4 倍，GELU，用 1×1 卷积实现 |
| A9 | K、V 收到 8×8（UTNet 构造函数的默认边长）；Dropout 比例 0；d 为每个头的通道数；B 用 Swin 相对位置；先缩放再加 B |
| A10 | 交叉注意力只用 X4、X5；随后三个自注意力 block；一次双线性回到全分辨率，再 3×3+ReLU；Ct=128 |
| A11 | 空间路径实现公式 (6)，不实现与之冲突的公式 (5)；在 64×64 上计算 N×N，再插值回去；softmax 沿 i |
| A12 | 通道路径的 1×1 把 Ct/2 映射到 Ct |
| A13 | 融合路径两层全连接的隐藏宽度等于 Cc+Ct，中间 ReLU |
| A14 | 分割头 3×3 保持 192 通道；输出 logits |
| A15 | en=1000。图 12 写 1000 个 epoch 后收敛，论文没有另给 epoch |
| A16 | Adam β1=0.9，β2=0.999，ε=1e-8；学习率不衰减 |
| A17 | β 用自然对数；可传入 epoch 计数，否则用当前 target；像素平均；log(pt) 下界 1e-6 |
| A18 | Kaiming 初始化。不加载 ResNeSt-50。官方 ResNeSt-50 是 K=1 的 deep stem，张量形状对不上图 2 |
| A19 | 翻转概率 0.5；颜色幅度 0.2/0.2/0.2/0.05；仿射 ±15°、平移 0.1、缩放 [0.9, 1.1]。未接数据 |
| A20 | 种子 20230513 |
| A21 | 无额外输入 |
| A22 | 类别顺序 background、EX、HE、MA、SE |

公式之间有三处不能同时字面成立，所以上表里的 A11 必须写明：

1. 公式 (5) 把 D1 全局池化成一个向量，紧接着的句子又要求把它 reshape 成 N×Cc/2。
2. 公式 (6) 是 N×N。Dc 若真是 512×512，这个矩阵单张 float32 约 256 GB，和 24 GB 的 RTX 3090 不能同时成立。
3. 空间路径的散文写 sigmoid，公式 (6) 写 softmax。

因此空间路径跟公式 (6)，并把特征先收到 C3 那一档的 64×64。64 不是拿表 6 的 13 GB 或 46.9 M 参数拟合出来的。

“followed by a Swin-Transformer” 按图 2 理解成相对位置偏置 B 的来源。图里的 Transformer block 只有一个 MHSSA，没有第二套窗口注意力。

## 随机种子

`SEED = 20230513`，是论文在线发表日。`set_seed` 会设置 `PYTHONHASHSEED`、Python `random`、NumPy（若已安装）和 PyTorch。

## 原文数字，只供事后对照

这些数在 `tcnet.assumptions.POSTHOC`。训练代码不读取它们。

- 表 6，IDRiD，MHSSA：46.9 M 参数，414 GFLOPs，推理 133.3 ms，batch 2 约 13011 MB。
- 表 2，IDRiD，本文方法：mAcc 0.6985，mIoU 0.5696，mKappa 0.6387，mDice 0.6968。
- 表 3，DDR，本文方法：mAcc 0.5171，mIoU 0.3969，mKappa 0.4070，mDice 0.5092。
- 图 12：γ1=3、γ2=2 时，约 1000 个 epoch 后损失到 0.049。这里只借用 1000 作为 en，不把 0.049 当作训练目标。

参数量若和 46.9 M 不同，保留这个差，不回头加宽网络。

按现在锁住的宽度，CPU 上统计到的参数量是 14,898,918，约 14.90 M。其中 CNN 编码器 3.66 M，CNN 解码器 3.84 M，Transformer 分支 6.97 M，LLCS 0.10 M，分割头 0.33 M。表 6 的 46.9 M 大约是这个数的三倍。差出来的部分对应论文没写的 stage 深度和通道宽度，不往回改。

512×512 的一次前向得到 `(N, 5, 512, 512)`。这只核对张量形状，不是训练，也不是表 2 的复现。

## 这次没有做的事

- 没有 IDRiD、DDR 或 ISIC 的划分和读取。DDR 正文写了 483 张训练、274 张测试；IDRiD 只写了共 81 张。两者都还不是数据模块。
- 没有学习率搜索，没有按表 2 调 γ 或 β。
- 没有声明已经复现表中的 mAcc 或 FLOPs。
