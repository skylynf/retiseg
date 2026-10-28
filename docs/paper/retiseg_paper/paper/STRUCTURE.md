# 论文结构与图表清单

题名：Assessment of computational methods for diabetic retinopathy lesion segmentation in fundus images
体例模板：Lu 等，Nature Methods 23:248–259（`literature/papers/00_template/`）。
依据：`docs/paper/13`（总纲）、`docs/paper/14`（方法素材 M1–M14）、`docs/design/05`、`docs/design/12`、`bench/configs/*.yaml`。
本文件只定“写什么、放在哪、数据从哪来”，不改任何协议。与 13 冲突时以 13 为准；实验范围以 `docs/paper/16`（2026-10-11 范围精简）为准，下表的面板已按 16 第四节改过。

状态标记：**实** = 图表里的数字已经存在于仓库（文档、配置或审计 JSON）；**拟** = 目前是模拟占位，由 `scripts/build_placeholder.py` 生成，出数后替换。

## 一、正文骨架（与模板逐节对应）

| 本文 | 模板对应 | 回答 | 实验 | 图 | 表 |
|---|---|---|---|---|---|
| 引言（4 段） | Introduction | — | — | — | — |
| R1 数据收集与研究设计 | Data collection and study design | — | 文献流程、models.yaml | 图 1 | 表 1、表 2 |
| R2 文献里的数字 | （本文新增，接在 R1 后） | Q1 | L1、L2、E1 的评测实现对照 | 图 2 | — |
| R3 原始模型 | Performance of original models | Q1 | E1、E1r | 图 3 | 表 3 |
| R4 统一重训：内部测试与独立测试 | Performance of retrained models | Q2 | E2、nnU-Net、E3a | 图 4 | 表 4、表 5 |
| R5 基础模型 | （本文新增） | Q2 | E5 | 图 5 | 表 6 |
| R6 标注粒度与训练来源 | Source effects / Cross effects | Q3 | 标注风格审计、E3b、E4 | 图 6 | 表 7（上半） |
| R7 病灶大小、分辨率与训练量 | Size effects / P-to-N ratios | Q3 | E7、E8、病灶大小分层 | 图 7 | 表 7（下半） |
| R8 计算代价 | Computational efficiency | — | E10 | 扩展数据图 10 | 补充表 7 |
| 讨论 | Discussion | Q1–Q3 | — | — | — |
| 方法 | Methods | — | M1–M14 | — | 方法表（各模型在 B-std 下保留的设定） |

Nature Methods 的 Analysis 正文通常不放表。表 1–7 按 CVPR/ICLR 风格排好，投稿时表 1、2、4 可作 Extended Data Table，其余进补充表；编号在 `main.tex` 里只改一处宏。

## 二、主图（7 张）与面板

| 图 | 面板 | 内容 | 形式（模板出处） | 数据 | 状态 |
|---|---|---|---|---|---|
| 1 研究设计 | a | 流程：数据 → 模型 → 四种评价 → 两种测试场景 → 指标 | 示意图（模板图 1a） | — | 实 |
|  | b | 7 个测试来源的划分与粒度档 | 分段条形 | M3 划分数 | 实 |
|  | c | 38 个模型变体：模型族 × 复现等级，分层 | 堆叠条形 | models.yaml | 实 |
|  | d | 检索与筛选流程 | 漏斗 | literature/06、08 | 实 |
|  | e | 影响因素一览（Q3 各实验改动的唯一因素） | 示意图（模板图 1b） | 05 第四节 | 实 |
| 2 文献里的数字 | a | 复现漏斗：登记 → 有代码 → 有训练代码 → 有权重 → 复现 | 漏斗 | models.yaml；L2 | 前四级实，末级拟 |
|  | b | 各论文的划分（IDRiD、DDR）、AUPR 实现、基线数字来源（重跑、一代转引、二代以上、无法核对） | 四行堆叠条形 | L1：`literature/meta/paper_audit.tsv`、`reported_results.tsv`，经 `scripts/collect_literature.py` | 实（2026-10-11） |
|  | c | 同一方法在不同论文里被报出的 IDRiD mAUPR 范围（≥3 篇的方法）加本项目重训值 | 区间图 | L1；E2 | 文献值实，重训值拟 |
|  | d | 同一批概率图：11 阈值实现 对 精确 AUPR | 哑铃图 | E1；全部模型的概率图 | M2MRF 实，其余拟 |
|  | e | 两种实现下的名次变化 | 名次连线 | 同上 | 实 |
| 3 原始模型 | a | 每个作者权重的三个数：论文、作者函数、本项目协议 | 哑铃图 | E1 | M2MRF 实 |
|  | b | 作者权重逐类 AUPR，官方测试加独立测试 | 热图加总分条（模板图 2e） | E1 | IDRiD 实，其余拟 |
|  | c | 作者配方重训对论文数字（M2MRF IDRiD 3 种子、HRDecoder IDRiD 1 种子；砍掉的 M2MRF DDR 与 HACDR-Net 标 not rerun） | 点加参考线 | E1r | 实（10-28 起三种子齐） |
|  | d | 作者权重：内部对独立 | 配对连线（模板图 4） | E1 | 实 |
| 4 统一重训 | a | 内部测试逐类 AUPR（IDRiD、DDR），旁边是 mAUPR 条 | 热图加条（模板图 2e、3b） | E2 | 实 |
|  | b | 模型 × 7 个测试集，按训练来源分两块 | 点阵加总分列（模板图 3b、3d） | E2、E3a | 实 |
|  | c | 内部对独立，每个模型一条线 | 配对连线加检验（模板图 4） | E3a | 实 |
|  | d | 四个模型族的比较 | 散点加均值 ± s.d.（模板图 3c） | E2、E3a | 实 |
|  | e | 临界差图，按 7 个测试集 | CD 图 | E2、E3a | 实 |
|  | f | 论文名次对重训名次，Kendall τ（10 个模型的文献值：专用模型取原文，其余取账本中位数） | 名次连线 | L1、E2 | 文献值实，重训值拟 |
| 5 基础模型 | a | 四档用法逐类 AUPR | 热图 | E5 | 实 |
|  | b | 理想提示：提示数量—性能（SAM ViT-B、MedSAM、SAM 2.1 框提示，k ∈ {1, 5, 20, 全部}；SAM 2.1 另有点提示） | 折线（模板图 5d） | E5b | 实 |
|  | c | SAM 3 文本提示的措辞敏感性 | 点图 | E5a | 实 |
|  | d | 同一 UPerNet 头下四种编码器（RETFound-MAE、MAE-IN1k、RETFound-DINOv2、DINOv2），只在 DDR 训练，内部与独立 | 哑铃图 | E5d | 实 |
|  | e | 各档最好的基础模型对专用模型和 nnU-Net | 条形 | E2、E5 | 实 |
| 6 标注与来源 | a | 标注风格审计：每图病灶数 × 平均面积 | 散点（Playout 2024 图 2） | label_audit | IDRiD、DDR 的病灶数实，其余拟 |
|  | b | 训练集 × 测试集矩阵，表头标粒度档（5 个模型，种子 0） | 热图 | E2、E3b | 实 |
|  | c | 相对同库训练的损失 | 发散热图 | 同上 | 实 |
|  | d | E4 六档训练集在细标注测试上的表现（U-Net、M2MRF，种子 0） | 配对点线 | E4 | 实 |
| 7 大小、分辨率、训练量 | a | 四类病灶的相对直径分布与小、中、大边界 | 分位区间 | lesion_size_freeze.json | 实 |
|  | b | 缩小输入后在训练目标里消失的病灶比例 | 分组条形 | input_label_audit.json | 实 |
|  | c | 按病灶大小分层的召回，逐模型 | 热图 | E2 病灶级指标 | 实 |
|  | d | 输入分辨率 512、1440、原分辨率块：mAUPR 与小病灶召回（M2MRF、RETFound 加 UPerNet） | 折线 | E7 | 实 |
|  | e | 训练量 10%、50%、100% 与细粒度合并（U-Net、M2MRF） | 折线（模板图 5d） | E8 | 实 |

扩展数据图（2026-10-26 状态）：ED1 各库示例与标注风格（需图像，工作站画）；ED2 视野与画布（同）；ED3 泄漏审计（已做，以补充表 15 交代，不单独作图）；ED4 评测实现敏感性（并入图 2d/e）；ED5 逐图分布（撤销，直方图留仓库）；ED6（取消；B1 在任何运行完成前停止）；**ED7 学习率扫描（只有 HRNet / M2MRF 的 DDR，实）**；**ED8 损失（U-Net 四种损失，实）**；**ED9 DiaRetDB1 官方图像级协议 + 超广角两种推理尺度（实）**；**ED10 计算代价（参数量、可训参数、两源 GPU 时，实；FLOPs / 显存 / 推理时间等算力审计）**。补充材料：单独的 `supplement.tex`（`make supplement`），18 张表（第 18 张是正文实验名称与仓库代号的对照表）+ 3 个说明 + 4 项数据，全部由 `scripts/make_supplement.py` 从 results.csv 生成；清单见 `sections/extended.tex` 与 14 备注 66。

## 三、表（LaTeX，`tables/generated/`）

| 表 | 内容 | 数据 | 状态 |
|---|---|---|---|
| 1 | 数据集总表：来源、分辨率、视野、标注方式、粒度档、划分、角色、能否公开概率图 | M1、M3、M14 | 实 |
| 2 | 模型总表（第一、二层 25 个）：模型族、复现等级、轨、代码、权重、用法 | models.yaml | 实 |
| 3 | 原始模型三个数加作者配方重训 | E1、E1r | 实（M2MRF 三种子、HRDecoder 一种子） |
| ST3 | 原文数字抽取：每篇的设定（`paper_audit.tsv`）和全部对比数字及来源（`reported_results.tsv`） | L1 | 实 |
| 4 | **主表**：B-std 内部测试，逐类 AUPR、mAUPR ± 种子 s.d.、Dice、参数量 | E2 | 实 |
| 5 | 独立测试：5 个独立测试集、内部减独立、平均名次 | E3a | 实 |
| 6 | 基础模型四档 | E5 | 实 |
| 7 | 影响因素汇总：E4 粒度、E7 分辨率、E8 训练量、E9 损失（按 16 缩减后的模型和档位） | E4、E7–E9 | 实 |
| 方法表 | 各模型在 B-std 下保留的设定与推理几何 | author_recipe、bs_frozen.yaml | 实 |

补充表 ST1–ST7 的定义不变（见 13 第五节）；ST5（全部模型 × 全部测试集 × 逐类）由同一份长表直接导出。

## 四、数据流

```
runs/*/metrics_test.json ──collect_runs.py──┐   （environment.json → E10 参数量和训练时；data/cost_measured.tsv、cost_audit.tsv 也由它读入）
literature/meta/*.tsv ──collect_literature.py┤
                                            ├─> data/results.csv（长表，每行带 source=measured|placeholder）
scripts/build_placeholder.py ───────────────┘
        │
        ├─> figures/*.py      → figures/out/*.pdf|png（用到占位数据的面板自动加“simulated”角标）
        ├─> scripts/make_tables.py  → tables/generated/*.tex（占位数字自动灰蓝色）
        └─> scripts/make_numbers.py → generated/numbers.tex（正文里的数字宏）
```

长表的键是（experiment, model, variant, train_set, seed, test_set, cls, metric）。`collect_runs.py` 只按键覆盖，不改其他行。

## 五、需要确认的三件事

1. **排名检验的区组。** 12 写“按 7 个测试集”。IDRiD、DDR 两个内部区组各用对应训练来源的检查点；5 个独立区组这里取“IDRiD 训练与 DDR 训练两组检查点的平均”。另一种做法是只用 DDR 训练的检查点，或拆成 12 个（训练, 测试）区组。`figures/retifig/data.py::block_scores` 一处可改。
2. **DiaRetDB1 进排名时用什么分数。** 这里用官方图像级 ROC AUC（四类平均，只取 DiaRetDB1 这一集的 `img_auc` 行：`retifig.data.diaretdb1_auc`，2026-10-26 之前误把八个集的图像级 AUC 平均进来了）；Friedman 只看区组内名次，各区组的指标不必相同。像素 AUPR 单列并标“区域真值”。
3. **H2Former。** 第一批不跑，所以主表和所有图里没有它；表 2 保留并注明原因。若以后补跑，在 `scripts/build_placeholder.py::MAIN_MODELS` 加一行即可。
