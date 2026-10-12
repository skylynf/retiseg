# L1 原文数字抽取 — 抽取规范

目的：对每篇 DR 病灶分割方法论文，记录 (1) 它的评测设定，(2) 它的对比表里每个数字是谁的、从哪来的。
输入：`/tmp/claude-0/-home-claude/94f46ae8-cd76-54fb-9df1-120905963d86/scratchpad/l1txt/<paper>.txt`（pdftotext -layout 的输出；表格列可能错位，要结合表头和上下文仔细对齐；数字先在文中找到对应表格，再逐列核对）。
输出：两份 TSV（UTF-8，制表符分隔，第一行表头，不要加引号，空值写 `NA`）。

## 文件 1：`<group>_audit.tsv` —— 每篇论文一行

列（顺序固定）：
1. paper — 文本文件名去掉 .txt（如 Liu2023_M2MRF）
2. year — 发表年份
3. venue — 期刊或会议简称
4. datasets — 用到的眼底病灶分割数据集，分号分隔（如 `IDRiD;DDR;e-ophtha`）
5. idrid_split — `official` / `own` / `not stated` / `NA`（没用 IDRiD）。official = 用 IDRiD 官方 54 训练 / 27 测试；own = 自己重新划分（含交叉验证、把 81 张混合后划分）
6. idrid_split_detail — 原文怎么写的，简短引用（如 `54 train / 27 test (official)`；`81 images, 5-fold CV`）
7. ddr_split — 同上，DDR 官方是 383/149/225
8. ddr_split_detail
9. input_resolution — 训练/测试输入尺寸，原文写法（如 `resized to 1024x1024`；`patches 512x512 from original`；`not stated`）
10. fov_only — 评测是否只算视野内像素：`yes` / `no` / `not stated`
11. main_metric — 主指标名，原文写法（如 `AUC_PR`、`AUPR`、`Dice`、`F1`、`mAP`）
12. aupr_impl — AUPR/AUC-PR 的实现：`exact`（按全部阈值或 sklearn average_precision / precision_recall_curve 积分）/ `11-threshold`（固定 11 个或少量阈值，如 M2MRF/L-Seg 代码的做法）/ `per-image mean`（逐图算再平均）/ `not stated`（原文没说、也没有可查的代码说明）/ `NA`（主指标不是 AUPR）
13. aupr_impl_evidence — 判断依据，短引用或“code: <repo>”；没有则 NA
14. per_image_or_pooled — 指标是逐图平均（`per-image`）还是整个测试集像素合并（`pooled`）还是 `not stated`
15. code_stated — 论文是否声明公开代码：`yes (<url>)` / `no` / `not stated`
16. baseline_rerun_statement — 原文对对比方法数字来源的说明：`rerun all`（声明全部自己复现/重训）/ `rerun some`（部分自己跑、部分引用，写明哪些）/ `copied`（明确说引自原文）/ `not stated`。后面括号里写依据的原句（截短）
17. n_compared — 主对比表里对比方法（不含本文）的数目
18. notes — 其他值得记的（指标异常定义、只报三类、用了验证集挑阈值等）

## 文件 2：`<group>_results.tsv` —— 每个“论文 × 对比方法 × 数据集 × 类别 × 指标”一行

只抽 **主对比表**（本文方法 vs 其他方法，在 IDRiD、DDR、e-ophtha、FGADR、TJDR、Retinal-Lesions、DiaRetDB1 上的病灶分割结果）。不抽消融表、不抽分类/分级结果、不抽视盘/血管。
指标：抽 AUPR（AUC-PR / AP / mAP 等名字都算）为主；若该论文主指标不是 AUPR，则抽它的主指标（Dice 或 F1 或 IoU），`metric` 列照实写。同一表里既有 AUPR 又有 Dice 时两者都抽。

列（顺序固定）：
1. paper — 同上
2. table — 表号，原文写法（`Table 2`）
3. dataset — `IDRiD` / `DDR` / `e-ophtha` / `FGADR` / `TJDR` / `Retinal-Lesions` / `DiaRetDB1` / 其他原名
4. method — 表里的方法名，原文写法（本文方法写原文名字，如 `M2MRF` 或 `Ours (M2MRF)`）
5. method_norm — 归一化名，从下面清单选；不在清单里的写原名：
   `U-Net`, `U-Net++`, `FCN`, `SegNet`, `DeepLabv3`, `DeepLabv3+`, `PSPNet`, `HED`, `HRNet`, `UPerNet`, `Swin-Unet`, `TransUNet`, `Attention U-Net`, `CE-Net`, `L-Seg`, `CARNet`, `RTNet`, `M2MRF`, `PMCNet`, `HACDR-Net`, `WSRFNet`, `MLNet`, `VTA`, `TC-Net`, `CLC-Net`, `DRG-Net`, `PGAFT`, `SAA`, `LezioSeg`, `DB-UNet`, `MSAG`, `LFRC-Net`, `TEDR-YNet`, `CSIF`, `RILBP-YNet`, `TVFNet`, `SSMD-UNet`, `TP-DRSeg`, `SAM`, `SAM 2`, `MedSAM`, `SAMed`, `SAM-Adapter`, `nnU-Net`, `HEDNet+cGAN`, `Deep-multiscale (Guo2024)`, `Collaborative semi-supervised (Zhou2019)`, `Ophthalmology-SAM`, `SAM2-DR`
   同一架构带不同骨干（如 DeepLabv3+ ResNet50 / ResNet101）分两行，method 列保留原文骨干名，method_norm 相同。
6. is_proposed — `yes`（本文提出的方法）/ `no`
7. lesion — `MA` / `HE` / `EX` / `SE` / `mean`（表里的平均列）。注意同义：microaneurysms=MA，hemorrhages/haemorrhages=HE，hard exudates=EX，soft exudates/cotton wool spots=SE。
8. metric — `AUPR` / `Dice` / `F1` / `IoU` / `AUC` / `Sensitivity` 等
9. value — 数值，原文的小数（百分数写成百分数，如 67.55；小数写小数，如 0.6755；不要换算）
10. provenance_stated — 该行数字在原文里的来源标注：`rerun`（本文自己跑的：原文说 re-implemented / retrained / we trained，或表里有 * 号之类表示自己跑）/ `cited`（表里带引用、或文字说引自原文）/ `not stated` / `self`（is_proposed=yes 的行）
11. provenance_ref — 若 cited，引用的是哪篇（写第一作者年份或表里的引用标记，如 `[12]` 并在 notes 写出对应参考文献的第一作者年份）；否则 NA
12. notes — 如“表里以 % 表示”“三类平均”“骨干 ResNet-101”

## 规则
- 数字必须逐个从表格文本中读出；对不上的、读不清的写 `NA` 并在 notes 说明，不要猜。
- 一个表里一个方法在一个数据集上的四类加平均 = 最多 5 行（每种指标）。
- 平均列若原文没有而是你算的，不要写。
- 文本里的百分号 / 小数混用时，value 照原文；在 audit 的 notes 里写“AUPR 以 % 表示”。
- 若论文只在一个表里同时放 IDRiD 和 DDR，按 dataset 拆行。
- 参考文献编号对应：从文末 References 找到被引条目，provenance_ref 写 `第一作者年份`（如 `Guo2019`），找不到写引用标记。
- 完成后用 python 检查两份 TSV 每行列数一致（audit 18 列，results 12 列）。
