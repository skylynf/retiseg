文档总索引
按模板论文的路径组织：Lu Y. 等，Assessment of computational methods in predicting TCR-epitope binding recognition，Nature Methods 23:248–259，2025-11-28 在线发表，doi:10.1038/s41592-025-02910-0。PDF 在 literature/papers/00_template/。
本文暂定题名：Assessment of computational methods for diabetic retinopathy lesion segmentation in fundus images。总纲是 paper/13。

一、目录
paper/    论文写作。13 总纲：三个问题、研究设计、实验清单与状态、排期、对照模板的文章结构和图表。14 方法素材与备注：M1–M14 可直接改写成正文的方法、补充材料来源、局限和审稿回复要交代的备注。
design/   研究设计。03 模型归类（五族）；05 实验计划（每个实验怎么做）；09 第一批任务（B1 来源配方的逐模型配方和偏差）；12 数据集（全部库的角色，独立测试来源的划分、标签映射和指标）。
ops/      执行记录。00 进度（只记已经发生的事和数字）；07 本地数据（磁盘上有什么）；11 服务器执行（命令顺序和记录规则）。
literature/（仓库根目录下）文献。04 代码审计、06 检索、08 全文筛选、LEDGER.tsv、papers/、meta/，流程见 literature/README.txt。
文件编号沿用建立时的顺序，跨目录不重复，文中“见 05”“12 第三节”之类的引用按编号找。新文件从 15 起编。
2026-10-07 整理时删除：01 研究设计（并入 13 第二节）、02 数据集归类（并入 12 第零节）、10 实验规划审查（已采纳，并入 05）。经过记在 00 文末。

二、模板路径与本文的对应
下表左列是模板的结构，中间是本文对应的部分和实验编号，右列是设计、已有数字和论文写法各在哪里。图号按 13 第五节。

引言
  模板：领域难点、已有基准的不足、本文的评估策略
  本文：13 第五节“引言”四条；已有工作的不足以 Playout 2024（literature/meta/01_reviews/）为主要对照
  素材：scope_definition.txt；literature/06 第三节（检索灵敏度）

结果 1  数据收集与研究设计（模板 Fig. 1，Data collection and study design）
  本文：R1，图 1。数据集总表与粒度档、模型收集与复现等级、四种评价、内部与独立测试
  设计：13 第二节；12 第零、二节；03；bench/configs/models.yaml
  文献：literature/LEDGER.tsv；literature/06、08（检索与筛选计数）；literature/04（代码审计）
  写法：14 的 M1（数据集）、M2（预处理）、M3（划分）、M4（模型收集与复现等级）；备注 2、11、19

结果 2  原始模型（模板 Fig. 2，Building test sets to evaluate the original models；Performance of original models）
  本文分两块：
  R2 文献里的数字，图 2。L1 原文数字抽取、L2 复现漏斗、评测实现带来的偏差
  R3 原始模型，图 3。E1 作者权重（官方测试与独立测试，去掉该模型训练用过的库），E1r 作者配方重训
  设计：05 第一节第 1、2、7 条，第四节 Q1
  数字：00 的 E0、E1、E1r 段（M2MRF：IDRiD 作者 mAUPR 67.55 对本项目 0.639，DDR 49.20 对 0.401）
  写法：14 的 M5、M6；备注 3、4

结果 3  统一重训，内部测试与独立测试（模板 Fig. 3，Performance of retrained models）
  本文：R4，图 4。E2 B-std 主表（3 种子）、nnU-Net、E2b B1 第二列、E2c 小规模调参、E3a 独立测试、内部与独立之差、排名
  设计：05 第四节 E2、E2b、E2c、E3a；09（B1）；12 第二、五节；bench/configs/bs_frozen.yaml、b1_frozen.yaml
  执行：11；数字和状态在 00（审查采纳、服务器第一批）
  写法：14 的 M7、M8、M10、M11、M12；备注 1、5、6、14、15、16、17、18
  本文另加 R5 基础模型，图 5。E5 零样本、理想提示、参数高效微调、编码器加解码头；设计在 05 的 E5，写法在 14 的 M9（待补）

结果 4  来源效应与交叉效应（模板 Fig. 4，Source effects of negative data；Cross effects of training data）
  本文：R6，图 6。标注风格审计、训练集 × 测试集矩阵（E2 与 E3b）、标注粒度（E4）
  设计：12 第五节；05 第四节 E3b、E4
  写法：14 的 M1 粒度档、M10；备注 7、8、9、10、12、13

结果 5  规模效应（模板 Fig. 5，Performance of retrained models across sample sizes）
  本文：R7，图 7。训练量和多来源合并（E8），输入分辨率（E7），按病灶大小分层的召回
  设计：05 第四节 E7、E8；12 第五节 E8
  素材：bench/eval/input_label_audit.json（缩放后病灶消失），14 第二节

结果 6  正负比（模板 Fig. 6，P-to-N ratios）
  本文没有人工构造的正负比。对应的是病灶的极端不平衡：病灶大小分档（M11）和针对不平衡的损失（E9，放扩展数据 ED8）
  设计：05 第四节 E9；bench/eval/lesion_size_freeze.json

计算代价（模板 Extended Data Fig. 10，Time and memory usage）
  本文：R8，扩展数据 ED10。E10 参数量、FLOPs、显存、训练和推理时间
  写法：14 的 M13

讨论
  本文：13 第五节“讨论”：主要发现对应 Q1–Q3，给开发者和使用者的建议清单，局限
  素材：14 第四节备注（局限和审稿回复）

方法（模板 Methods 各小节 → 14 的小节）
  Workflow of model evaluation                      13 第二节
  Data collection                                   M1
  Model collection                                  M4；literature/README.txt 的分阶段流程
  Preprocessing                                     M2
  Generation of negative data                       本文对应阴性图像和缺失标注的定义、忽略区：M2、M11
  Construction of consensus test sets（原始模型）    M5（独立测试去掉该模型训练用过的库）
  Construction of training, test and independent sets   M3、M10（泄漏审计对应模板的 CD-HIT）
  各影响因素的评估                                    05 第四节、12 第五节；14 待补
  Time and resource consumption                     M13
  Model preparation and tuning                      M7、M8；E2c
  Metrics                                           M11、M12
  Data availability、Code availability              M14

补充材料
  扩展数据图 ED1–ED10、补充表 ST1–ST7、每个模型一节的补充说明：13 第五节
  各表的数据来源：14 第二节

三、写在哪里
发生了什么、看了哪些数：00，一段一个日期，写清提交号。
论文里怎么写：14，与做出决定的那次提交一起改。
总纲、实验状态、排期：13 第三、四节。
每个实验的做法：05；数据集的用法：12；B1 的逐模型配方：09。
磁盘状态：07；服务器操作：11。
文献：literature/，规则见 literature/README.txt。
文件之间冲突时以较新日期的决定为准，并在 00 记一笔。读任何测试集之前，协议和划分先提交；看过测试指标以后不改预算、增强和选择规则。
