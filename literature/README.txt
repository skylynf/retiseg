文献
本目录只放文献本身和文献流程的记录，是论文方法部分“数据收集”和“模型收集”（模板 Lu 2026 Methods 的 Data collection、Model collection）的依据。
项目的设计、执行和论文写作在 docs/，总索引是 docs/README.txt。

一、目录
scope_definition.txt     纳入范围的定义（四类病灶、主比较集的条件、粒度、主指标、检索冻结日）
LEDGER.tsv               文献总表，一篇一行。列：文献ID、归类、批次、主任务纳入、重训轨、PDF、记录、题名。PDF 与记录两列相对本目录
04_代码审计.txt          阶段三：作者代码的克隆、训练入口和权重
06_检索.txt              阶段二：固定检索式、计数、灵敏度、优先名单
08_全文筛选.txt          阶段二：优先名单的全文取舍
search/                  检索脚本 fixed_search.py 和原始记录 raw/（三个库的 jsonl、screened.tsv、priority_abstracts.txt）
papers/<类别>/           PDF
meta/<类别>/             每篇一份识别记录 txt，与 PDF 同名、同类别目录；没有全文的只有 txt

类别目录对应论文里的角色，也就是模板里数据表和模型表的分组：
  00_template                              2   模板论文与补充材料，只借体例
  01_reviews                               3   综述和先前基准（Playout 2024）
  02_datasets                             13   数据集论文；各库在论文里的角色见 docs/design/12 第零节
  03_dr_methods/cnn_multiscale            28   DR 专用方法，多尺度 CNN（18 篇有全文）
  03_dr_methods/relation_transformer      10   DR 专用方法，关系或 Transformer（6 篇有全文）
  03_dr_methods/semi_supervised            3   DR 专用方法，半监督（2 篇有全文）
  03_dr_methods/single_lesion              4   单病灶旁支，不与四类平均混排
  04_foundation_models/dr_adapted          3   在本病上做过实验的基础模型适配
  04_foundation_models/not_on_dr           7   原文没做本病、作为基线的基础模型
  05_general_baselines                    12   通用分割基线（含 FCT、H2Former）
  06_octa                                  1   DRAC 超广角 OCTA，独立任务
合计 86 条，71 篇有 PDF。目录由 LEDGER 的“归类”唯一决定：数据集四个标签都进 02_datasets，其余标签一一对应一个目录。
模型族的划分和每族的说明见 docs/design/03；进哪一层、哪条轨以 bench/configs/models.yaml 为准。

二、分阶段的文献流程
阶段一  收集。LEDGER 的“批次”记录每篇在哪一轮进来：
  模板    2 条   模板论文与补充材料
  batch1 40 条   第一轮滚雪球收集：数据集、主要专用方法、基础模型和通用骨干（含后来补齐全文的 Guo 2024）
  batch2 18 条   第二轮滚雪球补充（含后来补齐全文的 CLC-Net、TC-Net）；原有 19 条，重复的 FCT 已删
  batch3 26 条   2026-10-03 固定检索的优先名单，纳入方法 24 篇、数据集 2 个
阶段二  检索和筛选（06、08）。检索冻结日 2026-09-30。三个库识别 1389 条，过布尔门槛 948 条，去重 764 篇，进入全文队列 112 篇；优先名单 27 条 DOI 加 TJDR 线索，纳入 26、排除 3。滚雪球语料里检索没回到来的论文留在账本，不因此移出。
阶段三  代码审计和复现等级（04）。只记作者自己的代码，仓库作为子模块放在 official_code/。复现等级 R0–R4、评价轨 A/B/C 和分层登记在 bench/configs/models.yaml，作者联系的回复写进对应 meta。
阶段四  原文数字抽取（L1，未开始）。逐篇抽对比表写进 meta/reported_results.tsv，做法见 docs/design/05 第四节。

三、记录规则
阅读新文献时，复制任意一份 meta txt 的字段，不要另起格式。账本以 LEDGER.tsv 和 meta 下的 txt 为准，两者同步改。
txt 的 PDF 字段从仓库根目录写起。有全文时，PDF 文件名与 txt 同名，命名为“第一作者姓年份_简称”。
新条目按“归类”放进对应类别目录，“批次”写它进来的那一轮。
论文表格中的数字必须注明来源：原文汇报、作者权重或本项目重训。
归类只能使用这些标签：
模板
综述
数据集-细像素
数据集-粗或区域
数据集-检出级
数据集-分级依赖
主比较-CNN多尺度
主比较-关系或Transformer
主比较-半监督
主比较-基础模型适配
单病灶
基线-通用分割
基线-基础模型未做本病
OCTA独立任务
