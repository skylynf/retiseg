# paper/

论文正文、表和图的全部源文件。结构和图表清单见 `STRUCTURE.md`；这里只说怎么用。

## 一条命令

```bash
make -C paper            # 图、表、正文数字、参考文献、main.pdf
make -C paper collect    # 从 ../runs 读入已评测的结果，覆盖对应的占位行，然后重建
make -C paper final      # 没有占位行以后用：去掉所有“模拟”标记
make -C paper test       # 用真实的 bench.eval.evaluate 端到端检查 collect_runs.py
```

依赖：Python（numpy、pandas、scipy、matplotlib、pyyaml、pillow）和 TeX Live（pdflatex、latexmk）。统计量直接调用 `bench/eval/stats.py`，与基准用的是同一份代码。

## 目录

```
main.tex                    导言区和章节装配
sections/                   摘要与引言、结果、讨论、方法、扩展数据
tex/retiseg-tables.tex      表格宏（\ph \best \second \sd \up \down \famchip \tier \rlevel …）
tables/generated/*.tex      由 scripts/make_tables.py 生成，不要手改
generated/numbers.tex       由 scripts/make_numbers.py 生成，正文用 \rn{键} 引用
refs.bib                    手工维护（2026-11-01 起）：45 条被引文献逐条按 Crossref / 出版页 / arXiv 核对，Nature 体例（naturestyle.bst）
naturestyle.bst             Nature 体例的编号文献格式（五位以上作者写“第一作者 et al.”，题名链到 DOI）
refs_ledger.bib             scripts/make_bib.py 从 literature/meta 导出的账本（`make ledger-bib`），只作新增条目的起点
data/results.csv            唯一的数据表（长表），每行带 source = measured | placeholder
figures/retifig/            style.py（版式与配色）、data.py（读数与聚合）、panels.py（可复用面板）
figures/fig*.py, ed_figs.py 每张图一个脚本，输出到 figures/out/*.pdf 和 *.png
scripts/build_placeholder.py  生成占位数据（带固定种子）；已有 results.csv 时拒绝覆盖
scripts/collect_runs.py       读 runs/*/metrics_*.json，按键覆盖占位行
```

## 哪些是真数，哪些是模拟

2026-10-12 起（服务器快照接入后）真数还包括：B-std 7 个模型全部已打分种子和 nnU-Net 两种配置的内部测试（E2 的 IDRiD、DDR 列，含 11 阈值、Dice、病灶级召回）；U-Net 旧配方的六个 `registered` 运行；E1r 有效批 4 的 M2MRF；E2c 的 ×1 参照；E10 的参数量和训练时（environment.json）；L1 文献审计的全部行（图 2b、c，图 4f 的文献值）。一个（实验, 模型, 变体, 训练源, 测试集）组一旦有实测种子，占位种子就被 `collect_runs.py` 删掉，均值只在实测种子上算。2026-10-13 服务器又跑了一次 collect_runs.py（115 份报告写入 data/results.csv，未跑 make）：补上 HACDR-Net 与 U-Net 整批 Dice 的种子 1、2，E2c 四项验证，nnU-Net DDR 默认第二次，当时已完成的 E3a 主变体，以及四项 M2MRF 影响因素的内部测试。E5、E7、E8、E9 和未打分的 E3b、E4 仍是占位。图还是 10-12 的 make 结果，工作站 make 之后才跟这张表一致。

真数（黑色，图上无标记）：数据集规模和划分；models.yaml 的 38 个变体、分层和复现等级；检索与筛选计数；M2MRF 作者权重的全部数字（67.55/67.24/0.639、49.20/48.94/0.401，以及 16 位存储后的 0.6446、0.426）；E1r 第一次重训（57.63、0.493）；病灶大小分位数和分档边界；缩放后病灶消失比例；U-Net 和 M2MRF 的参数量；6 个 B-std 模型在 IDRiD 上种子 0 的 GPU 时（取自 00）；方法表里各模型保留的设定。

模拟（蓝灰色数字、图上“simulated”角标、正文左侧竖线）：其余全部，包括 B-std 主表、独立测试、基础模型、影响因素、文献审计 L1、计算代价的大部分。生成模型只保证形状和量级合理，**不代表任何预期结论**。

`collect_runs.py` 读入一份真实报告后，该报告覆盖的行变成 measured；一个面板只要还用到任何占位行，角标就保留。2026-10-20 起：名次统计（Friedman、临界差、配对检验、族均值、发表名次 τ）只对实测模型算（`retifig.data.measured_models`），未实测的模型（MedSAM）排在图 4/7 行序末尾并单独标记；图 6b–d、7d–e 与 make_numbers 的因素键只用实测格，缺格画 "–" / 标 pending，不再把占位值平均进去。2026-10-22 起：表 6 每格的占位标记按它读到的行是否实测决定（以前整表硬标占位），单种子行不再打 ±nan；`runs/REFINED/…`（Refined IDRiD 重评，experiment REF，含 `E1_<模型>` 的发布权重）也由 `collect_runs.py` 读入，make_numbers 给出 ref.*、cd.second/third、indep.*.nlower/gain 键。 2026-10-23 起：`runs/E5enc_<编码器>_ddr_seed0_lr1/`（编码器对照，E5 / encoder_control，RETFound-MAE 行从主表复制）也读入；占位里的对照名 ImageNet-sup. 改为 MAE-IN1k；图 5d 只画已实测的编码器。2026-10-24 起：`runs/audit/label_style.json`（bench/scripts/label_style_audit.py 的八源标注风格审计）由 `collect_runs.py` 读成 audit 行，图 6a 八个来源全为实测，其 simulated 角标只在仍有占位来源时出现；集成行的 `ensemble:without …` 变体（成员缺图时的 --without）在表 5 带角标 p。2026-10-25 起：`runs/ENS/REF/reference/<集>`（参考模型两格的集成，done.json 带 paper 键 N8 / reference:ensemble）也由 `collect_runs.py` 读入；发布检查点的“论文数”由 `collect_runs.E1_REPORTED`（HRDecoder 论文表 1、HACDR-Net 的 L1 抽取）写成实测；RETFound-MAE 的编码器参照行按训练源各复制一份（IDRiD 列）；图 3 只画跑过的检查点（M2MRF、HRDecoder）与作者配方重训（砍掉的组标 not rerun），表 3 同；B1（来源配方）在任何运行完成前停止，不再有补充表。2026-10-26 起：论文不再打算测的占位行由 `collect_runs.drop_retired_placeholders` 删除（14 备注 65），图 2a 漏斗末级由作者配方重训推出，`runs/E3a` 的 `__fov` 在主变体缺失时顶替主行、否则作为 `<variant>:fov` 行并入（ED9）；补充材料是单独的 `supplement.tex`（`make supplement` 先跑 `scripts/make_supplement.py` 生成 `tables/generated/supp_*.tex` 与 `generated/supp_note_models.tex`，再 latexmk），17 张表、3 个说明都从 results.csv 与仓库事实生成。

2026-10-26（稿件自查，本轮）：`data/results.csv` 已无占位行（nnU-Net 两种配置的参数量由 `collect_runs.NNUNET_PARAMS` 按 console.log 记录的 plan 实例化计数写入，`params_m_plan` 另记两源各自的 plan；L2 的"作者联系 / 回复"两行无图表读取，已退役），`main.tex` 与 `supplement.tex` 改为 `\retifinaltrue`，`make_tables.write()` 只在表里确有 `\ph{}` 时才印占位说明。**修正了一处汇总错误**：`retifig.data.block_scores` 与 `make_tables.table_independent` / `ensemble_rows` 读 `img_auc` 时没有按 `test_set == DiaRetDB1` 过滤，表 5 的 DiaRet 列、Friedman 的 DiaRetDB1 区组和图 4e 实际平均了八个测试集的图像级 AUC；现在走 `retifig.data.diaretdb1_auc()`（χ² 由 103.5 变为 97.9，名次有变动）。正文的实验代号（E1、E1r、E2 / B-std、E3a、E3b、E4、E5、E7、E8、E9、E10、B1、L1、L2）全部改为实验名称，对照表是补充表 18（`make_supplement.EXPERIMENT_NAMES`）；仓库、runs/ 与 results.csv 的键不变。表 8 补齐 19 个模型（`table_recipes.frozen`，来自各 run 的 recipe.json 与 bs_frozen.yaml）；表 3a 每格下加 95% bootstrap 区间；表 1 DiaRetDB1 的概率图改为公开（与补充数据 4 一致，需核对许可）。名次检验、配对检验与表 5 都用种子均值（`retifig.data.score_matrix`），正文原来写"seed 0"，本轮按代码改正文。正文新增键：`clin.*`（按分级分层段）、`human.*`（人类参照段，原手写数 3/5 与表不符，实为 5/7）、`ci.*`（区间宽度与四个模型逐类区间）、`indep.<源>.<集>.mean/pct/first.*/rho`、`indep.pairs.*`、`params.*`。

2026-10-28：10-28 包进表（M2MRF 作者配方重训种子 1、2：作者函数 0.605 / 0.627、本协议 0.540 / 0.566；RETFound 参考格 DDR 内部 0.451）。新键 `e1r.M2MRF.IDRiD.{n,fn.lo,fn.hi,fn.sd,exact.lo,exact.hi,exact.sd}`（R3 一句按三种子写）、`n8.*`（参考模型两成员与集成；R4 末尾一句不再是待完成）、`mci.*`（mAUPR 的 bootstrap 区间，等服务器跑 `bench/scripts/maupr_ci.py` 后 `collect_runs.py` 读 `summary.mAUPR_ci` 成 cls=mean 的 `aupr_ci_lo/hi` 行）。**待修的数据**：因素格（BF_*）里 M2MRF / RETFound / SAMed / SAM2-UNet / HRDecoder 的 20 格到 10-28 为止都是按最后一步打分的（bench/train.py 的 roles_for 只认 BS；docs/ops/21 第 97 条），与方法写的验证选择不符；服务器按 `bench/scripts/rescore_factor_roles.py` 换成 best.pt 并重跑 13 格的 E3a 后，表 7、图 6 / 7、ED7 之外的因素数、R4 的 HRDecoder 案例段（2880 格 0.627）、R6 的四源矩阵都会变，下一包进表后按键重算；最后一步的分数留在 run 目录的 `*_laststep` 文件里。

2026-10-30：10-30 包进表（1,723 份报告；1,551 份带 `summary.mAUPR_ci`）。上一条“待修的数据”已修：因素格 20 格按验证选择重评、13 格 E3a 重跑后，R4 案例段、R6 两段、R7 两段、讨论 Q3 的数全部改为 `\rn` 键（`src.*`、`cross.max.train/test`、`e4.<模型>.*`、`e7.<模型>.*`、`e7.hrdecoder.*`、`e7.drop.*`、`e8.<模型>.*`），`fig6_sources.matrix()` 改为主表种子 0（与图注一致）。新进表：RetiSeg-Ref（`runs/ENS/REF/reference`，键 `n8.*`、`ensall.*`、`n8.vs.all.*`；R4 末段）、算力审计（`data/cost_audit.tsv`，键 `cost.{flops,infer,mem}.*`；ED10b 与补充表 8 读 `peak_infer_mem_gb`，列名“推理峰值显存”）、mAUPR 区间（cls=mean 的 `aupr_ci_lo/hi`，键 `mci.*`；补充表 4 末列）。results.csv 无占位，numbers.tex 886 键。

2026-11-01（定稿版，14 备注 78）：`main.tex` 改为预印本排版（题名、作者、单位、通讯邮箱、页眉、摘要框、符号脚注、去行号；`\retifinaltrue` 固定），`sections/abstract_intro.tex` 重写（摘要约 230 词，两处脚注），`results.tex` / `discussion.tex` / `methods.tex` 去模型评价式措辞与过程披露、压缩 R4 内部段与 R5，`extended.tex` 只剩四张 ED 图（编号 1–4，`\label{edfig:*}` 引用）。`fig3_original.py` 面板 c 只画有 run 的组；`fig1_design.py` 的 Tier 2 标签；`make_tables.py` 表 2 备注与表 7 分组行；`make_supplement.py` 的 `_clean_note`（补充说明 1 登记备注清洗）、补充表 18 去 E2b 行、ED 编号；`supplement.tex` 题块与补充说明 2 改写。

2026-11-05（第八轮，叙事重组）：
- 主线：已发表的性能排名不足以支持方法选择；评价实现、标注规范与成像条件影响比较，因此区分数值可重复性、受控条件下的性能、跨来源泛化三层。摘要、引言、结果、讨论都按这三层重写。
- 结果顺序：研究设计 → 已发表数字不可直接比较 → 评价实现改变分数与名次（含作者权重、作者配方）→ 受控条件下领先模型不可区分（含已发表排名与重训排名无关、配方组件导致的偏离）→ 基础模型取决于用法 → 跨来源时只有标注一致处排名才成立（独立测试、相关、Friedman、RETFound / EyePACS、分级）→ 标注规范决定哪些训练数据可迁移 → 成像条件限制小病灶检测（含训练量）→ 集成、参考模型与代价。每段先给结论句，长句拆短，正文不再逐项复述表格数值。
- 细节移入补充材料新加的补充说明 2（additional results）：度量实现细节、作者权重与作者配方、内部测试逐模型数值与区间、HRDecoder 案例、基础模型（零样本、SAM 3 措辞、oracle 提示、编码器数值）、独立测试逐集结果、训练来源与标注粒度细节、Refined IDRiD、病灶大小 / 分辨率 / 训练量、按 DR 分级、人类参照、集成与参考模型、计算代价。supplement.tex 用 xr-hyper 引正文的图表编号（\externaldocument[M-]{main}），所以先编 main.tex 再编 supplement.tex。
- 因图表位置调整：基础模型表变为表 5、独立测试表变为表 6（补充表 18 与 make_supplement.py 同步）；图 4 仍为重训结果、图 5 为基础模型。
- 引言：不再回顾 CNN、Transformer、半监督与基础模型的发展史（这些方法的引用移到结果的模型清单里），改为问题导向：三层性质 × 三类外部因素（评价实现、标注规范、成像条件）。
- 讨论：重写为概念总结（三层各一段 + 启示 + 局限与展望），去掉案例细节、配方细节与逐条限定。
- 去防御性写法：方法中“reported without interpretation”“only to explain”“is marked as such”“is not a ranked row”等改为直述；稀有类塌缩一段拆成短句。
- 字数（不含方法、文献、图注）：摘要约 195 词，引言约 645，结果约 2,770，讨论约 815，正文合计约 4,230（原约 9,500）；方法约 5,280；补充说明 2 约 5,430。正文 PDF 34 页（至参考文献 31 页）。

2026-11-01（第七轮）：页眉左侧改为 “RetiSeg: assessment of DR lesion segmentation methods”（补充材料后接 “· Supplementary Information”）；表 2 表注删去已过期的“作者回复截至 2026-10-31”，改为“Code status as of the audit of 2–4 October 2026”（make_tables.py 同步）。另导出正文至参考文献的单独 PDF（第 1–39 页，不含 Extended Data）。

2026-11-01（第六轮）：正文页眉左侧改为 “RetiSeg”（补充材料为 “RetiSeg · Supplementary Information”）；摘要脚注写入仓库地址并注明仍在完善中（“is still being refined”），概率图与参考模型权重将在该处发布；Code availability 仍写“将公开”。

2026-11-01（第五轮，引言压缩）：
- 引言由约 1,090 词压到约 720 词（五段）：第一段把疾病负担、筛查与分级、图像级 AI 及其局限合为一段；英格兰筛查与致盲认定、病灶感知系统的细节删去；部署与可解释性（Beede 2020、Ruamviboonsuk 2022、Sayres 2019）移到讨论局限的临床一句；第五段把五条研究线合成一句，T 细胞受体评估的引用只留在第四段。摘要脚注删去括号内容。
- 参考文献 83 → 69：删去 14 条只为铺背景、可由其他条目替代的文献（Yau 2012、Wong 2016、Scanlon 2017、Liew 2014、Ting 2019、Walter 2002、Abràmoff 2010、Long 2015、Litjens 2017、Moor 2023、Maier-Hein 2024、Varoquaux 2022、Quellec 2017、Wang 2017），refs.bib 同步只留被引条目；EyePACS（Cuadros 2009）改在方法 Retinal-Lesions 处引用，e-ophtha、ROC 在讨论的未来工作处再引一次。
- 长度对照（2026-11 查 nature.com）：Nat. Commun. Article 正文 ≤ 5,000 词（不含摘要、方法、文献、图注）、摘要 ≤ 200 词、≤ 10 个图表、文献 ≤ 70、方法通常 < 3,000 词；Nat. Methods Analysis 正文 3,000 词（编辑同意可到 5,000）、摘要 ≤ 150 词、≤ 6 个图表、文献建议 ≤ 50；npj Digit. Med. 不设硬性字数、摘要 ≤ 150 词、文献 ≤ 60。本稿：摘要约 209 词，引言约 720，结果约 7,090，讨论约 1,680（正文合计约 9,500），方法约 5,360，图 7 + 表 8，文献 69。

2026-11-01（第四轮，引言与背景文献）：
- 引言前三段重写：DR 的负担（Teo 2021：2020 年约 1.03 亿成人有 DR、2,850 万为威胁视力的 DR，2045 年预计 1.605 亿与 4,480 万）、筛查与分级（英格兰 2003 年起的照片筛查；2009–2010 年 DR 已不再是英格兰和威尔士工作年龄人群致盲认定的首位原因；ETDRS 与国际临床分级由病灶定义）、AI 筛查（Gulshan 2016、Ting 2017、IDx-DR 关键试验、泰国全国项目前瞻评估、七个系统真实世界敏感度 50.98–85.90%、部署与阅片者差异、显著图辅助效果不一、病灶感知系统）、病灶分割的发展（早期图像处理与 ROC 挑战、DIARETDB1 / e-ophtha、FCN / U-Net、像素级数据集、方法族、视网膜基础模型）。问题清单补上外部验证、指标陷阱、数据重叠的文献；引言末尾一句交代正与眼科医生复核模型错误、临床部分在后续版本报告；讨论局限补一句“临床后果未评估”；方法 Human reference 末句（临床复核进行中）删去以免重复。
- 结果与方法给 LoRA、ViT、UPerNet、MAE、DINOv2 补引用。
- 参考文献由 45 条增至 83 条；新增 38 条同样逐条按 Crossref / 出版页 / 会议页核对，全作者、Nature 体例。

2026-11-01（第三轮，按用户意见收回一部分版式改动）：
- 题块恢复原来的结构：左上 “Analysis” 标签（俄亥俄州立大学猩红色 #BA0C2F）、通栏猩红细线、题名与作者居左（无衬线体）；单位与通讯恢复原来的字体与字号（无衬线 8 pt、灰色），每个单位一行；右上留俄亥俄州立大学 logo 位（`\osulogo`，读 `figures/logo/osu_logo.pdf|png|jpg`，文件还没放进来，见该目录 README.txt）。摘要恢复为居左的底色框，框线由蓝色改为猩红色。小节标题、图注、页眉恢复为无衬线体（黑 / 灰）。正文仍为 11 pt Times、US Letter。
- 题名改为 *Assessment of computational methods for diabetic retinopathy lesion segmentation*。
- 代码与数据一律改为“将公开”：去掉题块中的 GitHub 地址；摘要末句改为 “We will release … RetiSeg†”，脚注说明代码库（数据导出、统一训练、推理、评测、划分名单、出图脚本）连同概率图与参考模型权重将公开；Code / Data availability、结果与方法中的 RetiSeg-Ref、讨论、补充数据、表 8 表注同步改为将来时；正文不再出现仓库链接。
- 去重复：结果的 Computational cost 与方法的 Time and resource consumption 分工——硬件、GPU 时、推理审计的口径只在方法，结果只报结果；结果首节的文献检索数字（1,389 / 764 / 112 …）只留在方法 Model collection；结果中参考模型的构造只留一句、细节在方法；HRDecoder 发布权重的说明在 R4 不再重复；R4 删去重复的表 8 指针；方法 Workflow 删去与引言、结果首节重复的 Q1–Q3 复述；方法 Granularity tiers 删去与结果重复的“只在档内平均”；讨论局限删去与数据可用性、表 1 重复的许可说明；补充说明 2、3 与方法逐段重复（预算、选择规则、配方修订、评测协议、超广角），整节删去，独有的一句（nnU-Net 用自己的调度与检查点选择）并入方法 nnU-Net 段；方法训练量一句的“Supplementary Note 2”指针删去；扩展数据清单同步。

2026-11-01（第二轮，arXiv 版）：
- 版式：`main.tex`、`supplement.tex` 改为黑白学术排版（US Letter；正文 11 pt、补充 10 pt；居中题块、标准 abstract 环境、页眉“Yang et al. / 题名”、页码；Nature 式图注“Fig. 1 | …”；文献编号“1.”、链接为黑色）。去掉副标题、“Analysis / Preprint”标签、彩色题名与摘要底色框。题名改为 *Assessment of deep learning methods for diabetic retinopathy lesion segmentation*。
- 作者信息：单位 1 改为 Photogrammetric Computer Vision Lab, The Ohio State University；单位 2 改为 Department of Computer Science and Engineering, HKUST；通讯作者加 Alper Yilmaz（yilmaz.15@osu.edu）。摘要脚注“仓库将随终版公开”删去（仓库已公开，与 Code availability 矛盾），题块下给仓库地址。
- 参考文献：`refs.bib` 改为手工维护的 45 条完整条目（全作者、全题名、期刊缩写、卷期页、DOI），`naturestyle.bst` 排成 Nature 体例；`make_bib.py` 改写到 `refs_ledger.bib`，Makefile 去掉 refs.bib 规则（`make ledger-bib`）。核对中发现：L-Seg 的 DOI 原为错号（改为 10.1016/j.neucom.2019.04.019）；M2MRF、PMCNet、RTNet、TP-DRSeg、HRDecoder、SAM 2（ICLR 2025）、SAM 3（ICLR 2026）、SAM-Adapter（ICCVW 2023）、SAM2-UNet（Vis. Intell. 2026）、Retinal-Lesions（ICPR 2020，2021 出版）改用正式发表版本。
- 小标题：讨论的“Reproducibility (Q1) / Comparison (Q2) / What decides performance (Q3)”改为 “Reproducibility of published results / Comparison under equal conditions / Determinants of performance”；方法的“What is shared”改为“Shared settings”。
- U-Net 的稀有类塌缩与损失修订：结果、方法（工作流末句与 Rare-class collapse）、补充说明 2 三处统一为同一说法——登记配方（逐图 Dice、平滑 1）在 DDR 上塌缩；修订（整批 Dice、平滑 1e-5）是在登记 run 已打分之后、依据训练 / 验证曲线与诊断 run 做的；SAM2-UNet、MedSAM 是训练前登记的规则。结果一句的数字统一为种子 0（0.119 → 0.378；IDRiD 0.658），三种子均值另括注；R7 损失一句“0.378（the main table）”改为“主表配方、种子 0”。方法的 Track B 定义补一句：U-Net 用参考 PyTorch 实现（作者发布的是 Caffe 包），与表 2、补充说明 1 的“reference implementation / no author repo”对上。
- 补充表 6：登记行原来取了三种子均值（U-Net IDRiD .594、DDR .114），与表题“seed 0”和修订行不一致；make_supplement.s6_registered 加 seed == 0，改为 .606 / .119（与正文一致）。
- RETFound 与 EyePACS：方法的测试场景段点名写明 RETFound 彩照预训练含全部 Kaggle EyePACS（88,702 / 904,170），Retinal-Lesions 与 FGADR 的 _3 批取自 EyePACS，故这两列对 RETFound 不完全独立；表 5 这四格加角标 e 与表注（make_tables.PRETRAIN_OVERLAP）；名次段加一句：去掉这两个区组后 RETFound 仍第四（5.67 对 5.50），前三不变（按 results.csv 复算，χ² 97.9 与表一致）；讨论局限同步。方法 FGADR 描述补 _3 批来源。
- 去重方法：方法原写“perceptual hashes and feature similarity … 以及基础模型预训练集与测试集之间”，与补充表 15（血管特征）不一致、且预训练集并未逐图检索；改为与补充表 15 同一种血管特征（192×192、5×5 黑顶帽、视野内 NCC、取镜像较大值；阈值 0.5，FGADR 内 0.33），预训练集按来源判定。补充表 15 表题同步补全参数。
- 其他数字：内部测试段“种子 0 两个基础模型对 HRNet 的领先 0.022 / 0.019”按 results.csv 改为 0.026 / 0.024（HRNet 种子 0 为 0.443），三种子 0.003 → 0.004；“前两名之差 0.012 / 0.016”过时，改为 IDRiD < 0.001、DDR \rn{bstd.DDR.gap12}；SAM-Adapter 在 IDRiD 与 SAMed 并列第一（0.706），原文写成“在 0.04 内跟随”，结果与讨论改写；CLC-Net 对 U-Net 在 IDRiD 是高 0.02，不是“持平”。表 2 表题“Track C 单列报告”与正文不符，改为“表中以上标 C 标出”。`\rn` 的负数前导连字符现排成减号。
- 补充表 18 里“in training”：现文件里已经没有（N8 一行为 “Reference model (RetiSeg-Ref)”），无需改。图 4f 图内的“τ = -0.11”是图片里的连字符，未重画。

## 服务器结果回来以后

1. 把服务器的快照包（docs/ops/17 第一节的格式）解开到仓库根目录的 `runs/`（gitignore）。
2. `make collect`（在本目录）。脚本读 `runs/BS_*`、`B1_*`、`*_registered`、`NN_*`、`E1r_*`、`E2c_*`、`BF_*`、`E1_*`、`runs/E3a/<来源>/<测试集>/metrics.json`、`runs/E5/…`、`runs/ENS/…`（集成）、`runs/HUMAN/DiaRetDB1/expert_*`（留一专家参照）、`runs/CLIN/strata/…`（按分级分层）、`runs/REFINED/<run>/{refined,original}/metrics.json`（Refined IDRiD），加 `data/cost_measured.tsv`、`data/cost_audit.tsv`，打印读到了哪些报告、替换和删掉了多少占位行。
3. 重读正文里带竖线的段落：数字和“谁第一”会自动更新，措辞不会。

独立测试的推理还没有接进 bench/，所以 `collect_runs.py` 里“独立测试报告叫 `metrics_<测试集>.json`”是这里的假设；接线定下文件名以后改 `TESTSET` 一处。nnU-Net、E1、E3b、E4–E9 的收集入口也要等各自的输出格式定了再加，目前只能手工往 `results.csv` 里按键写行。

## 正文按“已完成”写、目前还没实现的步骤

方法和结果按全部实验做完的口吻写。下面这些在 bench/ 里还没有，写作时对照 00 逐条确认：

- 独立测试推理的接线（E3a）和超广角原分辨率滑窗。
- 泄漏审计（方法里写了做法和“扩展数据图 3”）。
- DiaRetDB1 官方图像级 ROC/WER 与区域级命中；表 5 的 DiaRetDB1 列现在按图像级 AUC 写。
- 训练损失的忽略区（Retinal-Lesions 的 127），E4 之前要补。
- 病灶级的严格匹配辅指标（讨论的局限里提到）。
- nnU-Net、E5 四档、E1r 的 bs4 与 HACDR-Net、L1 抽取。
- 排名检验里独立区组的取法（STRUCTURE.md 第五节第 1 条）还没有写进 docs/ 或 protocol.yaml。

另外有两处是“披露”而不是“待办”：方法第一段写了协议冻结前做过链路评测、视野规则和分档边界改过一次；U-Net 的 B1 配方修订看过一次测试数（14 备注 1）。措辞请你过目。

## 提交前要核对的事实

这些不在 docs/ 里，是按原文常识写的，或者是这次为了排版先填的：

- 表 1 的视野：IDRiD 50°、DDR 45°；DDR 的标注方式写的是 pixel masks。
- 方法里“FGADR 许可禁止再分发派生数据；Retinal-Lesions 按申请条款同样处理”，后一半以申请条款原文为准。
- 文献流程里优先名单写成“27 个 DOI、TJDR 线索、另有一篇由更正 DOI 进入”，合计 29 条去向（纳入 26、排除 3）；06 第六节原文是“27 条 DOI 加 TJDR 线索”，两处请统一。
- 表 6 的 Backbone 列（SAM ViT-H、SAM 2 Hiera-L、MedSAM/SAMed ViT-B、RETFound ViT-L）：按 E5 实际配置改。
- `refs.bib`：2026-11-01 已全部补全并核对（见上面的改动记录）；新增引用时直接写进 refs.bib，按同一体例。
- 图 2c、表 3b 里 HACDR-Net、WSRFNet 的“论文报告值”是占位，L1 抽取后替换。
- 表 1 里 DiaRetDB1 的概率图 2026-10-26 改为"公开"（与补充数据 4、方法的数据可用性一致）；M14 仍未写，请确认许可。
- 参考文献：已完成（2026-11-01，第二轮）。

## 换版式

正文现在是单栏 US Letter、11 pt Times、1 英寸边距（补充材料 10 pt、0.75 英寸），黑白题块、无编号小节、Nature 式图注（Fig. 1 | …）。换成期刊模板时只动 `main.tex` 的导言区；表和图不依赖文档类。图的宽度按 180 mm 设计，字号 5–7 pt。
