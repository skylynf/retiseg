# retiseg 检查点（2026-11-10 batch01 冻结，见 docs/clinical/34 第七节；2026-11-07 临床合作定位核对，见 docs/clinical/34；2026-11-06 发送前核对，见 docs/clinical/33；2026-11-03 按试判修订，见 docs/clinical/32；2026-11-01：论文定稿版——题名、作者、预印本排版；摘要精简、正文去模型评价与过程披露、补充材料同步；服务器实验全部完成；文档整理，过时文件进 docs/archive/。剩下的是用户侧：许可核对、作者联系数、临床试判）

给下一次会话或其他 agent 的最小上下文。细节都在仓库里，按下面的路径读；本文件只说"在哪、什么状态、怎么接着做"。

## 项目是什么
DR 眼底病灶分割（MA/HE/EX/SE）的 assessment 论文，体例照 Lu 2026 Nature Methods。仓库 github.com/skylynf/retiseg（2026-10-19 起公开；用户本机 `~/research/retiseg`，服务器 `/home/qurr/luming/retiseg/retiseg`，8×A100，用 GPU 2–7）。三个问题：Q1 复现（作者权重、作者配方）、Q2 统一比较（B-std 主表、独立测试、基础模型四档）、Q3 决定因素（训练来源、标注粒度、分辨率、训练量、损失）。2026-10-18 起另有 `docs/paper/26`：高层评估、补充实验（HRDecoder 案例、集成、人类参照、临床端点等）、临床医生方案、投稿策略。

## 先读哪些文件（按顺序）
1. `docs/ops/00_进度.txt` —— 当前状态、决定日志、待用户决定的清单、最近的服务器记录（旧记录在 `docs/ops/archive/`）。
2. `docs/paper/26_高层评估与补充实验.txt` —— 增补实验 N1–N8 的设计与预登记规则（都已完成）；`docs/ops/17_服务器任务单.txt` 第三版 —— 协作规则、回传格式、当前待办（只剩临床合作第 72 条）；`docs/paper/16_范围精简.txt` —— 实验范围与砍项（历史）。
3. `docs/paper/13` 第三节状态表（历史）；`docs/paper/14` 论文里要用的数字和方法（M1–M17、备注 1–78；备注 78 是定稿改动）；`docs/paper/27` 用户侧自查与待办。
4. E5 推理档：`docs/ops/25_albert合并记录.txt`、`docs/design/23`、`24`（D1–D9 的依据在 24 第十七节）。
5. 临床合作（2026-10-28 起）：`docs/clinical/28_临床合作总计划.txt`（总纲）、`docs/clinical/29_protocol_en.md`（英文方案）、`docs/clinical/30_reader_guide_en.md`（阅读指南）、`docs/clinical/31_会议纪要_20261029.txt`（与 Peranut 的讨论与落实）、`docs/clinical/32_试判结果与正式批次设计_20261103.txt`（试判结果、问卷 3.0.0、分层的正式批次）、`docs/clinical/33_发送前核对_20261106.txt`（一次抽样与阅读顺序、临床表述、盲法、20 + 80）、`docs/clinical/34_定位核对与正式批次_20261107.txt`（定位改正、配额规则、服务器下一步）；代码与问卷在仓库根目录 `clinical/`（`clinical/README.md`）。
6. 论文源：`docs/paper/retiseg_paper/paper/`（`main.tex` 预印本排版与作者信息；`sections/`；`STRUCTURE.md` 图表清单；`README.md` 流水线与改动记录；`supplement.tex`，`make supplement`）。过时的规划文件在 `docs/archive/`（03、05、09、11、18、19、10-09 的预览 PDF）。

## 现状（2026-11-10；仓库 origin/main cd290fb 加 10-30、11-01、11-03、11-06、11-07、11-10 写回）
- 主表（14 M7、M9）：19 行全部实数；MedSAM 是修订配方（整批 Dice；0.642 / 0.413，备注 55），塌缩 run 以 _registered 留在补充表。SAMed、RETFound 三种子齐：DDR 列 SAMed 0.460 ± 0.027、RETFound 0.450 ± 0.018、HRNet 0.447 ± 0.009——种子 0 的"基础模型领先"在三种子下落在散布内，R4 与讨论已按此写。`data/results.csv` 1,698 份报告、77,485 行实测；占位只剩 L2 的作者联系两行与 nnU-Net 的参数量。
- HRDecoder 案例 N1 收口（14 备注 46）：作者配方 0.674、两倍预算 0.615 / 0.304、作者画布（2880，等效 epoch）0.642（按验证选择重评后；最后一步是 0.627，备注 71）、1440 主表 0.647——画布与训练量都不是解释，作者配方多出的 0.03–0.05 归于协议替换的增强（±45° 旋转、无光度扰动）、54 张全训无验证划分、报最后一步。主表行不换（预登记）。
- 独立测试（14 M10）：19 模型 × 两源 × 八集合齐；HRDecoder DDR 已在正确归一化下重测（备注 57），DDR 源 all / fam:fundus 集成重算。名次统计（19 模型）：χ² 103.5、临界差 9.90、SAMed 1.38、HRNet 3.63、RETFound 5.00，前 12 不可分；正文全读 `\rn` 键。
- 编码器对照（14 M9、备注 56；collect 读 runs/E5enc_*）：C1、C2 都收齐——MAE-IN1k + UPerNet 0.519（DDR）/ 0.634（TJDR_std）高于 RETFound-MAE 0.466 / 0.589；DINOv2 0.510 / 0.619 高于 RETFound-DINOv2 0.490 / 0.616（C2 −0.020，独立三集合均值 +0.006）。结论：同头同配方同预算、各一个种子下，视网膜预训练在两对里都没有加分；三个非 RETFound-MAE 的编码器高于主表 DDR 列全部行。R5、讨论 Q2、方法已按实数写；图 5d 四个编码器全实测。IDRiD 列两对齐全：MAE-IN1k 0.729、DINOv2 0.739（高于主表 IDRiD 全部行），RETFound-MAE 0.708（主表行），RETFound-DINOv2 0.711；C1 / C2 在 IDRiD −0.021 / −0.028，与 DDR 同号。
- Refined IDRiD（备注 54）、人类参照（M16a）、分层（M17a）：同 10-22；MedSAM 修订 run 已补评。
- 推理档（14 M9）：测试阶段收齐，R5 全部实数——零样本任意病灶 ≤0.08（nnU-Net 同指标 0.78），SAM 3 选定措辞 0.40 / 0.18，理想框 SAM 0.86 / 0.77 对同信息参照 nnU-Net 0.86 / 0.71（框内勾画不输监督模型，差在找病灶），理想点 0.12 / 0.02。集成（14 M15）：18 人重算完成（DDR 源 all 0.677 / 0.637 / 0.266 / 0.335 / 0.496，R4 已换）；FGADR / Retinal-Lesions 不含 HACDR 的六个组合两源出数（粗标注集合上集成只比最好单模型高 0.01–0.02，细档 0.02–0.06，R4 与讨论分两档写）；all / fam:fundus 在这两个集合按 --without HACDR-Net 出数（DDR 源 all 0.291 / 0.278，与最好单模型持平；表 5 角标 p，R4 写明）；HACDR 的 CUBLAS 报错在探针后只剩 seed0 检查点一种解释（21 第 87 条），最后一步探针低优先（17 第 59 条）。
- 论文（11-01 定稿版；14 备注 78）：题名 Benchmarking deep learning methods for diabetic retinopathy lesion segmentation；作者 Luming Yang（通讯 yang.7670@osu.edu）、Siqing Li、Yuanchen Liu、Chang Yu、Haoxian Liu（HKUST）、Zhenyu Bu、Alper Yilmaz（其余 OSU）；预印本排版（页眉、题块、带底色的摘要框、符号脚注）。摘要精简到约 230 词、口吻客观，末句脚注说明预印本与仓库将公开；引言首句脚注说明病灶术语正与临床医生核对。正文：去掉对具体模型的评价式措辞（R2 审计发现、R3 作者代码、R4 内部测试按机制而非模型改写、R5 压缩）、删除与论文目标无关的过程披露（来源配方轨、砍掉的重训、登记日期、“after test numbers had been read”类表述、ED 状态表）；因素格更正的披露从补充说明 2 删去（内部更正，未发表过）；全文无破折号式插入；计算资源写明 8×A100 40 GB、约 2,000 GPU 时。ED 图重编为 1–4（学习率、损失、DiaRetDB1 / 超广角、算力）。补充说明 1 的登记备注经清洗（去日期、网盘 ID、邮箱、内部文件引用）。main.pdf 32 页、supplement.pdf 34 页；numbers.tex 886 键。10-30 的内容（因素格重评、RetiSeg-Ref 八个集合、算力审计、mAUPR 区间）见 14 备注 58、72、74、75。
- 下一步：服务器——只剩 17 第三节的临床合作第 72 条（（a）先重打两格源内图，再 `ensemble.py reference … --regenerate-maps` 留图；（b）正式批次：定位改正已合并、重挖与 dry-run 做完（11-10），batch01 11-10 冻结；下一步 b5：抽一次 100 张、打第一部分 20 张的包，见 17 第 72 条 b5 与 docs/clinical/34 第七节）。本机——论文侧等用户四项：DDR / TJDR 派生模型许可与 DiaRetDB1 概率图公开的核对（方法里只写“按三个来源的条款”）、10-31 后作者联系数（图 2a 不画，补充说明不提）、refs.bib 账本条目作者补全、投稿前通读；ED1 / ED2 示例图不再列入（需要时在工作站画）。
- 医生 / 临床合作（docs/clinical/28，11-03 按 32 修订，11-10 按 34 更新）：Dr. Peranut Chotcomwongse 11-02 读完试判（演示包，IDRiD 4 张图、18 个区域，有效 37 分钟）；区域几乎都离黄斑远（0 个在中心子区、1 个在 1 DD 内），16 个错误全部"不改变 ICDR、不改变处理"。问卷 3.0.0（`clinical/schema/review_schema.json`）：删 AI 分级与转诊题；DME 按 ETDRS 网格（中心子区 1 mm、内环即 1 DD、外环）判，工具 1.0.0 在医生的黄斑点上画网格；标签与 AI 分开判（两者都错由两个答案组合）；IRMA / NV 可标点。正式批次 `clinical/designs/batch01.yaml`（11-10 冻结）：图按 DDR 3–4 级 → 黄斑 1 DD 内有 HEX 分歧 → DDR 0–1 级 → 其余分层，图内按中心子区 HEX、内环 HEX、可疑 IRMA / NV、黄斑附近红色病灶、其他分歧、对照各抽 1 个，设计权重估计全体比例；黄斑与视盘先用 `clinical/scripts/locate_landmarks.py` 估计并在 QC 页改。分析 `analyze_reviews.py` 3.0（两轮廓归属、后果 0–3 级、µm 与 ETDRS 区带、DME 对照、IRMA / NV、加权比例）。11-06 发送前核对（docs/clinical/33）：重叠重抽改变入样概率，分层臂改为每层只抽一次，工具 1.1.0 改为同一张图所有 A 答完再一次揭示（分级一步在轮廓出现后锁定）；问卷 3.1.0（HEX 区带只是位置，门槛与 HEX 区带分开，图级处理题）；黄斑—视盘距离 4.7 mm；batch01 改为 100 张、先读 20 张（第一部分）再读 80 张，分析按读者在各层的完成数调整权重；29、30 v2.1 写清盲法与临床定义出处。11-07 定位核对（docs/clinical/34）：服务器自动定位在 DDR 上 88 / 225 张黄斑偏离超过 1 DD，本机逐张看过五个集合 425 张、改正 116 张（clinical/landmarks/batch01/）；配额差额由 sample_review.py 的 shortfall_to 加到 ddr_rest；batch01.yaml 带 status，改成 frozen 才能抽样。11-10：L1（IDRiD CC BY 4.0、DDR CC BY-NC-SA 4.0、TJDR 公开下载，包里写明出处与许可）、D4（MAPLES-DR 不进，MESSIDOR 条款禁止再分发）、D8（照提议稿）已定，batch01 冻结（DDR 70、IDRiD 10、TJDR_std 11、TJDR_uwf 9），每个集合记下挖掘输出的哈希（sets[].inputs，对不上不抽）；build_pack.py 在 READ_ME_FIRST.txt 写明出处与许可，遇到 MAPLES-DR、FGADR、Retinal-Lesions、DiaRetDB1 拒绝打包；29 升 v2.2。D3 读者人数不影响抽样。
- 用户待办：提交推送本机 11-10 写回（batch01 冻结）；第一部分的包回来后先走一遍再发给医生（附 29 v2.2、30）；把 29、30 v2.1 发给医生并确认第一部分的时间；L1 许可核对；投稿前核对作者单位与顺序。

## 数据流（结果 → 论文）
服务器 `python bench/scripts/pack_handoff.py` 打包 `runs/` 小文件 → 本机解到仓库 `runs/`（包顶层还有 docs/ops/00 的服务器版本，不要整包覆盖到仓库根）→ `python docs/paper/retiseg_paper/paper/scripts/collect_runs.py`（认 BS/B1/*_registered/NN_*/E1r/E2c/BF_*/E1_<slug>/runs/E3a/runs/E5；排除名单不对齐的报告拒收；BSprobe_ 不读）→ `make -C docs/paper/retiseg_paper/paper`。`collect_literature.py` 写 L1 行。整表重生成只在有 `runs/` 的机器上：`build_placeholder.py --force` → `collect_literature.py` → `collect_runs.py` → make。

## 规则（不要违反）
协作规则在 17 第一节：服务器只执行 HEAD 里的 17（队列与启动器已停用，剩余步骤是单条命令）；排除名单不对齐的指标不进 14、不打包、collect 拒收；服务器只追加 00 第三节，14/13/CHECKPOINT/论文由本机改；回传用 `bench/scripts/pack_handoff.py`。看过测试指标后不改预算、增强、选择规则；`bench/configs/*_frozen.yaml` 只增不改（e5_frozen.yaml 在读到首个 E5 测试指标前允许改写，之后只增）。集成的组合规则、读者研究的名单与端点规则都是预登记（14 M15–M17、备注 47–48），登记后不改。每个论文相关决定记 14，每次改动记 00；新文档从 32 起编（28–31 是 docs/clinical/）。

## 自动测试
GitHub Actions（`.github/workflows/tests.yml`）：每次 push/PR 用 CPU torch 跑 `bench/tests`（去掉 H2Former 前向，数据与权重相关的用例自动跳过）。红了先看是不是缺跳过标记或测试读了机器状态（21 第 52 条），记 21，不放松断言。协议 sha256 锁文件在读出首个 E5 测试指标后加（21 第 49 条）。

## 本机环境限制
云端工作区没有 torch；涉及 torch 的测试要在用户机器或服务器跑；不依赖 torch 的队列与文档测试在云端可跑（2026-10-19 核过 test_launch_queue 的 11 个）。云端能 clone 公开仓库、能装 pandas/matplotlib/TeX Live，所以 collect 与 make 可以在云端做，结果写回本机。
两台本地机器：docs 里的"本机"是 Luming 的工作站（4060 Ti，有数据、torch 2.6）；"工作站albert"是 Albert 的机器（RTX 5090，无数据和权重，torch 2.11，另有 retiseg-sam3 环境），只做代码和单元测试。
