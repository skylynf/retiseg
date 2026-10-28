# retiseg 检查点（2026-10-28：参考模型两格训完、M2MRF 作者配方三种子进表；发现因素格 20 格按最后一步打分，服务器按 17 第 68 条重评；RetiSeg-Ref 的发布入口 bench.reference 到位；mAUPR 区间脚本到位）

给下一次会话或其他 agent 的最小上下文。细节都在仓库里，按下面的路径读；本文件只说"在哪、什么状态、怎么接着做"。

## 项目是什么
DR 眼底病灶分割（MA/HE/EX/SE）的 assessment 论文，体例照 Lu 2026 Nature Methods。仓库 github.com/skylynf/retiseg（2026-10-19 起公开；用户本机 `~/research/retiseg`，服务器 `/home/qurr/luming/retiseg/retiseg`，8×A100，用 GPU 2–7）。三个问题：Q1 复现（作者权重、作者配方）、Q2 统一比较（B-std 主表、独立测试、基础模型四档）、Q3 决定因素（训练来源、标注粒度、分辨率、训练量、损失）。2026-10-18 起另有 `docs/paper/26`：高层评估、补充实验（HRDecoder 案例、集成、人类参照、临床端点等）、临床医生方案、投稿策略。

## 先读哪些文件（按顺序）
1. `docs/ops/00_进度.txt` —— 当前状态、决定日志、待用户决定的清单、最近的服务器记录（旧记录在 `docs/ops/archive/`）。
2. `docs/paper/26_高层评估与补充实验.txt` —— 当前计划的增补部分（风险、新实验 N1–N8 与预登记规则、医生 T1–T3、排期与先砍顺序）；`docs/ops/17_服务器任务单.txt` —— 规则、回传格式、队列顺序、服务器还要做的事（第四节第 19–24 条是 26 的服务器部分）；`docs/paper/16_范围精简.txt` —— 实验范围（第十节是 26 的增补）；`docs/paper/19` 第六节 —— 优先级与写作里程碑。
3. `docs/paper/13` 第三节状态表（末尾"26 增补"块）；`docs/paper/14` 论文里要用的数字和方法（M1–M17、备注 1–61）。
4. E5 推理档：`docs/ops/25_albert合并记录.txt`、`docs/design/23`、`24`（D1–D9 的依据在 24 第十七节）。
5. 临床合作（2026-10-28 起）：`docs/clinical/28_临床合作总计划.txt`（总纲）、`docs/clinical/29_protocol_en.md`（给斯坦福的英文方案）、`docs/clinical/30_reader_guide_en.md`（阅读指南）；代码与问卷在仓库根目录 `clinical/`（`clinical/README.md`）。
6. 论文源：`docs/paper/retiseg_paper/paper/`（`STRUCTURE.md` 图表清单；`README.md` 哪些是实数；`supplement.tex` 补充材料，`make supplement`；`27_稿件自查_20261026.txt` 用户侧的自查记录与待办）。

## 现状（2026-10-28；仓库 origin/main 1347976 加本次写回）
- 主表（14 M7、M9）：19 行全部实数；MedSAM 是修订配方（整批 Dice；0.642 / 0.413，备注 55），塌缩 run 以 _registered 留在补充表。SAMed、RETFound 三种子齐：DDR 列 SAMed 0.460 ± 0.027、RETFound 0.450 ± 0.018、HRNet 0.447 ± 0.009——种子 0 的"基础模型领先"在三种子下落在散布内，R4 与讨论已按此写。`data/results.csv` 1,698 份报告、77,485 行实测；占位只剩 L2 的作者联系两行与 nnU-Net 的参数量。
- HRDecoder 案例 N1 收口（14 备注 46）：作者配方 0.674、两倍预算 0.615 / 0.304、作者画布（2880，等效 epoch）0.627（这是最后一步的分数；该格按验证选择重评后会变，备注 71）、1440 主表 0.647——画布与训练量都不是解释，作者配方多出的 0.03–0.05 归于协议替换的增强（±45° 旋转、无光度扰动）、54 张全训无验证划分、报最后一步。主表行不换（预登记）。
- 独立测试（14 M10）：19 模型 × 两源 × 八集合齐；HRDecoder DDR 已在正确归一化下重测（备注 57），DDR 源 all / fam:fundus 集成重算。名次统计（19 模型）：χ² 103.5、临界差 9.90、SAMed 1.38、HRNet 3.63、RETFound 5.00，前 12 不可分；正文全读 `\rn` 键。
- 编码器对照（14 M9、备注 56；collect 读 runs/E5enc_*）：C1、C2 都收齐——MAE-IN1k + UPerNet 0.519（DDR）/ 0.634（TJDR_std）高于 RETFound-MAE 0.466 / 0.589；DINOv2 0.510 / 0.619 高于 RETFound-DINOv2 0.490 / 0.616（C2 −0.020，独立三集合均值 +0.006）。结论：同头同配方同预算、各一个种子下，视网膜预训练在两对里都没有加分；三个非 RETFound-MAE 的编码器高于主表 DDR 列全部行。R5、讨论 Q2、方法已按实数写；图 5d 四个编码器全实测。IDRiD 列两对齐全：MAE-IN1k 0.729、DINOv2 0.739（高于主表 IDRiD 全部行），RETFound-MAE 0.708（主表行），RETFound-DINOv2 0.711；C1 / C2 在 IDRiD −0.021 / −0.028，与 DDR 同号。
- Refined IDRiD（备注 54）、人类参照（M16a）、分层（M17a）：同 10-22；MedSAM 修订 run 已补评。
- 推理档（14 M9）：测试阶段收齐，R5 全部实数——零样本任意病灶 ≤0.08（nnU-Net 同指标 0.78），SAM 3 选定措辞 0.40 / 0.18，理想框 SAM 0.86 / 0.77 对同信息参照 nnU-Net 0.86 / 0.71（框内勾画不输监督模型，差在找病灶），理想点 0.12 / 0.02。集成（14 M15）：18 人重算完成（DDR 源 all 0.677 / 0.637 / 0.266 / 0.335 / 0.496，R4 已换）；FGADR / Retinal-Lesions 不含 HACDR 的六个组合两源出数（粗标注集合上集成只比最好单模型高 0.01–0.02，细档 0.02–0.06，R4 与讨论分两档写）；all / fam:fundus 在这两个集合按 --without HACDR-Net 出数（DDR 源 all 0.291 / 0.278，与最好单模型持平；表 5 角标 p，R4 写明）；HACDR 的 CUBLAS 报错在探针后只剩 seed0 检查点一种解释（21 第 87 条），最后一步探针低优先（17 第 59 条）。
- 论文（10-28）：10-26 用户侧的稿件自查（27：代号→名称、置信区间、\retifinaltrue、DiaRetDB1 列修正、nnU-Net 参数量实数）已并入主线。本轮：R3 作者配方重训按三种子写（作者函数 0.605–0.627 / 本协议 0.540–0.566，对发布权重低 5–7 分），表 3b 带 s.d.，摘要 / 讨论同步；R4 末尾参考模型一句按两成员 DDR 数写（SAMed 0.477、RETFound 0.451 对最佳单模型 0.469）并指向发布入口，方法 Reference model 段加入口一句；方法统计段 / 补充说明 3 写明 mAUPR 区间算法；补充说明 2 披露因素格打分更正。numbers.tex 556 键 0 占位；main.pdf 33 页、supplement.pdf 34 页。**待下一包重算**：因素格 20 格此前按最后一步打分（21 第 97 条、14 备注 71），服务器重评后表 7、图 6 / 7、补充表 7、R4 HRDecoder 案例段（2880 格 0.627 一句）、R6 四源矩阵、正文写死的因素数都要再核；mAUPR 区间进表后改讨论局限“留待终版”一句、R4 加区间宽度。
- 下一步（17 第四节第 65、68–71 条）：服务器——先停启动器；第 68 条 `rescore_factor_roles.py plan / apply`（CPU）+ 13 格 E3a 重跑（GPU 3–6，约两小时）；第 69 条参考模型链：五个外部集合不带 --drop-maps（launch_e3a --only，GPU 2）、源内 IDRiD / TJDR_std、`ensemble.py reference`、`python -m bench.reference bundle --out runs/release/retiseg_ref`、三张 DDR 测试图冒烟（成员图应与 prob_test 逐字节相同）、删图；第 70 条 `maupr_ci.py`（CPU 一小时）；第 65 条算力审计（GPU 7）；第 71 条打包。本机——新包进表后：因素数全面核对（表 7、图 6 / 7、R4 案例段、R6）、补充表 11 参考块、讨论局限的区间一句、R4 区间宽度、ED10b / 补充表 8 的算力列；N6、N7（CPU）仍未做；HRDecoder L1 其余列待抽。用户——10-31 后给作者联系数；ED1 / ED2 在工作站画；DDR / TJDR 派生模型许可与 DiaRetDB1 概率图公开的核对。
- 医生 / 临床合作（docs/clinical/28）：斯坦福合作医生要求先给完整计划、先做约 100 个标注的试点。方案：医生审阅 RetiSeg-Ref 与数据集标注的分歧区域（盲读 A → 揭示 → 归因、临床影响 0–3 与"决定性" → 整图小结），两阶段抽样（代表臂加权、富集臂），试点约 100 个区域 / 24 张图、每人约 2–2.5 小时；离线单文件阅读工具与挖掘 / 抽样 / 打包 / 分析脚本已写好并在 IDRiD 演示上跑通（`clinical/`，测试 `clinical/tests`）。取代 26 第四节 T1；T3 并入启动会。服务器：17 第四节第 72 条（a）在第 69 条删图前留图，（b）等通知。等 Luming：28 第十二节 D1–D6、第九节 L1–L3。
- 用户待办：提交推送本机写回（含新的 docs/clinical/ 与 clinical/）；临床合作的 L1–L3（许可核对、把 29 发给斯坦福并确认读者人数与时间、启动会）。

## 数据流（结果 → 论文）
服务器 `python bench/scripts/pack_handoff.py` 打包 `runs/` 小文件 → 本机解到仓库 `runs/`（包顶层还有 docs/ops/00 的服务器版本，不要整包覆盖到仓库根）→ `python docs/paper/retiseg_paper/paper/scripts/collect_runs.py`（认 BS/B1/*_registered/NN_*/E1r/E2c/BF_*/E1_<slug>/runs/E3a/runs/E5；排除名单不对齐的报告拒收；BSprobe_ 不读）→ `make -C docs/paper/retiseg_paper/paper`。`collect_literature.py` 写 L1 行。整表重生成只在有 `runs/` 的机器上：`build_placeholder.py --force` → `collect_literature.py` → `collect_runs.py` → make。

## 规则（不要违反）
协作规则在 17 第一节：服务器只执行 HEAD 里的 17；一台主机一个启动器、一份队列（`bench/configs/queue.yaml`，`launch_queue.py`，所有要用的卡写进 RETISEG_GPUS；两倍预算探测在拿出来的一张卡上跑）；排除名单不对齐的指标不进 14、不打包、collect 拒收；服务器只追加 00 第三节，14/13/CHECKPOINT/论文由本机改；回传用 `bench/scripts/pack_handoff.py`。看过测试指标后不改预算、增强、选择规则；`bench/configs/*_frozen.yaml` 只增不改（e5_frozen.yaml 在读到首个 E5 测试指标前允许改写，之后只增）。集成的组合规则、读者研究的名单与端点规则都是预登记（14 M15–M17、备注 47–48），登记后不改。每个论文相关决定记 14，每次改动记 00；新文档从 31 起编（28–30 是 docs/clinical/）。

## 自动测试
GitHub Actions（`.github/workflows/tests.yml`）：每次 push/PR 用 CPU torch 跑 `bench/tests`（去掉 H2Former 前向，数据与权重相关的用例自动跳过）。红了先看是不是缺跳过标记或测试读了机器状态（21 第 52 条），记 21，不放松断言。协议 sha256 锁文件在读出首个 E5 测试指标后加（21 第 49 条）。

## 本机环境限制
云端工作区没有 torch；涉及 torch 的测试要在用户机器或服务器跑；不依赖 torch 的队列与文档测试在云端可跑（2026-10-19 核过 test_launch_queue 的 11 个）。云端能 clone 公开仓库、能装 pandas/matplotlib/TeX Live，所以 collect 与 make 可以在云端做，结果写回本机。
两台本地机器：docs 里的"本机"是 Luming 的工作站（4060 Ti，有数据、torch 2.6）；"工作站albert"是 Albert 的机器（RTX 5090，无数据和权重，torch 2.11，另有 retiseg-sam3 环境），只做代码和单元测试。
