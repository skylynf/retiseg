# retiseg 检查点（2026-10-24 第二次：v4 包进表——编码器 C2 收齐、18 人集成重算、FGADR / RL 六组合、八源标注审计；R4 / R5 / R6 / 讨论按实数改；HACDR 的 CUBLAS 判为环境问题）

给下一次会话或其他 agent 的最小上下文。细节都在仓库里，按下面的路径读；本文件只说"在哪、什么状态、怎么接着做"。

## 项目是什么
DR 眼底病灶分割（MA/HE/EX/SE）的 assessment 论文，体例照 Lu 2026 Nature Methods。仓库 github.com/skylynf/retiseg（2026-10-19 起公开；用户本机 `~/research/retiseg`，服务器 `/home/qurr/luming/retiseg/retiseg`，8×A100，用 GPU 2–7）。三个问题：Q1 复现（作者权重、作者配方）、Q2 统一比较（B-std 主表、独立测试、基础模型四档）、Q3 决定因素（训练来源、标注粒度、分辨率、训练量、损失）。2026-10-18 起另有 `docs/paper/26`：高层评估、补充实验（HRDecoder 案例、集成、人类参照、临床端点等）、临床医生方案、投稿策略。

## 先读哪些文件（按顺序）
1. `docs/ops/00_进度.txt` —— 当前状态、决定日志、待用户决定的清单、最近的服务器记录（旧记录在 `docs/ops/archive/`）。
2. `docs/paper/26_高层评估与补充实验.txt` —— 当前计划的增补部分（风险、新实验 N1–N8 与预登记规则、医生 T1–T3、排期与先砍顺序）；`docs/ops/17_服务器任务单.txt` —— 规则、回传格式、队列顺序、服务器还要做的事（第四节第 19–24 条是 26 的服务器部分）；`docs/paper/16_范围精简.txt` —— 实验范围（第十节是 26 的增补）；`docs/paper/19` 第六节 —— 优先级与写作里程碑。
3. `docs/paper/13` 第三节状态表（末尾"26 增补"块）；`docs/paper/14` 论文里要用的数字和方法（M1–M17、备注 1–61）。
4. E5 推理档：`docs/ops/25_albert合并记录.txt`、`docs/design/23`、`24`（D1–D9 的依据在 24 第十七节）。
5. 论文源：`docs/paper/retiseg_paper/paper/`（`STRUCTURE.md` 图表清单；`README.md` 哪些是实数）。

## 现状（2026-10-24 第二次；仓库 origin/main 80f5d73 加本次写回）
- 主表（14 M7、M9）：19 行全部实数；MedSAM 是修订配方（整批 Dice；0.642 / 0.413，备注 55），塌缩 run 以 _registered 留在补充表。SAMed、RETFound 三种子齐：DDR 列 SAMed 0.460 ± 0.027、RETFound 0.450 ± 0.018、HRNet 0.447 ± 0.009——种子 0 的"基础模型领先"在三种子下落在散布内，R4 与讨论已按此写。`data/results.csv` 1,653 份报告、68,722 行实测。
- HRDecoder 案例 N1 收口（14 备注 46）：作者配方 0.674、两倍预算 0.615 / 0.304、作者画布（2880，等效 epoch）0.627、1440 主表 0.647——画布与训练量都不是解释，作者配方多出的 0.03–0.05 归于协议替换的增强（±45° 旋转、无光度扰动）、54 张全训无验证划分、报最后一步。主表行不换（预登记）。
- 独立测试（14 M10）：19 模型 × 两源 × 八集合齐；HRDecoder DDR 已在正确归一化下重测（备注 57），DDR 源 all / fam:fundus 集成重算。名次统计（19 模型）：χ² 103.5、临界差 9.90、SAMed 1.38、HRNet 3.63、RETFound 5.00，前 12 不可分；正文全读 `\rn` 键。
- 编码器对照（14 M9、备注 56；collect 读 runs/E5enc_*）：C1、C2 都收齐——MAE-IN1k + UPerNet 0.519（DDR）/ 0.634（TJDR_std）高于 RETFound-MAE 0.466 / 0.589；DINOv2 0.510 / 0.619 高于 RETFound-DINOv2 0.490 / 0.616（C2 −0.020，独立三集合均值 +0.006）。结论：同头同配方同预算、各一个种子下，视网膜预训练在两对里都没有加分；三个非 RETFound-MAE 的编码器高于主表 DDR 列全部行。R5、讨论 Q2、方法已按实数写；图 5d 四个编码器全实测。IDRiD 列（备注 59）10-25 收口后开。
- Refined IDRiD（备注 54）、人类参照（M16a）、分层（M17a）：同 10-22；MedSAM 修订 run 已补评。
- 推理档（14 M9）：测试阶段收齐，R5 全部实数——零样本任意病灶 ≤0.08（nnU-Net 同指标 0.78），SAM 3 选定措辞 0.40 / 0.18，理想框 SAM 0.86 / 0.77 对同信息参照 nnU-Net 0.86 / 0.71（框内勾画不输监督模型，差在找病灶），理想点 0.12 / 0.02。集成（14 M15）：18 人重算完成（DDR 源 all 0.677 / 0.637 / 0.266 / 0.335 / 0.496，R4 已换）；FGADR / Retinal-Lesions 不含 HACDR 的六个组合两源出数（粗标注集合上集成只比最好单模型高 0.01–0.02，细档 0.02–0.06，R4 与讨论分两档写）；all / fam:fundus 在这两个集合等 HACDR——它的 DCN 在 cublasSgemm 报 CUBLAS_STATUS_NOT_SUPPORTED，代码路径、检查点、驱动都与 10-13 / 10-17 的成功运行相同，判为 retiseg-hacdr 环境问题（21 第 87 条）：服务器跑探针（17 第 55 条），10-25 按 --without HACDR-Net 出数并加角标（第 56 条）。
- 论文：R3、R4、R5、R6（含 Refined IDRiD；第一段按八源标注审计实数写——细档每图几十个小连通域，粗标注 / 区域共识每图 1.5–8 个、EX 直径 8–9 倍，FGADR 居中；方法加 Annotation style 一段）、R7、R8、讨论 Q2/Q3、方法已是实数，图 6a 全实测；仍是 simtext：L1/E1 段、B1 一句、摘要。待动：FGADR / RL 的 all 与 fam:fundus（10-25 兜底后换数、表 5 角标 p）、N8 参考模型小节、编码器 IDRiD 列进补充表。
- 下一步（17 第四节第 55–58 条，再 47–49）：服务器——HACDR cuBLAS 探针（55，只贴输出）；10-25 的 --without 兜底四份（56，CPU 约 4 小时）后删成员图；收口后同时开 RetiSeg-Ref 两格（47；queue 的 ref_model 阶段改 enabled: true）与编码器 IDRiD 两格（48）；FCT 修正配方最低优先（49）。本机——ensemble.py 加按目录指定成员的入口（参考模型两成员的集成，runs/ENS/REF/<集>，paper 键 N8 / reference:ensemble）；N8 小节；兜底出数后换 R4 的 all / fam:fundus 两列；N6、N7（CPU）仍未做；HRDecoder 的 L1 抽取待做；服务器 GPU 2–7 现在全空。
- 医生：三名眼科医生，每人 3 小时（26 第四节）；T1–T3 材料等联系好再做。
- 用户待办：提交推送本机写回；与医生约 T3。

## 数据流（结果 → 论文）
服务器 `python bench/scripts/pack_handoff.py` 打包 `runs/` 小文件 → 本机解到仓库 `runs/`（包顶层还有 docs/ops/00 的服务器版本，不要整包覆盖到仓库根）→ `python docs/paper/retiseg_paper/paper/scripts/collect_runs.py`（认 BS/B1/*_registered/NN_*/E1r/E2c/BF_*/E1_<slug>/runs/E3a/runs/E5；排除名单不对齐的报告拒收；BSprobe_ 不读）→ `make -C docs/paper/retiseg_paper/paper`。`collect_literature.py` 写 L1 行。整表重生成只在有 `runs/` 的机器上：`build_placeholder.py --force` → `collect_literature.py` → `collect_runs.py` → make。

## 规则（不要违反）
协作规则在 17 第一节：服务器只执行 HEAD 里的 17；一台主机一个启动器、一份队列（`bench/configs/queue.yaml`，`launch_queue.py`，所有要用的卡写进 RETISEG_GPUS；两倍预算探测在拿出来的一张卡上跑）；排除名单不对齐的指标不进 14、不打包、collect 拒收；服务器只追加 00 第三节，14/13/CHECKPOINT/论文由本机改；回传用 `bench/scripts/pack_handoff.py`。看过测试指标后不改预算、增强、选择规则；`bench/configs/*_frozen.yaml` 只增不改（e5_frozen.yaml 在读到首个 E5 测试指标前允许改写，之后只增）。集成的组合规则、读者研究的名单与端点规则都是预登记（14 M15–M17、备注 47–48），登记后不改。每个论文相关决定记 14，每次改动记 00；新文档从 27 起编。

## 自动测试
GitHub Actions（`.github/workflows/tests.yml`）：每次 push/PR 用 CPU torch 跑 `bench/tests`（去掉 H2Former 前向，数据与权重相关的用例自动跳过）。红了先看是不是缺跳过标记或测试读了机器状态（21 第 52 条），记 21，不放松断言。协议 sha256 锁文件在读出首个 E5 测试指标后加（21 第 49 条）。

## 本机环境限制
云端工作区没有 torch；涉及 torch 的测试要在用户机器或服务器跑；不依赖 torch 的队列与文档测试在云端可跑（2026-10-19 核过 test_launch_queue 的 11 个）。云端能 clone 公开仓库、能装 pandas/matplotlib/TeX Live，所以 collect 与 make 可以在云端做，结果写回本机。
两台本地机器：docs 里的"本机"是 Luming 的工作站（4060 Ti，有数据、torch 2.6）；"工作站albert"是 Albert 的机器（RTX 5090，无数据和权重，torch 2.11，另有 retiseg-sam3 环境），只做代码和单元测试。
