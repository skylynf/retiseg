# retiseg 检查点（2026-10-23，10-23 回传包进表：2880、MedSAM 修订、HRDecoder DDR 重测、SAMed 三种子、编码器 C1 之后）

给下一次会话或其他 agent 的最小上下文。细节都在仓库里，按下面的路径读；本文件只说"在哪、什么状态、怎么接着做"。

## 项目是什么
DR 眼底病灶分割（MA/HE/EX/SE）的 assessment 论文，体例照 Lu 2026 Nature Methods。仓库 github.com/skylynf/retiseg（2026-10-19 起公开；用户本机 `~/research/retiseg`，服务器 `/home/qurr/luming/retiseg/retiseg`，8×A100，用 GPU 2–7）。三个问题：Q1 复现（作者权重、作者配方）、Q2 统一比较（B-std 主表、独立测试、基础模型四档）、Q3 决定因素（训练来源、标注粒度、分辨率、训练量、损失）。2026-10-18 起另有 `docs/paper/26`：高层评估、补充实验（HRDecoder 案例、集成、人类参照、临床端点等）、临床医生方案、投稿策略。

## 先读哪些文件（按顺序）
1. `docs/ops/00_进度.txt` —— 当前状态、决定日志、待用户决定的清单、最近的服务器记录（旧记录在 `docs/ops/archive/`）。
2. `docs/paper/26_高层评估与补充实验.txt` —— 当前计划的增补部分（风险、新实验 N1–N8 与预登记规则、医生 T1–T3、排期与先砍顺序）；`docs/ops/17_服务器任务单.txt` —— 规则、回传格式、队列顺序、服务器还要做的事（第四节第 19–24 条是 26 的服务器部分）；`docs/paper/16_范围精简.txt` —— 实验范围（第十节是 26 的增补）；`docs/paper/19` 第六节 —— 优先级与写作里程碑。
3. `docs/paper/13` 第三节状态表（末尾"26 增补"块）；`docs/paper/14` 论文里要用的数字和方法（M1–M17、备注 1–57）。
4. E5 推理档：`docs/ops/25_albert合并记录.txt`、`docs/design/23`、`24`（D1–D9 的依据在 24 第十七节）。
5. 论文源：`docs/paper/retiseg_paper/paper/`（`STRUCTURE.md` 图表清单；`README.md` 哪些是实数）。

## 现状（2026-10-23；仓库 origin/main 234b379 加本次写回）
- 主表（14 M7、M9）：19 行全部实数；MedSAM 是修订配方（整批 Dice；0.642 / 0.413，备注 55），塌缩 run 以 _registered 留在补充表。SAMed、RETFound 三种子齐：DDR 列 SAMed 0.460 ± 0.027、RETFound 0.450 ± 0.018、HRNet 0.447 ± 0.009——种子 0 的"基础模型领先"在三种子下落在散布内，R4 与讨论已按此写。`data/results.csv` 1,507 份报告、61,650 行实测。
- HRDecoder 案例 N1 收口（14 备注 46）：作者配方 0.674、两倍预算 0.615 / 0.304、作者画布（2880，等效 epoch）0.627、1440 主表 0.647——画布与训练量都不是解释，作者配方多出的 0.03–0.05 归于协议替换的增强（±45° 旋转、无光度扰动）、54 张全训无验证划分、报最后一步。主表行不换（预登记）。
- 独立测试（14 M10）：19 模型 × 两源 × 八集合齐；HRDecoder DDR 已在正确归一化下重测（备注 57），DDR 源 all / fam:fundus 集成重算。名次统计（19 模型）：χ² 103.5、临界差 9.90、SAMed 1.38、HRNet 3.63、RETFound 5.00，前 12 不可分；正文全读 `\rn` 键。
- 编码器对照（14 M9、备注 56；collect 读 runs/E5enc_*）：C1 出数——MAE-IN1k + UPerNet 0.519（DDR）/ 0.634（TJDR_std）高于 RETFound-MAE 0.466 / 0.589，且高于主表 DDR 列全部行（一个种子）；C2（RETFound-DINOv2 对 DINOv2）约 10-24 晚收完。R5 训练档一段、方法一段已写；图 5d 只画实测编码器。
- Refined IDRiD（备注 54）、人类参照（M16a）、分层（M17a）：同 10-22；MedSAM 修订 run 已补评。
- 推理档（14 M9）：测试阶段已出 DDR、FGADR、Retinal-Lesions、TJDR_uwf 的 a / b 档（DDR：零样本任意病灶 0.03，SAM 2 全框 0.756、全点 0.019）；IDRiD 未出，SAM 3 的 text:selected 拼装未出；R5 零样本段等它们。集成 FGADR / Retinal-Lesions 列等成员图重推（nnU-Net 四成员用新加的 nnunet_e3a.py --regenerate-maps，17 第 43 条）。
- 论文：R3、R4（含 N1 全部、三种子、集成、人类参照）、R5 训练档段、R6（含 Refined IDRiD）、R7、R8、讨论 Q2/Q3、方法已是实数；仍是 simtext：R5 零样本段、L1/E1 段、B1 一句、摘要。待动：C2 句与讨论 Q2 的编码器一句、集成 FGADR / RL 列、R5 零样本段。
- 下一步（17 第四节第 43–45 条）：nnU-Net 成员重推 → 两集合的集成；编码器格补 MAPLES-DR；推理档 IDRiD 按队列。本机：C2 进表后写 R5 结尾与讨论一句；IDRiD 测试档到后一次写成 R5 零样本段；N6、N7（CPU）仍未做；HRDecoder 的 L1 抽取待做。
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
