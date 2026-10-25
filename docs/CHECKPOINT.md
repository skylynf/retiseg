# retiseg 检查点（2026-10-26：主文无占位数字；补充材料重做为独立 supplement.pdf；占位清退、ED9 新画、E3b 超广角进表；参考模型 SAMed 格已出、RETFound 格 10-27 完）

给下一次会话或其他 agent 的最小上下文。细节都在仓库里，按下面的路径读；本文件只说"在哪、什么状态、怎么接着做"。

## 项目是什么
DR 眼底病灶分割（MA/HE/EX/SE）的 assessment 论文，体例照 Lu 2026 Nature Methods。仓库 github.com/skylynf/retiseg（2026-10-19 起公开；用户本机 `~/research/retiseg`，服务器 `/home/qurr/luming/retiseg/retiseg`，8×A100，用 GPU 2–7）。三个问题：Q1 复现（作者权重、作者配方）、Q2 统一比较（B-std 主表、独立测试、基础模型四档）、Q3 决定因素（训练来源、标注粒度、分辨率、训练量、损失）。2026-10-18 起另有 `docs/paper/26`：高层评估、补充实验（HRDecoder 案例、集成、人类参照、临床端点等）、临床医生方案、投稿策略。

## 先读哪些文件（按顺序）
1. `docs/ops/00_进度.txt` —— 当前状态、决定日志、待用户决定的清单、最近的服务器记录（旧记录在 `docs/ops/archive/`）。
2. `docs/paper/26_高层评估与补充实验.txt` —— 当前计划的增补部分（风险、新实验 N1–N8 与预登记规则、医生 T1–T3、排期与先砍顺序）；`docs/ops/17_服务器任务单.txt` —— 规则、回传格式、队列顺序、服务器还要做的事（第四节第 19–24 条是 26 的服务器部分）；`docs/paper/16_范围精简.txt` —— 实验范围（第十节是 26 的增补）；`docs/paper/19` 第六节 —— 优先级与写作里程碑。
3. `docs/paper/13` 第三节状态表（末尾"26 增补"块）；`docs/paper/14` 论文里要用的数字和方法（M1–M17、备注 1–61）。
4. E5 推理档：`docs/ops/25_albert合并记录.txt`、`docs/design/23`、`24`（D1–D9 的依据在 24 第十七节）。
5. 论文源：`docs/paper/retiseg_paper/paper/`（`STRUCTURE.md` 图表清单；`README.md` 哪些是实数；`supplement.tex` 补充材料，`make supplement`）。

## 现状（2026-10-26；仓库 origin/main 5d738db 加本次写回）
- 主表（14 M7、M9）：19 行全部实数；MedSAM 是修订配方（整批 Dice；0.642 / 0.413，备注 55），塌缩 run 以 _registered 留在补充表。SAMed、RETFound 三种子齐：DDR 列 SAMed 0.460 ± 0.027、RETFound 0.450 ± 0.018、HRNet 0.447 ± 0.009——种子 0 的"基础模型领先"在三种子下落在散布内，R4 与讨论已按此写。`data/results.csv` 1,698 份报告、77,485 行实测；占位只剩 L2 的作者联系两行与 nnU-Net 的参数量。
- HRDecoder 案例 N1 收口（14 备注 46）：作者配方 0.674、两倍预算 0.615 / 0.304、作者画布（2880，等效 epoch）0.627、1440 主表 0.647——画布与训练量都不是解释，作者配方多出的 0.03–0.05 归于协议替换的增强（±45° 旋转、无光度扰动）、54 张全训无验证划分、报最后一步。主表行不换（预登记）。
- 独立测试（14 M10）：19 模型 × 两源 × 八集合齐；HRDecoder DDR 已在正确归一化下重测（备注 57），DDR 源 all / fam:fundus 集成重算。名次统计（19 模型）：χ² 103.5、临界差 9.90、SAMed 1.38、HRNet 3.63、RETFound 5.00，前 12 不可分；正文全读 `\rn` 键。
- 编码器对照（14 M9、备注 56；collect 读 runs/E5enc_*）：C1、C2 都收齐——MAE-IN1k + UPerNet 0.519（DDR）/ 0.634（TJDR_std）高于 RETFound-MAE 0.466 / 0.589；DINOv2 0.510 / 0.619 高于 RETFound-DINOv2 0.490 / 0.616（C2 −0.020，独立三集合均值 +0.006）。结论：同头同配方同预算、各一个种子下，视网膜预训练在两对里都没有加分；三个非 RETFound-MAE 的编码器高于主表 DDR 列全部行。R5、讨论 Q2、方法已按实数写；图 5d 四个编码器全实测。IDRiD 列两对齐全：MAE-IN1k 0.729、DINOv2 0.739（高于主表 IDRiD 全部行），RETFound-MAE 0.708（主表行），RETFound-DINOv2 0.711；C1 / C2 在 IDRiD −0.021 / −0.028，与 DDR 同号。
- Refined IDRiD（备注 54）、人类参照（M16a）、分层（M17a）：同 10-22；MedSAM 修订 run 已补评。
- 推理档（14 M9）：测试阶段收齐，R5 全部实数——零样本任意病灶 ≤0.08（nnU-Net 同指标 0.78），SAM 3 选定措辞 0.40 / 0.18，理想框 SAM 0.86 / 0.77 对同信息参照 nnU-Net 0.86 / 0.71（框内勾画不输监督模型，差在找病灶），理想点 0.12 / 0.02。集成（14 M15）：18 人重算完成（DDR 源 all 0.677 / 0.637 / 0.266 / 0.335 / 0.496，R4 已换）；FGADR / Retinal-Lesions 不含 HACDR 的六个组合两源出数（粗标注集合上集成只比最好单模型高 0.01–0.02，细档 0.02–0.06，R4 与讨论分两档写）；all / fam:fundus 在这两个集合按 --without HACDR-Net 出数（DDR 源 all 0.291 / 0.278，与最好单模型持平；表 5 角标 p，R4 写明）；HACDR 的 CUBLAS 报错在探针后只剩 seed0 检查点一种解释（21 第 87 条），最后一步探针低优先（17 第 59 条）。
- 论文（10-26）：主文 numbers.tex 346 键 0 占位。不再打算测的占位已清退（14 备注 65：WSRFNet 权重、砍掉的 E1r、B1、E2c 另一半、编码器未测集合、TJDR_std k 曲线、E10 审计量），图表只画实测或标 pending；E3b 在 TJDR_uwf 用 fov 顶替进表、R6 四源矩阵按五模型重算（备注 67）；ED9 新画、ED3 以补充表 15 交代、ED5 撤销、ED7 只画跑过的、ED10b 只画实测列（备注 68）；R4 加 DR 分级一段（备注 69）；表 5 / 6 / 7 的占位误标修掉（备注 70）。补充材料重做：`paper/supplement.tex` + `scripts/make_supplement.py`（`make supplement`）生成 17 张表、3 个说明、4 项数据（备注 66），主文引用改号。仍是 simtext：R4 末尾参考模型一句（等 N8）。图里仍标 simulated 的只有 ED10 的 nnU-Net 参数量。
- 下一步（17 第四节第 63、65–67 条）：服务器——第 65 条算力审计（几十分钟；ED10b 与补充表 8 的三列）；第 66 条 M2MRF 作者配方种子 1、2 出分（10-26 上午）打包；第 63 / 67 条 RETFound 参考格（约 10-27 中午）训完后 e3a 不带 --drop-maps、源内预测、`ensemble.py reference`、打包；第 59 条低优先。本机——N8 进表后写 R4 末尾一句、补充表 11 的参考模型块已自动；E1r 两种子进表后图 3c 自动多两点、R3 一句加散布；算力审计进表后 ED10b / 补充表 8 自动；N6、N7（CPU）仍未做；HRDecoder L1 其余列待抽。用户——10-31 后给作者联系数（图 2a 不画，补充说明 1 提）；ED1 / ED2 在工作站画。
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
