# retiseg 检查点（2026-10-22，10-22 回传包进表：MedSAM 种子 0、Refined IDRiD、探测三条、RETFound 三种子之后）

给下一次会话或其他 agent 的最小上下文。细节都在仓库里，按下面的路径读；本文件只说"在哪、什么状态、怎么接着做"。

## 项目是什么
DR 眼底病灶分割（MA/HE/EX/SE）的 assessment 论文，体例照 Lu 2026 Nature Methods。仓库 github.com/skylynf/retiseg（2026-10-19 起公开；用户本机 `~/research/retiseg`，服务器 `/home/qurr/luming/retiseg/retiseg`，8×A100，用 GPU 2–7）。三个问题：Q1 复现（作者权重、作者配方）、Q2 统一比较（B-std 主表、独立测试、基础模型四档）、Q3 决定因素（训练来源、标注粒度、分辨率、训练量、损失）。2026-10-18 起另有 `docs/paper/26`：高层评估、补充实验（HRDecoder 案例、集成、人类参照、临床端点等）、临床医生方案、投稿策略。

## 先读哪些文件（按顺序）
1. `docs/ops/00_进度.txt` —— 当前状态、决定日志、待用户决定的清单、最近的服务器记录（旧记录在 `docs/ops/archive/`）。
2. `docs/paper/26_高层评估与补充实验.txt` —— 当前计划的增补部分（风险、新实验 N1–N8 与预登记规则、医生 T1–T3、排期与先砍顺序）；`docs/ops/17_服务器任务单.txt` —— 规则、回传格式、队列顺序、服务器还要做的事（第四节第 19–24 条是 26 的服务器部分）；`docs/paper/16_范围精简.txt` —— 实验范围（第十节是 26 的增补）；`docs/paper/19` 第六节 —— 优先级与写作里程碑。
3. `docs/paper/13` 第三节状态表（末尾"26 增补"块）；`docs/paper/14` 论文里要用的数字和方法（M1–M17、备注 1–55）。
4. E5 推理档：`docs/ops/25_albert合并记录.txt`、`docs/design/23`、`24`（D1–D9 的依据在 24 第十七节）。
5. 论文源：`docs/paper/retiseg_paper/paper/`（`STRUCTURE.md` 图表清单；`README.md` 哪些是实数）。

## 现状（2026-10-22；仓库 origin/main e94ea60 加本次写回）
- 主表（14 M7、M9）：19 行全部有种子 0 并进表。`data/results.csv` 1,387 份报告、56,815 行实测。MedSAM 的 DDR run 在 MA、SE 上塌缩（验证整个训练在先验水平），按 10-12 与 U-Net 绑定的登记规则改整批 Dice、两库从初始权重重跑（14 备注 55；bs_frozen.yaml recipe_revisions 只增；塌缩 run 改名 _registered）。RETFound 三种子齐（DDR 0.450 ± 0.018，与 HRNet 的差只剩 0.003）；SAMed DDR 两种子 0.475 ± 0.009。
- HRDecoder 案例 N1（14 备注 46）：作者配方 0.674、两倍预算 IDRiD 0.615 / DDR 0.304（一倍 0.647 / 0.296）、HRNet DDR 两倍 0.469（0.447）——训练量不是解释。2880 画布格（N1c）两次退出，根因是包装器 `author_recipe` 从类卡重建卡片丢掉 `set_patch` 的 1920 窗口，预算按 960 块算成 83000 步（21 第 82 条）；已改，预算回到登记口径（21000 步、每步 4 倍像素，约 8 小时），服务器删目录重接（17 第四节第 37 条）。
- 独立测试（14 M10）：19 模型 × 两源 × 八集合种子 0 齐。两处待改：HRDecoder DDR 的八个集合在错误归一化下推理（e3a / predict 用了类卡的 IDRiD 值；代码已改，服务器重测并重算该 run 的分层与 DDR 源 all / fam:fundus 集成，21 第 83 条、17 第 38 条）；MedSAM 修订 run 进表后名次统计重算（现按 19 个模型：χ² 109.7、临界差 9.90、前 12 不可分）。名次段的正文全部读 `\rn` 键（make_numbers 新增 cd.second/third、indep.*.nlower/gain）。
- Refined IDRiD N4（14 备注 54，experiment REF）：27 张几何全部接受（pad，NCC ≥ 0.9988）；修订标注对 18 个模型的 mAUPR 改动 −0.002 至 0、名次不变（τ = 1.00）；1024 重评与 1440 主表差 ≤ 0.006。已写 R6 一段、方法 Label revision 一段、补充表 8；make_numbers 的 ref.* 键。
- 集成（14 M15）、人类参照（M16a）、分层（M17a）：同 10-21；MedSAM 不进集成（组合登记后不增减）。
- 推理档（14 M9）：测试阶段按代价从大到小走，FGADR、TJDR_uwf 已出（SAM 3 第三句 FGADR 0.108、TJDR_uwf 0.185；SAM 2 密度 128 FGADR 0.0096；框 kall FGADR 0.825、TJDR_uwf 0.922），IDRiD、DDR 最后；R5 等它们。影响因素（14 M9b）：29 格里 27 格进表（RETFound 原分辨率 0.449 已进；RETFound E3b TJDR_std 收尾；2880 重开）。编码器对照三格（MAE-IN1k → RETFound-DINOv2 → DINOv2，GPU 6 顺序，各约 27 小时）约 10-25 早晨收完，是队列收口的最长链。
- 论文：R3、R4（含 N1、集成、人类参照、三种子）、R6（含 Refined IDRiD）、R7、R8、讨论 Q2/Q3、方法已是实数；表 6 的 PEFT / 编码器行按数据显示实数；仍是 simtext：R5、L1/E1 段、B1 一句、摘要。表的待动项：MedSAM 行（修订后）、HRDecoder DDR 的表 5 行、集成 FGADR / Retinal-Lesions 列、2880 行、编码器对照（图 5d/e）。
- 下一步（17 第四节第 37–42 条）：服务器 pull 后 37（删 2880 目录）、38（HRDecoder DDR E3a 改名重测 + 分层 + 集成重算）、39（MedSAM 改名重跑）；40–42 按队列。本机：每包进表后更新 R4 名次数、R5（IDRiD / DDR 测试档到后一次写成）、N1c / MedSAM / 编码器句子；N6、N7（CPU）仍未做；HRDecoder 的 L1 抽取待做。
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
