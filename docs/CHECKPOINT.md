# retiseg 检查点（2026-10-14，含 albert 合并）

给下一次会话或其他 agent 的最小上下文。细节都在仓库里，按下面的路径读；本文件只说“在哪、什么状态、怎么接着做”。

## 项目是什么
DR 眼底病灶分割（MA/HE/EX/SE）的 assessment 论文，体例照 Lu 2026 Nature Methods。仓库 github.com/skylynf/retiseg（用户本机 `~/research/retiseg`，服务器 `/home/qurr/luming/retiseg/retiseg`，8×A100，用 GPU 2–7）。三个问题：Q1 复现、Q2 统一比较、Q3 决定因素。

## 先读哪些文件（按顺序）
1. `docs/ops/00_进度.txt` —— 当前状态 + 决定日志（检查点格式，旧记录在 `docs/ops/archive/`）。
2. `docs/paper/16_范围精简.txt` —— 实验范围；`docs/paper/19_外部独立测试与优先级.txt` —— 现在的优先级、写作里程碑、MAPLES-DR 与 HRDecoder E1 的接法；`docs/ops/17_服务器任务单.txt` —— 服务器要做的事（最新一节是第十六节）和回传格式。
3. `docs/paper/13` 第三节状态表；`docs/paper/14` 论文里要用的数字和方法（M1–M14、备注 1–38）。
4. 论文源：`docs/paper/retiseg_paper/paper/`（`STRUCTURE.md` 图表清单；`README.md` 哪些是实数）。

## 现状（2026-10-14 04:51 +0800 回传包，本机 10-14 接入）
- 主表数字在 14 M7，`data/results.csv` 124 份报告，make 通过。U-Net 整批 Dice 三种子均值 0.644 / 0.371；HACDR-Net 三种子均值 0.538 / 0.252。nnU-Net 默认 DDR 三次均值 0.431；IDRiD 的重复在训。
- 独立测试（14 M10）：B-std 七模型两库在 FGADR、Retinal-Lesions、TJDR_uwf 齐，nnU-Net 两库七集合齐；R4 第一段已写实数（DDR 训练平均降 47–56%，IDRiD 训练降 71–75%）。TJDR_std、DiaRetDB1、交叉集对 B-std 模型还在队列。E1r 仍只有 M2MRF IDRiD 种子 0（作者函数 61.19，本项目 0.549）。
- 影响因素：M2MRF 四项已打分（14 M9b）。TJDR 标准 0.554；FGADR 完整测试集 0.349（同眼 106 张还没去掉）；DDR 测试集上 DDR+FGADR 抽 383 为 0.336、全集为 0.320。另两档 E4 已训完、还没有 metrics_test.json。E2c 只有验证数（14 备注 17），×1/3 时验证名次对调，学习率未改。
- 服务器在跑（GPU 2–7）：nnU-Net IDRiD rep1；E3a 种子 0（主变体 done 63）；E3b U-Net TJDR、E8 DDR 50%、E8 细粒度合并、E7 长边 512。旧 launch_factor 没杀。GPU 0、1 是他人 vLLM。下一步顺序在 17 第十三节（E3a > E5 训练档种子 0 > 影响因素剩余 > nnU-Net E3b > E5 推理档）。
- 权重：SAM、SAM 2、MAE 和三份 gated（SAM 3、RETFound ×2，sha256 在 e5_frozen.yaml）已在 `pretrained/`。MedSAM 和 DINOv2 没有。E5 训练未开始。
- 专用模型扩充（10-14，`docs/paper/18`）：主表专用一族扩到 7 个（HRDecoder、PMCNet、WSRFNet 轨 B；CLC-Net、TC-Net 轨 C），包装器在 `bench/models/`，任务表已生成，queue.yaml 的 fundus_seed0 阶段关着等服务器冒烟（17 第十五节）；用户要加 `official_code/HRDecoder` 子模块并下载其权重。
- 队列第二版（10-14 晚，`bench/configs/queue.yaml`，理由在 19 第四节）：E3a 最多两张卡 → E5 种子 0 → 专用模型种子 0（冒烟后开）→ E5 推理档 a/b → 便宜的影响因素格子 → nnU-Net E3b → 贵的 E3b/E7 格子 → 种子 1、2（关）。写作按种子 0 的数先写，多种子只换误差棒。
- MAPLES-DR 是第八个独立测试集（细档，官方测试 60 张；`bench/data/maples.py`，e3a_frozen 已追加）：等用户向 adcis.net 申请 MESSIDOR 原图，到后服务器 export + 泄漏审计（17 第十六节第 5 条），队列自动接；论文侧表 1/图 6/方法在 report.json 出来后改。
- HRDecoder 作者权重已下载：`bench/predict_hrdecoder.py` + `bench.e3a author --model HRDecoder`（retiseg-hacdr），命令在 17 第十六节第 4 条；表 3a 已留两行。
- E5 推理档（2026-10-14 合并，`docs/ops/25_albert合并记录.txt`）：`bench/e5/` 用 albert 分支的实现（按 16 的范围：零样本点密度 {64, 128} × 面积上限，在 IDRiD+DDR 验证池上按模型选；b 档 SAM ViT-B / SAM 2.1 / MedSAM，k ∈ {1, 5, 20, all}；阈值取验证池），main 10-13 的实现和它生成的 `runs/E5/prompts/` 已删（提交 3d00468 里还在），提示文件由 albert 的 `prompts.py` 重新生成到 `runs/E5_prompts/`。实现记录 `docs/design/23`、流程 `docs/design/24`、评审 `docs/design/20`、待办 `docs/ops/21`。队列里 E5 推理档是 `e5_infer_pool` / `e5_infer_test` 两个阶段（`launch_queue.py` 的 e5_infer 调 `launch_e5_infer.plan`），**现在关着**，等 Luming 确认 D1–D9 和 128 点预算（00「待用户 Luming 处理」与 25 第四节）。
- nnU-Net 的 E3b 已接线（10-14；`nnunet_run.sh TJDR_std|FGADR default`）。影响因素运行的评测现在带训练源同眼排除（launch_factor.py）。

## 数据流（结果 → 论文）
服务器打包 `runs/` 小文件（17 第一节格式）→ 本机解到仓库 `runs/` → `python docs/paper/retiseg_paper/paper/scripts/collect_runs.py`（认 BS/B1/*_registered/NN_*/E1r/E2c/BF_*/runs/E3a/runs/E5）→ `make -C docs/paper/retiseg_paper/paper`。`collect_literature.py` 写 L1 行。占位数据由 `build_placeholder.py --force` 重生成后要再跑两个 collect。

## 规则（不要违反）
协作规则在 17 第十四节：服务器只执行 HEAD 里的 17；一个启动器一份队列（`bench/configs/queue.yaml`，`launch_queue.py`）；排除名单不对齐的指标不进 14、不打包、collect 拒收；服务器只追加 00 第三节，14/13/CHECKPOINT/论文由本机改；回传用 `bench/scripts/pack_handoff.py`。看过测试指标后不改预算、增强、选择规则；协议文件 `bench/configs/*_frozen.yaml` 只增不改（推理档的 `e5_frozen.yaml` 文件头另有一条：读到第一个 E5 测试指标之前允许改写，2026-10-12 的步骤 0、评审修复 6、7 和 2026-10-14 的评审修复 16 都在这个窗口内并逐条登记在 00；之后同样只增不改）；每个会进论文的决定同一次提交写进 14；每次改动在 00 记一笔；文件编号从 26 起（编号表在 docs/README.txt）。

## 本机环境限制
云端工作区没有 torch、连不上 PyPI；涉及 torch 的测试（`bench/tests/test_factor.py` 的 run_cell、assert 用例）要在用户机器或服务器跑。
两台本地机器：docs 里的“本机”是 Luming 的工作站（4060 Ti，有数据、torch 2.6）；“工作站albert”是 Albert 的机器（RTX 5090，无数据和权重，torch 2.11，另有 retiseg-sam3 环境），只做代码和单元测试，16GB 虚拟内存上限下整套测试 194 通过 / 18 跳过 / 6 失败 / 4 错误（跳过失败是缺数据，错误是 H2Former 在上限下建不出网络）。Albert 的工作笔记在 `docs/notes(albert)/`（未跟踪）；其中已入库的：`docs/design/24_E5推理档流程.txt`（E5 推理档 14 步流程与 D1–D9 依据）、`docs/design/20_代码审计.md`（2026-10-12 的 peer review）、`docs/ops/21_现存问题.txt`（分级待办清单）、`docs/ops/22_fixPlan.md`（评审配套的修复建议）。
