# retiseg 检查点（2026-10-14 晚，albert 合并与 MAPLES-DR 数据到位之后）

给下一次会话或其他 agent 的最小上下文。细节都在仓库里，按下面的路径读；本文件只说"在哪、什么状态、怎么接着做"。

## 项目是什么
DR 眼底病灶分割（MA/HE/EX/SE）的 assessment 论文，体例照 Lu 2026 Nature Methods。仓库 github.com/skylynf/retiseg（用户本机 `~/research/retiseg`，服务器 `/home/qurr/luming/retiseg/retiseg`，8×A100，用 GPU 2–7）。三个问题：Q1 复现（作者权重、作者配方）、Q2 统一比较（B-std 主表、独立测试、基础模型四档）、Q3 决定因素（训练来源、标注粒度、分辨率、训练量、损失）。

## 先读哪些文件（按顺序）
1. `docs/ops/00_进度.txt` —— 当前状态、决定日志、待用户决定的清单、最近的服务器记录（旧记录在 `docs/ops/archive/`）。
2. `docs/ops/17_服务器任务单.txt`（第二版）—— 规则、回传格式、队列顺序、服务器还要做的事；`docs/paper/19_外部独立测试与优先级.txt` —— 优先级的理由、写作里程碑、MAPLES-DR 与 HRDecoder E1 的接法；`docs/paper/16_范围精简.txt` —— 实验范围。
3. `docs/paper/13` 第三节状态表；`docs/paper/14` 论文里要用的数字和方法（M1–M14、备注 1–40）。
4. E5 推理档：`docs/ops/25_albert合并记录.txt`（取舍与待确认项）、`docs/design/23`、`24`（实现与流程，D1–D9 的依据在 24 第十七节）。
5. 论文源：`docs/paper/retiseg_paper/paper/`（`STRUCTURE.md` 图表清单；`README.md` 哪些是实数）。

## 现状（2026-10-14 晚；仓库 origin/main 0491603 之后）
- 主表（14 M7）：B-std 七模型三种子齐（FCT DDR 一个种子），nnU-Net 两配置（默认 DDR 三次均值 0.431）；`data/results.csv` 整表重生成后 123 份报告、6,246 行实测，make 与 PDF 通过。
- 独立测试（14 M10）：主变体 55 份；R4 第一段实数（DDR 训练平均降 47–56%，IDRiD 训练降 71–75%）。队列 e3a 阶段最多两张卡继续补；每个新检查点自动排入。
- 第八个测试集 MAPLES-DR：标注与 MESSIDOR 原图已在服务器（`dataset/maples-dr`、`dataset/messidor`）；`bench/data/maples.py` 直接读（不装包），本机两张图核对通过；论文侧第八列已加（细档，官方测试 60 张）。等服务器导出 + 泄漏审计（17 第十节），之后 e3a 自动补。R4 定稿前要测完。
- E5 训练档：未开始，等冒烟（17 第六节）；权重除 MedSAM、公开 DINOv2 外都在 `pretrained/`。
- E5 推理档：albert 分支的实现（`bench/e5/`，`launch_e5_infer.py`，队列阶段 e5_infer_pool / e5_infer_test），关着等 Luming 确认 D1–D9、评审修复 6/7/16、预算规则（00「待用户 Luming 处理」）。第一次池运行把这些定下来。
- 专用模型扩充（`docs/paper/18`）：五个包装器与任务表已写，等服务器冒烟后开 fundus_seed0（17 第八节）。HRDecoder 作者权重已下载，E1 入口 `bench/predict_hrdecoder.py` + `bench.e3a author --model HRDecoder`，等服务器冒烟（17 第九节）。
- 影响因素：M2MRF 四项打分（FGADR 训练的 0.349 待带 106 张排除的重评）；U-Net/M2MRF 其余格子旧 launch_factor 在接；SAM2-UNet/RETFound 的重格子在队列后段；nnU-Net E3b 接线完成，等数据转换（17 第十一节）。
- 队列（`bench/configs/queue.yaml` 第二版）：e3a（≤2 卡）→ e5_seed0 → fundus_seed0（关）→ e5_infer_pool/test（关）→ factor_cheap → nnunet_e3b → factor_e3b_heavy / factor_e7_heavy → 种子 1、2（关）→ e1r（关）。写作按种子 0 的数先写，多种子只换误差棒。

## 数据流（结果 → 论文）
服务器 `python bench/scripts/pack_handoff.py` 打包 `runs/` 小文件 → 本机解到仓库 `runs/` → `python docs/paper/retiseg_paper/paper/scripts/collect_runs.py`（认 BS/B1/*_registered/NN_*/E1r/E2c/BF_*/E1_<slug>/runs/E3a/runs/E5；排除名单不对齐的报告拒收）→ `make -C docs/paper/retiseg_paper/paper`。`collect_literature.py` 写 L1 行。整表重生成只在有 `runs/` 的机器上：`build_placeholder.py --force` → `collect_literature.py` → `collect_runs.py` → make。

## 规则（不要违反）
协作规则在 17 第一节：服务器只执行 HEAD 里的 17；一个启动器一份队列（`bench/configs/queue.yaml`，`launch_queue.py`）；排除名单不对齐的指标不进 14、不打包、collect 拒收；服务器只追加 00 第三节，14/13/CHECKPOINT/论文由本机改；回传用 `bench/scripts/pack_handoff.py`。看过测试指标后不改预算、增强、选择规则；`bench/configs/*_frozen.yaml` 只增不改（e5_frozen.yaml 在读到首个 E5 测试指标前允许改写，之后只增）。每个论文相关决定记 14，每次改动记 00；新文档从 26 起编（22 已并入 20，不再用）。

## 本机环境限制
云端工作区没有 torch、连不上 PyPI；涉及 torch 的测试要在用户机器或服务器跑；torch 导入替身下 bench/tests 的队列与 E5 文件可跑（结果在 00）。
两台本地机器：docs 里的"本机"是 Luming 的工作站（4060 Ti，有数据、torch 2.6）；"工作站albert"是 Albert 的机器（RTX 5090，无数据和权重，torch 2.11，另有 retiseg-sam3 环境），只做代码和单元测试。
