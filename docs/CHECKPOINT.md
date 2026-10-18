# retiseg 检查点（2026-10-18，服务器 10-18 修复核对之后）

给下一次会话或其他 agent 的最小上下文。细节都在仓库里，按下面的路径读；本文件只说"在哪、什么状态、怎么接着做"。

## 项目是什么
DR 眼底病灶分割（MA/HE/EX/SE）的 assessment 论文，体例照 Lu 2026 Nature Methods。仓库 github.com/skylynf/retiseg（用户本机 `~/research/retiseg`，服务器 `/home/qurr/luming/retiseg/retiseg`，8×A100，用 GPU 2–7）。三个问题：Q1 复现（作者权重、作者配方）、Q2 统一比较（B-std 主表、独立测试、基础模型四档）、Q3 决定因素（训练来源、标注粒度、分辨率、训练量、损失）。

## 先读哪些文件（按顺序）
1. `docs/ops/00_进度.txt` —— 当前状态、决定日志、待用户决定的清单、最近的服务器记录（旧记录在 `docs/ops/archive/`）。
2. `docs/ops/17_服务器任务单.txt`（第二版）—— 规则、回传格式、队列顺序、服务器还要做的事；`docs/paper/19_外部独立测试与优先级.txt` —— 优先级的理由、写作里程碑、MAPLES-DR 与 HRDecoder E1 的接法；`docs/paper/16_范围精简.txt` —— 实验范围。
3. `docs/paper/13` 第三节状态表；`docs/paper/14` 论文里要用的数字和方法（M1–M14、备注 1–40）。
4. E5 推理档：`docs/ops/25_albert合并记录.txt`（取舍与待确认项）、`docs/design/23`、`24`（实现与流程，D1–D9 的依据在 24 第十七节）。
5. 论文源：`docs/paper/retiseg_paper/paper/`（`STRUCTURE.md` 图表清单；`README.md` 哪些是实数）。

## 现状（2026-10-17；仓库 origin/main 183a8b5 之后，加本次写回）
- 主表（14 M7、M9）：19 行里 16 行有种子 0 并进表（`data/results.csv` 470 份报告、23,850 行实测）。B-std 七模型三种子、nnU-Net 两配置；E5 训练档四模型种子 0——SAMed 0.717 / 0.469 与 RETFound+UPerNet 0.708 / 0.466 在两库都高于 HRNet（0.700 / 0.447），SAM-Adapter 0.706 / 0.433；SAM2-UNet DDR 稀有类塌缩（MA 0.023、SE 0.022），按 10-12 登记的规则改整批 IoU、两库重跑（14 备注 44，bs_frozen.yaml 只增）；专用模型 TC-Net 0.653 / 0.361、PMCNet 0.652 / 0.312、WSRFNet 0.582 / 0.234，HRDecoder、CLC-Net 等打分。
- 独立测试（14 M10）：16 个模型 × 两源 × 八集合种子 0 与两个作者权重模型全齐。结论不变：性能取决于测试集的标注方式与视野；SAMed 领先细档独立三集平均（0.404 / 0.426）。R4、摘要、讨论、方法（两次修订）已按实数改。
- 影响因素（14 M9b）：U-Net / M2MRF 的 21 格全部打分进表；E7 原分辨率在训；nnU-Net E3b 在训 / 排队；重格子在后段。R5 文字未写。
- E5 推理档：池 10-18 在 GPU 7 跑（SAM 2 密度 64 DDR 验证池）；SAM 3 按官方示例在 CUDA 上整段 bfloat16 autocast（21 第 61 条），它的项等服务器 pull 到该提交；MedSAM 缺权重跳过；测试档等 `--select` 写回。
- 队列：一条启动器（服务器要按 17 第四节第 15 条重启一次，拿到 claim.json、失败暂停、factor_score）；顺序 e3a 种子 0 → factor_score → 推理池 → e5_seed0（SAM2-UNet 两库重排）→ fundus_seed0 → 测试档（关）→ 因素 → 重格子 → e3a_rest → 种子 1、2（关）。
- 10-18 服务器记录、等下一包进表：SAM2-UNet 修订后 0.664 / 0.360（修订有效）；HRDecoder 种子 0 0.648 / 0.296 反常（21 第 62 条，先核对再解释）；nnU-Net E3b 0.636 / 0.449；M2MRF 原分辨率 0.312。服务器接下来：17 第四节第 16（SAM 3 项等 pull）、17（HRDecoder 核对材料）、18（打包）条。
- 用户待办：提交推送本机写回；两条新决定不同意时改 17（SAM2-UNet 修订是登记规则的执行，不是新决定）。

## 数据流（结果 → 论文）
服务器 `python bench/scripts/pack_handoff.py` 打包 `runs/` 小文件 → 本机解到仓库 `runs/` → `python docs/paper/retiseg_paper/paper/scripts/collect_runs.py`（认 BS/B1/*_registered/NN_*/E1r/E2c/BF_*/E1_<slug>/runs/E3a/runs/E5；排除名单不对齐的报告拒收）→ `make -C docs/paper/retiseg_paper/paper`。`collect_literature.py` 写 L1 行。整表重生成只在有 `runs/` 的机器上：`build_placeholder.py --force` → `collect_literature.py` → `collect_runs.py` → make。

## 规则（不要违反）
协作规则在 17 第一节：服务器只执行 HEAD 里的 17；一台主机一个启动器、一份队列（`bench/configs/queue.yaml`，`launch_queue.py`，所有要用的卡写进 RETISEG_GPUS）；排除名单不对齐的指标不进 14、不打包、collect 拒收；服务器只追加 00 第三节，14/13/CHECKPOINT/论文由本机改；回传用 `bench/scripts/pack_handoff.py`。看过测试指标后不改预算、增强、选择规则；`bench/configs/*_frozen.yaml` 只增不改（e5_frozen.yaml 在读到首个 E5 测试指标前允许改写，之后只增）。每个论文相关决定记 14，每次改动记 00；新文档从 26 起编（22 已并入 20，不再用）。

## 自动测试
GitHub Actions（`.github/workflows/tests.yml`，2026-10-14 写好，用户还没放进仓库）：每次 push/PR 用 CPU torch 跑 `bench/tests`（去掉 H2Former 前向，数据与权重相关的用例自动跳过）。红了先看是不是缺跳过标记或测试读了机器状态（21 第 52 条），记 21，不放松断言。协议 sha256 锁文件在读出首个 E5 测试指标后加（21 第 49 条）。

## 本机环境限制
云端工作区没有 torch、连不上 PyPI；涉及 torch 的测试要在用户机器或服务器跑；torch 导入替身下 bench/tests 的队列与 E5 文件可跑（结果在 00）。
两台本地机器：docs 里的"本机"是 Luming 的工作站（4060 Ti，有数据、torch 2.6）；"工作站albert"是 Albert 的机器（RTX 5090，无数据和权重，torch 2.11，另有 retiseg-sam3 环境），只做代码和单元测试。
