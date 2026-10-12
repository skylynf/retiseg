# retiseg 检查点（2026-10-12）

给下一次会话或其他 agent 的最小上下文。细节都在仓库里，按下面的路径读；本文件只说“在哪、什么状态、怎么接着做”。

## 项目是什么
DR 眼底病灶分割（MA/HE/EX/SE）的 assessment 论文，体例照 Lu 2026 Nature Methods。仓库 github.com/skylynf/retiseg（用户本机 `~/research/retiseg`，服务器 `/home/qurr/luming/retiseg/retiseg`，8×A100，用 GPU 2–7）。三个问题：Q1 复现、Q2 统一比较、Q3 决定因素。

## 先读哪些文件（按顺序）
1. `docs/ops/00_进度.txt` —— 当前状态 + 决定日志（检查点格式，旧记录在 `docs/ops/archive/`）。
2. `docs/paper/16_范围精简.txt` —— 实验范围（剩余约 810 A100 时）；`docs/ops/17_服务器任务单.txt` —— 服务器要做的事和回传格式。
3. `docs/paper/13` 第三节状态表；`docs/paper/14` 论文里要用的数字和方法（M1–M14、备注 1–38）。
4. 论文源：`docs/paper/retiseg_paper/paper/`（`STRUCTURE.md` 图表清单；`README.md` 哪些是实数）。

## 现状（2026-10-12）
- 主表：7 个 B-std 模型 + nnU-Net 两配置有测试数（HRNet 0.700/0.447 第一；M2MRF 0.657/0.369；HACDR 0.562/0.269）。U-Net 改整批 Dice（IDRiD 0.658），DDR 在训。E1r：作者配方 61.19 对已发布 67.55。
- L1 文献审计完成（30 篇，`literature/meta/`）。图 2b/c、图 4a、表 4 等已是实数；E5、独立测试、影响因素仍是占位（图上带 simulated）。
- 服务器在跑：U-Net DDR、E2c DDR 四项、HACDR 种子 1/2 打分 → E3a 种子 0（单卡）。公开权重下载中；SAM 3 / RETFound ×2 / MedSAM 需用户申请。
- 代码已就绪、未在服务器跑：影响因素实验（`bench/factor.py`，28 个 cell，`server_run.sh factor`）；E5 训练档 5 个包装器。
- E5 推理档 `bench/e5/` 已写成（2026-10-12 工作站albert，提交 e886f2a…1992cdf，实现记录 `docs/design/18_E5推理档实现.txt`，服务器步骤 17 第六节）；同日的 peer review 修复（A1–A5、B1–B6、工程项）见 00「E5 推理档评审修复 1–13」各段，其中 D1–D9 的依据、评审修复 7 的选择口径、8 的 fov 回退、11 的图表口径需 Luming 确认（00「评审修复 10」）；还没写：nnU-Net 的 E3b。

## 数据流（结果 → 论文）
服务器打包 `runs/` 小文件（17 第一节格式）→ 本机解到仓库 `runs/` → `python docs/paper/retiseg_paper/paper/scripts/collect_runs.py`（认 BS/B1/*_registered/NN_*/E1r/E2c/BF_*/runs/E3a/runs/E5）→ `make -C docs/paper/retiseg_paper/paper`。`collect_literature.py` 写 L1 行。占位数据由 `build_placeholder.py --force` 重生成后要再跑两个 collect。

## 规则（不要违反）
看过测试指标后不改预算、增强、选择规则；协议文件 `bench/configs/*_frozen.yaml` 只增不改（推理档的 `e5_frozen.yaml` 文件头另有一条：读到第一个 E5 测试指标之前允许改写，2026-10-12 的步骤 0 和评审修复 6、7 都在这个窗口内并逐条登记在 00；之后同样只增不改）；每个会进论文的决定同一次提交写进 14；每次改动在 00 记一笔；文件编号从 23 起（19 E5 推理档流程、20 代码审计、21 现存问题、22 修复建议，2026-10-12 入库）。

## 本机环境限制
云端工作区没有 torch、连不上 PyPI；涉及 torch 的测试（`bench/tests/test_factor.py` 的 run_cell、assert 用例）要在用户机器或服务器跑。
两台本地机器：docs 里的“本机”是 Luming 的工作站（4060 Ti，有数据、torch 2.6）；“工作站albert”是 Albert 的机器（RTX 5090，无数据和权重，torch 2.11，另有 retiseg-sam3 环境），只做代码和单元测试，16GB 虚拟内存上限下整套测试 194 通过 / 18 跳过 / 6 失败 / 4 错误（跳过失败是缺数据，错误是 H2Former 在上限下建不出网络）。Albert 的工作笔记在 `docs/notes(albert)/`（未跟踪）；其中已入库的：`docs/design/19_E5推理档流程.txt`（E5 推理档 14 步流程与 D1–D9 依据）、`docs/design/20_代码审计.md`（2026-10-12 的 peer review）、`docs/ops/21_现存问题.txt`（分级待办清单）、`docs/ops/22_fixPlan.md`（评审配套的修复建议）。
