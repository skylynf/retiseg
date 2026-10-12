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
- 还没写：E5 推理档 `bench/e5/`（零样本、理想提示、SAM 3 文本）；nnU-Net 的 E3b。

## 数据流（结果 → 论文）
服务器打包 `runs/` 小文件（17 第一节格式）→ 本机解到仓库 `runs/` → `python docs/paper/retiseg_paper/paper/scripts/collect_runs.py`（认 BS/B1/*_registered/NN_*/E1r/E2c/BF_*/runs/E3a）→ `make -C docs/paper/retiseg_paper/paper`。`collect_literature.py` 写 L1 行。占位数据由 `build_placeholder.py --force` 重生成后要再跑两个 collect。

## 规则（不要违反）
看过测试指标后不改预算、增强、选择规则；协议文件 `bench/configs/*_frozen.yaml` 只增不改；每个会进论文的决定同一次提交写进 14；每次改动在 00 记一笔；文件编号从 18 起。

## 本机环境限制
云端工作区没有 torch、连不上 PyPI；涉及 torch 的测试（`bench/tests/test_factor.py` 的 run_cell、assert 用例）要在用户机器或服务器跑。
