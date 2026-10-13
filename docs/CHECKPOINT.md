# retiseg 检查点（2026-10-13）

给下一次会话或其他 agent 的最小上下文。细节都在仓库里，按下面的路径读；本文件只说“在哪、什么状态、怎么接着做”。

## 项目是什么
DR 眼底病灶分割（MA/HE/EX/SE）的 assessment 论文，体例照 Lu 2026 Nature Methods。仓库 github.com/skylynf/retiseg（用户本机 `~/research/retiseg`，服务器 `/home/qurr/luming/retiseg/retiseg`，8×A100，用 GPU 2–7）。三个问题：Q1 复现、Q2 统一比较、Q3 决定因素。

## 先读哪些文件（按顺序）
1. `docs/ops/00_进度.txt` —— 当前状态 + 决定日志（检查点格式，旧记录在 `docs/ops/archive/`）。
2. `docs/paper/16_范围精简.txt` —— 实验范围（剩余约 810 A100 时）；`docs/ops/17_服务器任务单.txt` —— 服务器要做的事和回传格式。
3. `docs/paper/13` 第三节状态表；`docs/paper/14` 论文里要用的数字和方法（M1–M14、备注 1–38）。
4. 论文源：`docs/paper/retiseg_paper/paper/`（`STRUCTURE.md` 图表清单；`README.md` 哪些是实数）。

## 现状（2026-10-14 04:51 +0800 回传包，本机 10-14 接入）
- 主表数字在 14 M7，`data/results.csv` 124 份报告，make 通过。U-Net 整批 Dice 三种子均值 0.644 / 0.371；HACDR-Net 三种子均值 0.538 / 0.252。nnU-Net 默认 DDR 三次均值 0.431；IDRiD 的重复在训。
- 独立测试（14 M10）：B-std 七模型两库在 FGADR、Retinal-Lesions、TJDR_uwf 齐，nnU-Net 两库七集合齐；R4 第一段已写实数（DDR 训练平均降 47–56%，IDRiD 训练降 71–75%）。TJDR_std、DiaRetDB1、交叉集对 B-std 模型还在队列。E1r 仍只有 M2MRF IDRiD 种子 0（作者函数 61.19，本项目 0.549）。
- 影响因素：M2MRF 四项已打分（14 M9b）。TJDR 标准 0.554；FGADR 完整测试集 0.349（同眼 106 张还没去掉）；DDR 测试集上 DDR+FGADR 抽 383 为 0.336、全集为 0.320。另两档 E4 已训完、还没有 metrics_test.json。E2c 只有验证数（14 备注 17），×1/3 时验证名次对调，学习率未改。
- 服务器在跑（GPU 2–7）：nnU-Net IDRiD rep1；E3a 种子 0（主变体 done 63）；E3b U-Net TJDR、E8 DDR 50%、E8 细粒度合并、E7 长边 512。旧 launch_factor 没杀。GPU 0、1 是他人 vLLM。下一步顺序在 17 第十三节（E3a > E5 训练档种子 0 > 影响因素剩余 > nnU-Net E3b > E5 推理档）。
- 权重：SAM、SAM 2、MAE 和三份 gated（SAM 3、RETFound ×2，sha256 在 e5_frozen.yaml）已在 `pretrained/`。MedSAM 和 DINOv2 没有。E5 训练未开始。
- E5 推理档 `bench/e5/` 已写（10-13；服务器步骤 17 第六节，方法 14 M9“推理档实现”），适配器要在服务器核对。nnU-Net 的 E3b 已接线（10-14；`nnunet_run.sh TJDR_std|FGADR default`）。影响因素运行的评测现在带训练源同眼排除（launch_factor.py）。

## 数据流（结果 → 论文）
服务器打包 `runs/` 小文件（17 第一节格式）→ 本机解到仓库 `runs/` → `python docs/paper/retiseg_paper/paper/scripts/collect_runs.py`（认 BS/B1/*_registered/NN_*/E1r/E2c/BF_*/runs/E3a/runs/E5）→ `make -C docs/paper/retiseg_paper/paper`。`collect_literature.py` 写 L1 行。占位数据由 `build_placeholder.py --force` 重生成后要再跑两个 collect。

## 规则（不要违反）
看过测试指标后不改预算、增强、选择规则；协议文件 `bench/configs/*_frozen.yaml` 只增不改；每个会进论文的决定同一次提交写进 14；每次改动在 00 记一笔；文件编号从 18 起。

## 本机环境限制
云端工作区没有 torch、连不上 PyPI；涉及 torch 的测试（`bench/tests/test_factor.py` 的 run_cell、assert 用例）要在用户机器或服务器跑。
