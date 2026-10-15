# retiseg 检查点（2026-10-15 下午，10-15 回传包进表之后）

给下一次会话或其他 agent 的最小上下文。细节都在仓库里，按下面的路径读；本文件只说"在哪、什么状态、怎么接着做"。

## 项目是什么
DR 眼底病灶分割（MA/HE/EX/SE）的 assessment 论文，体例照 Lu 2026 Nature Methods。仓库 github.com/skylynf/retiseg（用户本机 `~/research/retiseg`，服务器 `/home/qurr/luming/retiseg/retiseg`，8×A100，用 GPU 2–7）。三个问题：Q1 复现（作者权重、作者配方）、Q2 统一比较（B-std 主表、独立测试、基础模型四档）、Q3 决定因素（训练来源、标注粒度、分辨率、训练量、损失）。

## 先读哪些文件（按顺序）
1. `docs/ops/00_进度.txt` —— 当前状态、决定日志、待用户决定的清单、最近的服务器记录（旧记录在 `docs/ops/archive/`）。
2. `docs/ops/17_服务器任务单.txt`（第二版）—— 规则、回传格式、队列顺序、服务器还要做的事；`docs/paper/19_外部独立测试与优先级.txt` —— 优先级的理由、写作里程碑、MAPLES-DR 与 HRDecoder E1 的接法；`docs/paper/16_范围精简.txt` —— 实验范围。
3. `docs/paper/13` 第三节状态表；`docs/paper/14` 论文里要用的数字和方法（M1–M14、备注 1–40）。
4. E5 推理档：`docs/ops/25_albert合并记录.txt`（取舍与待确认项）、`docs/design/23`、`24`（实现与流程，D1–D9 的依据在 24 第十七节）。
5. 论文源：`docs/paper/retiseg_paper/paper/`（`STRUCTURE.md` 图表清单；`README.md` 哪些是实数）。

## 现状（2026-10-15；仓库 origin/main 7462bb1 之后，加本次写回）
- 主表（14 M7）：B-std 七模型三种子齐（FCT DDR 一个种子），nnU-Net 两配置（默认 DDR 三次均值 0.431）；`data/results.csv` 按 10-15 包重 collect：268 份报告、13,224 行实测；表 3 / 表 5 的实测行已不标占位色。
- 独立测试（14 M10）：九个监督模型 × 两源 × 八集合种子 0 全齐，HRDecoder 作者权重八集合全齐。结论：性能取决于测试集的标注方式与视野——DDR 训练的在像素级常规视野来源上持平或更高，在粗 / 区域 / 超广角上降 33–57%；IDRiD 训练的到处都降。R4 两段、摘要、讨论已按实数改。缺：E5 训练档、专用模型、M2MRF 作者权重（.pth 待拷到服务器）。
- 第八个测试集 MAPLES-DR：服务器已导出（138/0/60）并审计（与五个训练源最高 0.343，名单为空）；读法里 duplicates 的键是保留图、值是副本（服务器改对，本机加测试）。论文侧第八列已加（细档，官方测试 60 张），等包进 results.csv。R4 定稿前要测完。
- E1 HRDecoder 作者权重：10-14 完成，作者函数对上 README（71.34 / 49.24），本项目协议 IDRiD 0.707、DDR 0.461（14 M5）；独立测试由 e3a 自动接。
- E5 训练档：四模型冒烟全过；SAM-Adapter DDR、RETFound DDR 种子 0 在 GPU 6、4 上跑（10-15 06:01 起）；SAM2-UNet 批 12 放不下，loader 批在它开跑前以 6/4/2 冒烟定（14 备注 42；服务器先改的批 1 不采用）。
- E5 推理档：冒烟未做（17 第四节第 4 条）；通过后 e5_infer_pool 以优先级 15 一张卡先于训练跑池，`--select` 写回两处 selected 后开测试阶段。表 6/图 5 的口径已按 10-14 的确认改好。
- 专用模型扩充（`docs/paper/18`）：五模型十项冒烟全过（10-15），fundus_seed0 已开，排在 E3a 种子 0 与 E5 种子 0 之后。
- 影响因素：21 个 U-Net/M2MRF 格子训完，M2MRF 的 7 个打分进表（FGADR 内部 0.359 已带排除），13 个没评测（旧启动器的旧代码，21 第 57 条）——队列的 `factor_score` 阶段（优先级 12）补打分；nnU-Net E3b 的数据已在，队列自动接；SAM2-UNet/RETFound 的重格子在后段。
- 服务器 10-15 要做的：17 第四节十条——先收口成一条启动器 `RETISEG_GPUS="2 3 4 5 6 7"`（三条队列同一分钟打开过同一目录；launch_queue 现已加 claim.json 兜底）、移走被三个进程写过的 FCT 目录、SAM2-UNet 的批、推理档冒烟、打包回传、重跑三份测试、算力审计。
- 用户待办：提交推送本机写回；把 M2MRF 的两份作者 .pth 拷到服务器 `runs/E1_m2mrf/`（21 第 58 条）；两条新决定（17 第十四节）不同意时改 17。CI 已绿的前提是 test_registry_cli 的修复已推。

## 数据流（结果 → 论文）
服务器 `python bench/scripts/pack_handoff.py` 打包 `runs/` 小文件 → 本机解到仓库 `runs/` → `python docs/paper/retiseg_paper/paper/scripts/collect_runs.py`（认 BS/B1/*_registered/NN_*/E1r/E2c/BF_*/E1_<slug>/runs/E3a/runs/E5；排除名单不对齐的报告拒收）→ `make -C docs/paper/retiseg_paper/paper`。`collect_literature.py` 写 L1 行。整表重生成只在有 `runs/` 的机器上：`build_placeholder.py --force` → `collect_literature.py` → `collect_runs.py` → make。

## 规则（不要违反）
协作规则在 17 第一节：服务器只执行 HEAD 里的 17；一台主机一个启动器、一份队列（`bench/configs/queue.yaml`，`launch_queue.py`，所有要用的卡写进 RETISEG_GPUS）；排除名单不对齐的指标不进 14、不打包、collect 拒收；服务器只追加 00 第三节，14/13/CHECKPOINT/论文由本机改；回传用 `bench/scripts/pack_handoff.py`。看过测试指标后不改预算、增强、选择规则；`bench/configs/*_frozen.yaml` 只增不改（e5_frozen.yaml 在读到首个 E5 测试指标前允许改写，之后只增）。每个论文相关决定记 14，每次改动记 00；新文档从 26 起编（22 已并入 20，不再用）。

## 自动测试
GitHub Actions（`.github/workflows/tests.yml`，2026-10-14 写好，用户还没放进仓库）：每次 push/PR 用 CPU torch 跑 `bench/tests`（去掉 H2Former 前向，数据与权重相关的用例自动跳过）。红了先看是不是缺跳过标记或测试读了机器状态（21 第 52 条），记 21，不放松断言。协议 sha256 锁文件在读出首个 E5 测试指标后加（21 第 49 条）。

## 本机环境限制
云端工作区没有 torch、连不上 PyPI；涉及 torch 的测试要在用户机器或服务器跑；torch 导入替身下 bench/tests 的队列与 E5 文件可跑（结果在 00）。
两台本地机器：docs 里的"本机"是 Luming 的工作站（4060 Ti，有数据、torch 2.6）；"工作站albert"是 Albert 的机器（RTX 5090，无数据和权重，torch 2.11，另有 retiseg-sam3 环境），只做代码和单元测试。
