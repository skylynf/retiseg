# retiseg 检查点（2026-10-19，10-19 回传包进表、26 的计划并入之后）

给下一次会话或其他 agent 的最小上下文。细节都在仓库里，按下面的路径读；本文件只说"在哪、什么状态、怎么接着做"。

## 项目是什么
DR 眼底病灶分割（MA/HE/EX/SE）的 assessment 论文，体例照 Lu 2026 Nature Methods。仓库 github.com/skylynf/retiseg（2026-10-19 起公开；用户本机 `~/research/retiseg`，服务器 `/home/qurr/luming/retiseg/retiseg`，8×A100，用 GPU 2–7）。三个问题：Q1 复现（作者权重、作者配方）、Q2 统一比较（B-std 主表、独立测试、基础模型四档）、Q3 决定因素（训练来源、标注粒度、分辨率、训练量、损失）。2026-10-18 起另有 `docs/paper/26`：高层评估、补充实验（HRDecoder 案例、集成、人类参照、临床端点等）、临床医生方案、投稿策略。

## 先读哪些文件（按顺序）
1. `docs/ops/00_进度.txt` —— 当前状态、决定日志、待用户决定的清单、最近的服务器记录（旧记录在 `docs/ops/archive/`）。
2. `docs/paper/26_高层评估与补充实验.txt` —— 当前计划的增补部分（风险、新实验 N1–N8 与预登记规则、医生 T1–T3、排期与先砍顺序）；`docs/ops/17_服务器任务单.txt` —— 规则、回传格式、队列顺序、服务器还要做的事（第四节第 19–24 条是 26 的服务器部分）；`docs/paper/16_范围精简.txt` —— 实验范围（第十节是 26 的增补）；`docs/paper/19` 第六节 —— 优先级与写作里程碑。
3. `docs/paper/13` 第三节状态表（末尾"26 增补"块）；`docs/paper/14` 论文里要用的数字和方法（M1–M17、备注 1–49）。
4. E5 推理档：`docs/ops/25_albert合并记录.txt`、`docs/design/23`、`24`（D1–D9 的依据在 24 第十七节）。
5. 论文源：`docs/paper/retiseg_paper/paper/`（`STRUCTURE.md` 图表清单；`README.md` 哪些是实数）。

## 现状（2026-10-19；仓库 origin/main 5d1fb72 之后，加本次写回）
- 主表（14 M7、M9）：19 行里 18 行有种子 0 并进表（只缺 MedSAM，等权重）。`data/results.csv` 611 份报告、30,601 行实测。SAMed 0.717 / 0.469、RETFound+UPerNet 0.708 / 0.466 在两库最高；HRNet-W48 0.700 / 0.447 是按发布初始化训练的网络里最高；SAM2-UNet 修订后 0.664 / 0.360；CLC-Net（轨 C）0.661 / 0.414 是专用一族在 DDR 上的最高；HRDecoder 0.647 / 0.296 远低于其发布检查点在本协议下的 0.707 / 0.461——包装器核对无误，归因实验是 26 的 N1（作者配方重训、两倍预算、2880 画布），措辞规则在 14 备注 46。
- 独立测试（14 M10）：18 模型 × 两源 × 八集合种子 0 全齐，两个作者权重模型、nnU-Net E3b 两检查点全齐。表 5 平均名次 SAMed 1.38、HRNet 3.50、RETFound 3.63；Friedman χ² 106.3，临界差 9.90，前 12 不可分。结论不变：性能取决于测试集的标注方式与视野。
- 推理档（14 M9）：a 档 DDR 验证池出数（任意病灶 AUPR ≤0.09；SAM 3 三句措辞差一个数量级）；IDRiD 验证池、选择、b 档、测试阶段待跑。
- 影响因素（14 M9b）：U-Net / M2MRF 全部格子、M2MRF E7 三档、nnU-Net E3b 两格进表；RETFound 四个重格子在训，SAM2-UNet E3b 在训 / 等重开。R5 文字未写。
- 26 增补的状态：预登记已写进 14（M15 集成、M16 人类参照、M17 临床端点、备注 46–49）；queue.yaml 第四版（e5_rest 只限 SAMed、RETFound 并打开）；两倍预算的三个 cell 在仓库（`bs_probe_hrnet_ddr_seed0`、`bs_probe_hrdecoder_{idrid,ddr}_seed0`），服务器按 17 第四节第 19 条跑；HRDecoder E1r 入口、集成脚本、factor.py 的 2880 画布、T1/T2 的材料待本机写。
- 医生：三名眼科医生，每人 3 小时（26 第四节：T3 规则会议 0.5 小时 → T1 错误裁定 1.5 小时 → T2 点标记 1 小时）。T3 先于一切。
- 用户待办：提交推送本机写回；下载 MedSAM、DINOv2 权重与 Refined IDRiD；与医生约 T3。

## 数据流（结果 → 论文）
服务器 `python bench/scripts/pack_handoff.py` 打包 `runs/` 小文件 → 本机解到仓库 `runs/`（包顶层还有 docs/ops/00 的服务器版本，不要整包覆盖到仓库根）→ `python docs/paper/retiseg_paper/paper/scripts/collect_runs.py`（认 BS/B1/*_registered/NN_*/E1r/E2c/BF_*/E1_<slug>/runs/E3a/runs/E5；排除名单不对齐的报告拒收；BSprobe_ 不读）→ `make -C docs/paper/retiseg_paper/paper`。`collect_literature.py` 写 L1 行。整表重生成只在有 `runs/` 的机器上：`build_placeholder.py --force` → `collect_literature.py` → `collect_runs.py` → make。

## 规则（不要违反）
协作规则在 17 第一节：服务器只执行 HEAD 里的 17；一台主机一个启动器、一份队列（`bench/configs/queue.yaml`，`launch_queue.py`，所有要用的卡写进 RETISEG_GPUS；两倍预算探测在拿出来的一张卡上跑）；排除名单不对齐的指标不进 14、不打包、collect 拒收；服务器只追加 00 第三节，14/13/CHECKPOINT/论文由本机改；回传用 `bench/scripts/pack_handoff.py`。看过测试指标后不改预算、增强、选择规则；`bench/configs/*_frozen.yaml` 只增不改（e5_frozen.yaml 在读到首个 E5 测试指标前允许改写，之后只增）。集成的组合规则、读者研究的名单与端点规则都是预登记（14 M15–M17、备注 47–48），登记后不改。每个论文相关决定记 14，每次改动记 00；新文档从 27 起编。

## 自动测试
GitHub Actions（`.github/workflows/tests.yml`）：每次 push/PR 用 CPU torch 跑 `bench/tests`（去掉 H2Former 前向，数据与权重相关的用例自动跳过）。红了先看是不是缺跳过标记或测试读了机器状态（21 第 52 条），记 21，不放松断言。协议 sha256 锁文件在读出首个 E5 测试指标后加（21 第 49 条）。

## 本机环境限制
云端工作区没有 torch；涉及 torch 的测试要在用户机器或服务器跑；不依赖 torch 的队列与文档测试在云端可跑（2026-10-19 核过 test_launch_queue 的 11 个）。云端能 clone 公开仓库、能装 pandas/matplotlib/TeX Live，所以 collect 与 make 可以在云端做，结果写回本机。
两台本地机器：docs 里的"本机"是 Luming 的工作站（4060 Ti，有数据、torch 2.6）；"工作站albert"是 Albert 的机器（RTX 5090，无数据和权重，torch 2.11，另有 retiseg-sam3 环境），只做代码和单元测试。
