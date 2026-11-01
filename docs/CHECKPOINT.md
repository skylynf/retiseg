# retiseg 检查点（2026-10-30：服务器实验全部收口——因素格按验证选择重评进表、RetiSeg-Ref 八个集合与发布目录、算力审计、mAUPR 区间；论文按新数重写并全文对照；剩下的是用户侧：许可、作者联系数、ED1 / ED2、临床试判）

给下一次会话或其他 agent 的最小上下文。细节都在仓库里，按下面的路径读；本文件只说"在哪、什么状态、怎么接着做"。

## 项目是什么
DR 眼底病灶分割（MA/HE/EX/SE）的 assessment 论文，体例照 Lu 2026 Nature Methods。仓库 github.com/skylynf/retiseg（2026-10-19 起公开；用户本机 `~/research/retiseg`，服务器 `/home/qurr/luming/retiseg/retiseg`，8×A100，用 GPU 2–7）。三个问题：Q1 复现（作者权重、作者配方）、Q2 统一比较（B-std 主表、独立测试、基础模型四档）、Q3 决定因素（训练来源、标注粒度、分辨率、训练量、损失）。2026-10-18 起另有 `docs/paper/26`：高层评估、补充实验（HRDecoder 案例、集成、人类参照、临床端点等）、临床医生方案、投稿策略。

## 先读哪些文件（按顺序）
1. `docs/ops/00_进度.txt` —— 当前状态、决定日志、待用户决定的清单、最近的服务器记录（旧记录在 `docs/ops/archive/`）。
2. `docs/paper/26_高层评估与补充实验.txt` —— 当前计划的增补部分（风险、新实验 N1–N8 与预登记规则、医生 T1–T3、排期与先砍顺序）；`docs/ops/17_服务器任务单.txt` —— 规则、回传格式、队列顺序、服务器还要做的事（第四节第 19–24 条是 26 的服务器部分）；`docs/paper/16_范围精简.txt` —— 实验范围（第十节是 26 的增补）；`docs/paper/19` 第六节 —— 优先级与写作里程碑。
3. `docs/paper/13` 第三节状态表（末尾"26 增补"块）；`docs/paper/14` 论文里要用的数字和方法（M1–M17、备注 1–61）。
4. E5 推理档：`docs/ops/25_albert合并记录.txt`、`docs/design/23`、`24`（D1–D9 的依据在 24 第十七节）。
5. 临床合作（2026-10-28 起）：`docs/clinical/28_临床合作总计划.txt`（总纲）、`docs/clinical/29_protocol_en.md`（英文方案）、`docs/clinical/30_reader_guide_en.md`（阅读指南）、`docs/clinical/31_会议纪要_20261029.txt`（与 Peranut 的讨论与落实）；代码与问卷在仓库根目录 `clinical/`（`clinical/README.md`）。
6. 论文源：`docs/paper/retiseg_paper/paper/`（`STRUCTURE.md` 图表清单；`README.md` 哪些是实数；`supplement.tex` 补充材料，`make supplement`；`27_稿件自查_20261026.txt` 用户侧的自查记录与待办）。

## 现状（2026-10-30；仓库 origin/main cd290fb 加本次写回）
- 主表（14 M7、M9）：19 行全部实数；MedSAM 是修订配方（整批 Dice；0.642 / 0.413，备注 55），塌缩 run 以 _registered 留在补充表。SAMed、RETFound 三种子齐：DDR 列 SAMed 0.460 ± 0.027、RETFound 0.450 ± 0.018、HRNet 0.447 ± 0.009——种子 0 的"基础模型领先"在三种子下落在散布内，R4 与讨论已按此写。`data/results.csv` 1,698 份报告、77,485 行实测；占位只剩 L2 的作者联系两行与 nnU-Net 的参数量。
- HRDecoder 案例 N1 收口（14 备注 46）：作者配方 0.674、两倍预算 0.615 / 0.304、作者画布（2880，等效 epoch）0.642（按验证选择重评后；最后一步是 0.627，备注 71）、1440 主表 0.647——画布与训练量都不是解释，作者配方多出的 0.03–0.05 归于协议替换的增强（±45° 旋转、无光度扰动）、54 张全训无验证划分、报最后一步。主表行不换（预登记）。
- 独立测试（14 M10）：19 模型 × 两源 × 八集合齐；HRDecoder DDR 已在正确归一化下重测（备注 57），DDR 源 all / fam:fundus 集成重算。名次统计（19 模型）：χ² 103.5、临界差 9.90、SAMed 1.38、HRNet 3.63、RETFound 5.00，前 12 不可分；正文全读 `\rn` 键。
- 编码器对照（14 M9、备注 56；collect 读 runs/E5enc_*）：C1、C2 都收齐——MAE-IN1k + UPerNet 0.519（DDR）/ 0.634（TJDR_std）高于 RETFound-MAE 0.466 / 0.589；DINOv2 0.510 / 0.619 高于 RETFound-DINOv2 0.490 / 0.616（C2 −0.020，独立三集合均值 +0.006）。结论：同头同配方同预算、各一个种子下，视网膜预训练在两对里都没有加分；三个非 RETFound-MAE 的编码器高于主表 DDR 列全部行。R5、讨论 Q2、方法已按实数写；图 5d 四个编码器全实测。IDRiD 列两对齐全：MAE-IN1k 0.729、DINOv2 0.739（高于主表 IDRiD 全部行），RETFound-MAE 0.708（主表行），RETFound-DINOv2 0.711；C1 / C2 在 IDRiD −0.021 / −0.028，与 DDR 同号。
- Refined IDRiD（备注 54）、人类参照（M16a）、分层（M17a）：同 10-22；MedSAM 修订 run 已补评。
- 推理档（14 M9）：测试阶段收齐，R5 全部实数——零样本任意病灶 ≤0.08（nnU-Net 同指标 0.78），SAM 3 选定措辞 0.40 / 0.18，理想框 SAM 0.86 / 0.77 对同信息参照 nnU-Net 0.86 / 0.71（框内勾画不输监督模型，差在找病灶），理想点 0.12 / 0.02。集成（14 M15）：18 人重算完成（DDR 源 all 0.677 / 0.637 / 0.266 / 0.335 / 0.496，R4 已换）；FGADR / Retinal-Lesions 不含 HACDR 的六个组合两源出数（粗标注集合上集成只比最好单模型高 0.01–0.02，细档 0.02–0.06，R4 与讨论分两档写）；all / fam:fundus 在这两个集合按 --without HACDR-Net 出数（DDR 源 all 0.291 / 0.278，与最好单模型持平；表 5 角标 p，R4 写明）；HACDR 的 CUBLAS 报错在探针后只剩 seed0 检查点一种解释（21 第 87 条），最后一步探针低优先（17 第 59 条）。
- 论文（10-30）：results.csv 无占位，numbers.tex 886 键；main.pdf 33 页、supplement.pdf 34 页。10-30 包把服务器的最后四件事进表：因素格 20 格按验证选择重评 + 13 格 E3a 重跑（R4 案例段 2880 格 0.642、R6、R7、讨论 Q3 按新数重写并全部改为 \rn 键，图 6 矩阵改为主表种子 0；14 备注 74）；RetiSeg-Ref 八个集合（DDR 0.501、IDRiD 0.710、TJDR_std 0.696、UWF fov 0.302、MAPLES 0.482、FGADR 0.309、RL 0.278、DiaRetDB1 0.339；R4 末段、讨论 Q2 / 建议 / 局限、摘要、引言、Code availability；备注 58）；算力审计 15 个模型（R8 推理一句、ED10b、补充表 8 的“推理峰值显存”；备注 75）；mAUPR 区间（R4 内部段、补充表 4 末列、讨论局限；备注 72）。全文逻辑对照一遍（备注 76）：讨论 Q2 关于 HRDecoder 的解释改为与案例研究一致；E8 的“递减收益”只对 U-Net 成立；原分辨率下 RETFound 的 HE 由跌变升。N6、N7 不做（备注 77）。仍待用户：ED1 / ED2 图像、作者联系数、refs.bib 账本作者、DiaRetDB1 概率图与 DDR / TJDR 派生模型的许可核对（方法里只写“按三个来源的条款”）。
- 下一步：服务器——只剩 17 第四节第 72 条（临床合作）：（a）先按第 63 条重打两格的 insource_IDRiD / TJDR_std 图（第 69 条（f）删了），再 `ensemble.py reference --sets DDR,IDRiD,TJDR_std,TJDR_uwf,MAPLES-DR --regenerate-maps` 留图；（b）等本机通知。第 59、49 条不做。本机——下一包只会是第 72 条的图（不进论文）；论文侧等用户的四项（上一行）；HRDecoder L1 其余列待抽。用户——见 00 用户待办。
- 医生 / 临床合作（docs/clinical/28，10-29 按 31 修订）：与 Dr. Peranut Chotcomwongse 讨论后先试判 15 张图（clinical/designs/trial01.yaml，约 55 个区域、约一小时），反馈后再定正式批次（batch01.yaml 草稿，约 100 张图）。医生看到的标签是 MA / HG / HEX / CWS；只判四类病灶，IRMA、NV 写进讨论与未来方向；B 步的临床后果是一题 ICDR 门槛（No DR↔轻度、轻度↔中度即转诊线、4-2-1 出血条件、DME）加一题"是否改变处理"；分析按到黄斑中心的距离（DD，三档）看标签漏标与临床相关性。问卷 `clinical/schema/review_schema.json`（2.0.0-trial），工具答题区为浅色交替卡片。服务器：17 第四节第 72 条（a）在第 69 条删图前留图，（b）出试判包等通知；来不及时工作站可用 M2MRF 作者权重出备用试判包（trial01_workstation.yaml）。等 Luming：L1 许可核对、D2（试判用哪个模型的图）。
- 用户待办：提交推送本机写回（含 docs/clinical/、clinical/；删除已改名的 clinical/designs/pilot01.yaml 与 clinical/schema/review_schema_v1.json）；L1 许可核对；出试判包并发给 Peranut（28 第九节）。

## 数据流（结果 → 论文）
服务器 `python bench/scripts/pack_handoff.py` 打包 `runs/` 小文件 → 本机解到仓库 `runs/`（包顶层还有 docs/ops/00 的服务器版本，不要整包覆盖到仓库根）→ `python docs/paper/retiseg_paper/paper/scripts/collect_runs.py`（认 BS/B1/*_registered/NN_*/E1r/E2c/BF_*/E1_<slug>/runs/E3a/runs/E5；排除名单不对齐的报告拒收；BSprobe_ 不读）→ `make -C docs/paper/retiseg_paper/paper`。`collect_literature.py` 写 L1 行。整表重生成只在有 `runs/` 的机器上：`build_placeholder.py --force` → `collect_literature.py` → `collect_runs.py` → make。

## 规则（不要违反）
协作规则在 17 第一节：服务器只执行 HEAD 里的 17；一台主机一个启动器、一份队列（`bench/configs/queue.yaml`，`launch_queue.py`，所有要用的卡写进 RETISEG_GPUS；两倍预算探测在拿出来的一张卡上跑）；排除名单不对齐的指标不进 14、不打包、collect 拒收；服务器只追加 00 第三节，14/13/CHECKPOINT/论文由本机改；回传用 `bench/scripts/pack_handoff.py`。看过测试指标后不改预算、增强、选择规则；`bench/configs/*_frozen.yaml` 只增不改（e5_frozen.yaml 在读到首个 E5 测试指标前允许改写，之后只增）。集成的组合规则、读者研究的名单与端点规则都是预登记（14 M15–M17、备注 47–48），登记后不改。每个论文相关决定记 14，每次改动记 00；新文档从 32 起编（28–31 是 docs/clinical/）。

## 自动测试
GitHub Actions（`.github/workflows/tests.yml`）：每次 push/PR 用 CPU torch 跑 `bench/tests`（去掉 H2Former 前向，数据与权重相关的用例自动跳过）。红了先看是不是缺跳过标记或测试读了机器状态（21 第 52 条），记 21，不放松断言。协议 sha256 锁文件在读出首个 E5 测试指标后加（21 第 49 条）。

## 本机环境限制
云端工作区没有 torch；涉及 torch 的测试要在用户机器或服务器跑；不依赖 torch 的队列与文档测试在云端可跑（2026-10-19 核过 test_launch_queue 的 11 个）。云端能 clone 公开仓库、能装 pandas/matplotlib/TeX Live，所以 collect 与 make 可以在云端做，结果写回本机。
两台本地机器：docs 里的"本机"是 Luming 的工作站（4060 Ti，有数据、torch 2.6）；"工作站albert"是 Albert 的机器（RTX 5090，无数据和权重，torch 2.11，另有 retiseg-sam3 环境），只做代码和单元测试。
