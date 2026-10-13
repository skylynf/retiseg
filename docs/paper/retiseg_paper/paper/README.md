# paper/

论文正文、表和图的全部源文件。结构和图表清单见 `STRUCTURE.md`；这里只说怎么用。

## 一条命令

```bash
make -C paper            # 图、表、正文数字、参考文献、main.pdf
make -C paper collect    # 从 ../runs 读入已评测的结果，覆盖对应的占位行，然后重建
make -C paper final      # 没有占位行以后用：去掉所有“模拟”标记
make -C paper test       # 用真实的 bench.eval.evaluate 端到端检查 collect_runs.py
```

依赖：Python（numpy、pandas、scipy、matplotlib、pyyaml、pillow）和 TeX Live（pdflatex、latexmk）。统计量直接调用 `bench/eval/stats.py`，与基准用的是同一份代码。

## 目录

```
main.tex                    导言区和章节装配
sections/                   摘要与引言、结果、讨论、方法、扩展数据
tex/retiseg-tables.tex      表格宏（\ph \best \second \sd \up \down \famchip \tier \rlevel …）
tables/generated/*.tex      由 scripts/make_tables.py 生成，不要手改
generated/numbers.tex       由 scripts/make_numbers.py 生成，正文用 \rn{键} 引用
refs.bib                    由 scripts/make_bib.py 从 literature/meta 生成
data/results.csv            唯一的数据表（长表），每行带 source = measured | placeholder
figures/retifig/            style.py（版式与配色）、data.py（读数与聚合）、panels.py（可复用面板）
figures/fig*.py, ed_figs.py 每张图一个脚本，输出到 figures/out/*.pdf 和 *.png
scripts/build_placeholder.py  生成占位数据（带固定种子）；已有 results.csv 时拒绝覆盖
scripts/collect_runs.py       读 runs/*/metrics_*.json，按键覆盖占位行
```

## 哪些是真数，哪些是模拟

2026-10-12 起（服务器快照接入后）真数还包括：B-std 7 个模型全部已打分种子和 nnU-Net 两种配置的内部测试（E2 的 IDRiD、DDR 列，含 11 阈值、Dice、病灶级召回）；U-Net 旧配方的六个 `registered` 运行；E1r 有效批 4 的 M2MRF；E2c 的 ×1 参照；E10 的参数量和训练时（environment.json）；L1 文献审计的全部行（图 2b、c，图 4f 的文献值）。一个（实验, 模型, 变体, 训练源, 测试集）组一旦有实测种子，占位种子就被 `collect_runs.py` 删掉，均值只在实测种子上算。2026-10-13 服务器又跑了一次 collect_runs.py（115 份报告写入 data/results.csv，未跑 make）：补上 HACDR-Net 与 U-Net 整批 Dice 的种子 1、2，E2c 四项验证，nnU-Net DDR 默认第二次，当时已完成的 E3a 主变体，以及四项 M2MRF 影响因素的内部测试。E5、E7、E8、E9 和未打分的 E3b、E4 仍是占位。图还是 10-12 的 make 结果，工作站 make 之后才跟这张表一致。

真数（黑色，图上无标记）：数据集规模和划分；models.yaml 的 38 个变体、分层和复现等级；检索与筛选计数；M2MRF 作者权重的全部数字（67.55/67.24/0.639、49.20/48.94/0.401，以及 16 位存储后的 0.6446、0.426）；E1r 第一次重训（57.63、0.493）；病灶大小分位数和分档边界；缩放后病灶消失比例；U-Net 和 M2MRF 的参数量；6 个 B-std 模型在 IDRiD 上种子 0 的 GPU 时（取自 00）；方法表里各模型保留的设定。

模拟（蓝灰色数字、图上“simulated”角标、正文左侧竖线）：其余全部，包括 B-std 主表、独立测试、基础模型、影响因素、文献审计 L1、标注风格审计里除 IDRiD 以外的点、计算代价的大部分。生成模型只保证形状和量级合理，**不代表任何预期结论**。

`collect_runs.py` 读入一份真实报告后，该报告覆盖的行变成 measured；一个面板只要还用到任何占位行，角标就保留。

## 服务器结果回来以后

1. 把服务器的快照包（docs/ops/17 第一节的格式）解开到仓库根目录的 `runs/`（gitignore）。
2. `make collect`（在本目录）。脚本读 `runs/BS_*`、`B1_*`、`*_registered`、`NN_*`、`E1r_*`、`E2c_*` 和 `runs/E3a/<来源>/<测试集>/metrics.json`，加 `data/cost_measured.tsv`、`data/cost_audit.tsv`，打印读到了哪些报告、替换和删掉了多少占位行。
3. 重读正文里带竖线的段落：数字和“谁第一”会自动更新，措辞不会。

独立测试的推理还没有接进 bench/，所以 `collect_runs.py` 里“独立测试报告叫 `metrics_<测试集>.json`”是这里的假设；接线定下文件名以后改 `TESTSET` 一处。nnU-Net、E1、E3b、E4–E9 的收集入口也要等各自的输出格式定了再加，目前只能手工往 `results.csv` 里按键写行。

## 正文按“已完成”写、目前还没实现的步骤

方法和结果按全部实验做完的口吻写。下面这些在 bench/ 里还没有，写作时对照 00 逐条确认：

- 独立测试推理的接线（E3a）和超广角原分辨率滑窗。
- 泄漏审计（方法里写了做法和“扩展数据图 3”）。
- DiaRetDB1 官方图像级 ROC/WER 与区域级命中；表 5 的 DiaRetDB1 列现在按图像级 AUC 写。
- 训练损失的忽略区（Retinal-Lesions 的 127），E4 之前要补。
- 病灶级的严格匹配辅指标（讨论的局限里提到）。
- nnU-Net、E5 四档、E1r 的 bs4 与 HACDR-Net、L1 抽取。
- 排名检验里独立区组的取法（STRUCTURE.md 第五节第 1 条）还没有写进 docs/ 或 protocol.yaml。

另外有两处是“披露”而不是“待办”：方法第一段写了协议冻结前做过链路评测、视野规则和分档边界改过一次；U-Net 的 B1 配方修订看过一次测试数（14 备注 1）。措辞请你过目。

## 提交前要核对的事实

这些不在 docs/ 里，是按原文常识写的，或者是这次为了排版先填的：

- 表 1 的视野：IDRiD 50°、DDR 45°；DDR 的标注方式写的是 pixel masks。
- 方法里“FGADR 许可禁止再分发派生数据；Retinal-Lesions 按申请条款同样处理”，后一半以申请条款原文为准。
- 文献流程里优先名单写成“27 个 DOI、TJDR 线索、另有一篇由更正 DOI 进入”，合计 29 条去向（纳入 26、排除 3）；06 第六节原文是“27 条 DOI 加 TJDR 线索”，两处请统一。
- 表 6 的 Backbone 列（SAM ViT-H、SAM 2 Hiera-L、MedSAM/SAMed ViT-B、RETFound ViT-L）：按 E5 实际配置改。
- 表 8 的 Swin-Unet 步数（标了 tbd）和 DDR 各模型的步数。
- `refs.bib`：账本条目的作者只有“第一作者 et al.”，需要按 PDF 补全；`make_bib.py` 末尾手写的 7 条（nnU-Net、nnU-Net Revisited、RETFound、Maier-Hein 2018、Demšar 2006、DINOv2、SAM 3）要逐条对出版页。
- 图 2c、表 3b 里 HACDR-Net、WSRFNet 的“论文报告值”是占位，L1 抽取后替换。
- 表 1 里 DiaRetDB1 的概率图能否公开（14 的 M14 没有写，表里标了 tbd）。

## 换版式

正文现在是单栏 A4 加 Times/Helvetica，方便审读。换成期刊模板时只动 `main.tex` 的导言区；表和图不依赖文档类。图的宽度按 180 mm 设计，字号 5–7 pt。
