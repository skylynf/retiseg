临床合作（docs/clinical/）
建立日期 2026-10-28；2026-10-29 按与 Dr. Peranut Chotcomwongse 的讨论修订（31）；2026-11-03 按试判结果修订（32）；2026-11-06 发送前核对（33）；2026-11-07 定位核对（34）。计算部分收尾之后的临床合作：请眼科医生审阅 AI 与数据集标签不一致的区域，判断谁对、错在哪、按 ICDR 会改变什么、是否改变对病人的处理。编号接着全仓库的顺序（docs/README.txt 第一节末尾的规则），本目录是 28–34。

  28_临床合作总计划.txt     总纲（中文）：研究问题、审查单位、抽样（试判已完成 → 正式批次约 100 张，按 ETDRS 位置分层）、阅读流程与问卷、ICDR / DME 后果的定义、分析（ETDRS 区带、DME 对照）、试判、分工、时间表、风险、待决定事项、状态。先读这份。
  29_protocol_en.md         研究方案（英文，v2.1），给合作医生；内容与 28 一致，去掉了内部编号；含逐步的盲法说明和临床定义的出处。
  30_reader_guide_en.md     阅读指南（英文，v2.1），随数据包发给每位读者：打开、保存、每一步怎么答（先答完一张图所有 A 再揭示）、门槛题的例子、ETDRS 网格与 HEX 位置、病灶定义（MA / HG / HEX / CWS）、快捷键。
  31_会议纪要_20261029.txt  与 Dr. Peranut Chotcomwongse 的判读讨论：纪要要点与每一条在仓库里怎么落实。
  32_试判结果与正式批次设计_20261103.txt  试判（11-02，4 张图、18 个区域）的结果与批注；问卷 3.0.0 的修改（ETDRS 网格与 DME、标签与 AI 分开判、IRMA / NV 标点、删 AI 分级题）；正式批次 batch01 的分层设计、分析改动、执行步骤。部分表述已由 33 更正（文中标【11-06 更正】）。
  33_发送前核对_20261106.txt  发给医生前的四项核对：重叠重抽改变入样概率（模拟量化）及修复（每层只抽一次、所有 A 答完再揭示）；临床表述与出处（HEX 区带只是位置，处理由医生判断，4.7 mm）；逐步的盲法；100 张分 20 + 80。
  34_定位核对与正式批次_20261107.txt  服务器 11-07 定位的逐张核对（DDR 约四成不可用，改正 116 张）、配额差额与冻结标记的代码化、服务器下一步（b2 合并改正、b3' 重挖、b4' dry-run、b5 等冻结）。

代码与问卷在仓库根目录的 clinical/：
  clinical/README.md                  流水线用法（服务器上的正式批次、工作站备用、分析、演示）
  clinical/schema/review_schema.json  问卷的唯一来源（版本号与改动记录在文件内；现为 3.1.0）
  clinical/reader_tool/index.html     阅读工具（单个离线 HTML，1.1.0，ETDRS 网格，所有 A 答完再揭示）
  clinical/scripts/                   locate_landmarks.py、mine_discrepancies.py、sample_review.py、build_pack.py、analyze_reviews.py、contours.py、common.py
  clinical/landmarks/batch01/         正式批次的定位改正（corrections_<集合>.json，34 第二节；服务器 locate_landmarks.py apply 合并）
  clinical/designs/                   batch01.yaml（正式批次，提议稿 status: proposed，冻结时改 frozen；100 张分 20 + 80）、demo_idrid.yaml（试判读过的演示包）、demo_idrid_strat.yaml（正式设计的演示）、trial01.yaml 与 trial01_workstation.yaml（作废，未运行）；冻结后不改，改动写新文件
  clinical/selections/                服务器生成的抽样名单（selection.json，即"钥匙"），回传后提交，作预登记
  clinical/tests/                     合成数据上的全流程测试（bench/tests/test_clinical_review.py 把它带进 GitHub Actions）
不进 git（.gitignore）：clinical/work/（挖掘与分析的中间文件）、clinical/packs/（带图像的数据包）、clinical/reviews/（读者导出的答案）；服务器的 runs/CLINREV/（定位、挖掘、抽样、数据包、分析）。

规则：
  1. 设计文件与问卷版本在打包前提交；读者打开数据包以后，设计不改，要改就写新文件、问卷升版本号（28 第八节第 3 条）。
  2. 医生看到的病灶标签是 MA、HG、HEX、CWS；数据与代码里仍是基准的类别代码 MA、HE、EX、SE（对应关系在 build_pack.py 的 CLASS_STYLE）。
  3. 钥匙（selection.json）不随数据包发给医生；数据包里没有数据集名称、原图编号、区域类型、分层、概率、权重和估计的黄斑位置（只带每张图的尺度，供医生没点视盘时画网格）。
  4. 读者的回答只存在工作站 clinical/reviews/ 与备份里，不进公开仓库，直到论文发表。
  5. 论文里怎么写，记 14（新增 M18 临床错误审阅）；发生了什么，记 00；服务器步骤在 17 第四节第 72 条。
