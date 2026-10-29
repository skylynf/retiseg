临床合作（docs/clinical/）
建立日期 2026-10-28；2026-10-29 按与 Dr. Peranut Chotcomwongse 的讨论修订（31）。计算部分收尾之后的临床合作：请眼科医生审阅 AI 与数据集标签不一致的区域，判断谁对、错在哪、按 ICDR 会改变什么、是否改变对病人的处理。编号接着全仓库的顺序（docs/README.txt 第一节末尾的规则），本目录是 28–31。

  28_临床合作总计划.txt     总纲（中文）：研究问题、审查单位、抽样（试判 15 张 → 正式批次约 100 张）、阅读流程与问卷、ICDR 后果的定义、分析（含到黄斑的距离）、试判、分工、时间表、风险、待决定事项、状态。先读这份。
  29_protocol_en.md         研究方案（英文），给合作医生；内容与 28 一致，去掉了内部编号。
  30_reader_guide_en.md     阅读指南（英文），随数据包发给每位读者：打开、保存、每一步怎么答、ICDR 后果的例子、病灶定义（MA / HG / HEX / CWS）、快捷键。
  31_会议纪要_20261029.txt  与 Dr. Peranut Chotcomwongse 的判读讨论：纪要要点与每一条在仓库里怎么落实。

代码与问卷在仓库根目录的 clinical/：
  clinical/README.md                  流水线用法（服务器试判、工作站备用、分析、演示）
  clinical/schema/review_schema.json  问卷的唯一来源（版本号在文件内；现为 2.0.0-trial）
  clinical/reader_tool/index.html     阅读工具（单个离线 HTML）
  clinical/scripts/                   mine_discrepancies.py、sample_review.py、build_pack.py、analyze_reviews.py、contours.py、common.py
  clinical/designs/                   trial01.yaml（试判）、trial01_workstation.yaml（备用）、batch01.yaml（正式批次草稿）、demo_idrid.yaml（演示）；冻结后不改，改动写新文件
  clinical/selections/                服务器生成的抽样名单（selection.json，即"钥匙"），回传后提交，作预登记
  clinical/tests/                     合成数据上的全流程测试（bench/tests/test_clinical_review.py 把它带进 GitHub Actions）
不进 git（.gitignore）：clinical/work/（挖掘与分析的中间文件）、clinical/packs/（带图像的数据包）、clinical/reviews/（读者导出的答案）。

规则：
  1. 设计文件与问卷版本在打包前提交；读者打开数据包以后，设计不改，要改就写新文件、问卷升版本号（28 第八节第 3 条）。
  2. 医生看到的病灶标签是 MA、HG、HEX、CWS；数据与代码里仍是基准的类别代码 MA、HE、EX、SE（对应关系在 build_pack.py 的 CLASS_STYLE）。
  3. 钥匙（selection.json）不随数据包发给医生；数据包里没有数据集名称、原图编号、区域类型、概率与权重。
  4. 读者的回答只存在工作站 clinical/reviews/ 与备份里，不进公开仓库，直到论文发表。
  5. 论文里怎么写，记 14（新增 M18 临床错误审阅）；发生了什么，记 00；服务器步骤在 17 第四节第 72 条。
