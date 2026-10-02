# -*- coding: utf-8 -*-
from pathlib import Path
ROOT = Path("/home/lelel/research/retiseg/literature")
META = ROOT / "meta"
R = []

def add(**k):
    R.append(k)

def render(d):
    keys = ["文献ID","PDF","批次","角色","主任务纳入","归类","身份核对","题名","作者","年份","出处","标识","任务","成像","病灶","数据","标注粒度","模型族","额外输入","报告指标","代码","权重","重训轨","摘要","管理备注"]
    d = dict(d)
    d.setdefault("管理备注", "可在文末追加笔记。论文表格中的数字必须注明来源：原文汇报、作者权重或本项目重训。")
    lines = []
    for k in keys:
        lines.append(f"{k}: {d[k]}")
    return "\n".join(lines) + "\n"

# The records follow. Each add() is one paper.

ROWS = []

def row(*a):
    ROWS.append(a)

# pdf, id, batch, role, include, tag, check, title, authors, year, venue, ident, task, imaging, lesions, data, grain, family, extra, metrics, code, weights, track, summary

row("batch1/00_template/Lu2026_TCR_epitope_NatureMethods.pdf","T-NM-2026","模板","方法学模板","否","模板","一致",
"Assessment of computational methods in predicting TCR-epitope binding recognition","Yanping Lu et al.","2026","Nature Methods 23:248-259","doi:10.1038/s41592-025-02910-0",
"TCR-表位结合预测的统一评估","不适用","不适用","21 个数据来源","不适用","评估框架","阴性样本来源；已见与未见表位；原文权重与重训","主指标 AUPRC",
"论文写评估代码在 GitHub，链接在排版中被截断","不适用","不重训",
"评估约 50 个模型，其中约 31 个可以在统一数据上重训。原文结果和重训结果分开，并用独立测试集解释性能。本文只用它的体例，不把它算作糖尿病视网膜病变文献。")

row("batch1/00_template/Lu2026_TCR_epitope_SI.pdf","T-NM-2026-SI","模板","补充表","否","模板","一致",
"Supplementary information for the Nature Methods assessment","Yanping Lu et al.","2026","Nature Methods supplementary","doi:10.1038/s41592-025-02910-0",
"数据表与模型表","不适用","不适用","Supplementary Table 1 与 2","不适用","评估框架","无","无","不适用","不适用","不重训",
"数据集表和模型表的格式来源。本项目台账按这个用法记录来源、规模、输入特征、代码和评估场景。")

row("batch1/01_prior_reviews/Playout2024_cross_dataset_generalization.pdf","R-PLAYOUT-2024","batch1","先前基准","作为先前基准","综述","一致，arXiv:2405.08329",
"Cross-dataset generalization for retinal lesions segmentation","Clement Playout, Farida Cheriet","2024","arXiv:2405.08329","arXiv:2405.08329",
"跨数据集病灶分割","彩色眼底","EX, CWS/SE, HE, MA","IDRiD, DDR, FGADR, Retinal-Lesions, MAPLES-DR","细、粗、混合三档","U-Net 类基线、权重平均与集成","无","分割指标",
"已读文本未发现作者代码","未发现","不把文中临时基线当作专用模型",
"最接近的已有工作。细标注数据集之间较容易迁移，混入粗标注的 Retinal-Lesions 往往会拉低细标注测试。本项目在此之上补上专用模型复现和基础模型。")

row("batch1/01_prior_reviews/applsci-13-05111.pdf","R-SEBASTIAN-2023","batch1","叙述性综述","否，范围更宽","综述","一致",
"A survey on diabetic retinopathy lesion detection and segmentation","Anila Sebastian et al.","2023","Applied Sciences 13:5111","doi:10.3390/app13085111",
"综述检出与分割","彩色眼底","MA, HE, EX，并含血管和视盘","IDRiD, DDR, e-ophtha, DIARETDB, MESSIDOR, DRIVE, STARE","未统一","综述","无","汇集原文数字","不适用","不适用","不重训",
"覆盖病灶、血管和视盘，没有统一重训。只作检索地图，不采用它汇集的数字。")

row("batch1/01_prior_reviews/s11042-023-18089-52.pdf","R-TAHIR-2024","batch1","系统综述","否，只覆盖微动脉瘤","综述","一致",
"Advances in retinal microaneurysms detection, segmentation and datasets","Muhammad Zeeshan Tahir, Muhammad Nasir, Sanyuan Zhang","2024","Multimedia Tools and Applications 83:74897-74935","doi:10.1007/s11042-023-18089-5",
"微动脉瘤检出与分割","彩色眼底","MA","DIARETDB, IDRiD, DDR, APTOS, ROC, e-ophtha, FGADR","区分图像级与像素级","综述","无","检出与分割指标","不适用","不适用","不重训",
"把方法分成传统图像处理和深度学习。支持把 ROC 放在病灶级检出，不放进像素重叠榜。")

row("batch1/02_datasets/data-03-00025.pdf","D-IDRID-2018","batch1","数据集","是","数据集-细像素","一致",
"Indian Diabetic Retinopathy Image Dataset (IDRiD)","Prasanna Porwal et al.","2018","Data 3(3):25","doi:10.3390/data3030025",
"分级、病灶分割、视盘","彩色眼底","MA 81, EX 81, HE 80, SE 40，另有视盘","516 张中 81 张有像素标注","细像素，专家复核轮廓","数据","同时有 DR/DME 分级和黄斑中心","挑战协议见另一篇",
"https://idrid.grand-challenge.org/","不适用","官方划分用于数据集内测试",
"四类里样本最少、标注最细的公开集。软性渗出只有约 40 张。分辨率高，下采样会损失微动脉瘤。")

row("batch1/02_datasets/Porwal2020_IDRiD_challenge.pdf","D-IDRID-CHAL-2020","batch1","挑战协议","是","数据集-细像素","一致，本地是投稿稿",
"IDRiD: Diabetic Retinopathy - Segmentation and Grading Challenge","Prasanna Porwal et al.","2020","Medical Image Analysis 59:101561","doi:10.1016/j.media.2019.101561",
"ISBI 2018 挑战总结","彩色眼底","MA, HE, EX, SE","IDRiD 官方划分","细像素","挑战报告","无","挑战当时的分割与分级指标",
"文中 GitHub 指向参赛者的 Mask R-CNN，不是官方代码","不适用","作为协议，不重训",
"记录 2018 年的任务定义。本地 PDF 是 Elsevier 投稿稿。后来论文常沿用这里的划分，但指标并不统一。")

row("batch1/02_datasets/li2019.pdf","D-DDR-2019","batch1","数据集","是","数据集-细像素","一致，接受稿",
"Diagnostic assessment of deep learning algorithms for diabetic retinopathy screening","Tao Li, Yingqi Gao, Kai Wang, Song Guo, Hanruo Liu, Hong Kang","2019","Information Sciences 501:511-522","doi:10.1016/j.ins.2019.06.011",
"分级、检测、分割","彩色眼底，多家医院、多种相机","EX, HE, MA, SE","13673 张分级，757 张像素标注，官方 train/val/test","细像素","数据","同时有检测框","作者已指出分割难于分级",
"https://github.com/nkicsl/DDR-dataset","不适用","官方划分是数据集内主协议",
"四类细像素标注里规模最大的公开集，测试集也更大。分级分数不能代表分割难度。")

row("batch1/02_datasets/Zhou2021_FGADR.pdf","D-FGADR-2021","batch1","数据集","是","数据集-粗或区域","一致，arXiv:2008.09772",
"A benchmark for studying diabetic retinopathy: segmentation, grading, and transferability","Yi Zhou, Boyang Wang, Lei Huang, Shanshan Cui, Ling Shao","2021","IEEE TMI 40:818-828","doi:10.1109/TMI.2020.3037771",
"分割、分割辅助分级、迁移","彩色眼底","MA, HE, EX, SE, IRMA, NV","Seg-set 1842 张；Grade-set 1000 张","混合，大量团块标注","数据加基线","IRMA 与 NV 为扩展标签","分割与二次加权 kappa",
"https://csyizhou.github.io/FGADR/","不适用","跨数据集实验中的粗标注来源",
"规模大，但粒度和 IDRiD、DDR 不同。扩展标签不并入四类平均。")

row("batch1/02_datasets/MAPLES-DR_2024_ScientificData.pdf","D-MAPLES-2024","batch1","数据集","是","数据集-细像素","一致",
"MAPLES-DR","Gabriel Lepetit-Aimon, Clement Playout, Marie Carole Boucher, Renaud Duval, Michael H. Brent, Farida Cheriet","2024","Scientific Data","doi:10.1038/s41597-024-03739-6",
"解剖结构与病灶像素标注","MESSIDOR 彩色眼底","MA, HE, EX, CWS，以及视盘、黄斑、血管等","约 200 张","细像素，七名专科医师","数据","图像须另取 MESSIDOR","数据说明",
"论文提到 LIV4D/AnnotationPlatform 与 gabriel-lepetitaimon/fundus-vessels-toolkit","不适用","细标注外部测试",
"在已有公开图像上重做细标注，并明确对比了粗标注数据集。")

row("batch1/02_datasets/Decenciere2014_MESSIDOR.pdf","D-MESSIDOR-2014","batch1","图像来源","否","数据集-分级依赖","一致",
"Feedback on a publicly distributed image database: the MESSIDOR database","Etienne Decenciere et al.","2014","Image Analysis and Stereology 33:231-234","doi:10.5566/ias.1155",
"分级","彩色眼底","无像素病灶真值","1200 张","图像级","数据","MAPLES 的图像本体","分级","数据集发布页","不适用","不单独做分割",
"没有病灶 mask。保留是因为 MAPLES-DR 的标签贴在这些图像上。")

row("batch1/02_datasets/Wei2020_RetinalLesions_LesionNet.pdf","D-RETLESION-2020","batch1","数据集与模型","数据用于粒度实验；Lesion-Net 进入主比较","数据集-粗或区域","一致",
"Learn to segment retinal lesions and beyond","Qijie Wei, Xirong Li et al.","2020","ICPR","arXiv:1912.11619",
"八类分割、病灶分类、分级","彩色眼底","MA、视网膜内出血、硬性渗出、棉絮斑、玻璃体积血、视网膜前出血、NV、纤维增殖","公开 1593 张","偏粗","Lesion-Net","扩展标签不并入四类平均","F1 与分级",
"https://github.com/WeiQijie/retinal-lesions","待打开仓库","有代码，可列入重训候选",
"八类最全的公开像素集，标注风格与细像素集不同。")

row("batch1/02_datasets/Kauppi2007_DIARETDB1.pdf","D-DIARETDB1-2007","batch1","数据集","谨慎纳入","数据集-粗或区域","一致",
"The DIARETDB1 diabetic retinopathy database and evaluation protocol","Tomi Kauppi et al.","2007","BMVC","BMVC 2007",
"病灶检出","彩色眼底","MA, HE, 硬性渗出, 软性渗出","89 张","四名专家的区域与置信度","数据与协议","无","依赖置信度阈值的 ROC","数据库页面","不适用","可作外部测试，须单独说明真值",
"专家圈的是区域，不是精确轮廓。像素 Dice 不能和 IDRiD 直接并列。")

row("batch1/02_datasets/Kauppi2006_DIARETDB0.pdf","D-DIARETDB0-2006","batch1","数据集","否","数据集-粗或区域","一致",
"DIARETDB0","Tomi Kauppi et al.","2006","技术报告","LUT 技术报告",
"筛查","彩色眼底","粗发现","130 张","粗","数据","无","敏感度与特异度","原网站","不适用","不进入像素主榜",
"DIARETDB1 的前身。只保留作历史协议。")

row("batch1/02_datasets/Decenciere2013_eophtha_TeleOphta.pdf","D-EOPHTHA-2013","batch1","数据集","两类外部测试","数据集-细像素","一致",
"TeleOphta","E. Decenciere et al.","2013","IRBM 34:196-203","doi:10.1016/j.irbm.2013.01.010",
"远程筛查，附带病灶轮廓","彩色眼底","e-ophtha-MA 与 e-ophtha-EX","公开子集","像素轮廓","系统论文","只有两类","筛查与检出",
"https://www.adcis.net/en/third-party/e-ophtha/","不适用","不参与四类平均",
"e-ophtha 的来源。只能报告 MA 与 EX。")

row("batch1/02_datasets/TMI-2009-0430_author.pdf","D-ROC-2010","batch1","检出挑战","病灶级旁支","数据集-检出级","一致，作者版",
"Retinopathy Online Challenge","Meindert Niemeijer et al.","2010","IEEE TMI 29:185-195","doi:10.1109/TMI.2009.2033909",
"微动脉瘤检出","彩色眼底","MA 中心与半径","100 张","圆，不是像素 mask","挑战","无","自由响应曲线","挑战页面","不适用","不计算像素 Dice",
"微动脉瘤检出的经典协议，与像素分割榜分开。")

print("rows", len(ROWS))
# stash
import pickle
pickle.dump(ROWS, open("/tmp/meta_rows.pkl","wb"))
