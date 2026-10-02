#!/usr/bin/env python3
"""Download open-access PDFs for the DR lesion-segmentation reading list."""

import csv
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path("/home/lelel/research/retiseg/literature")
UA = "retiseg-lit-bot/0.1 (academic literature collection; mailto:research@localhost)"
EMAIL = "research@localhost"

# batch, category, filename, title, doi_or_id, urls
ITEMS = [
    # ----- batch 1: template supplementary -----
    ("batch1", "00_template", "Lu2026_TCR_epitope_SI.pdf",
     "Lu 2026 Nature Methods supplementary", "10.1038/s41592-025-02910-0",
     ["https://static-content.springer.com/esm/art%3A10.1038%2Fs41592-025-02910-0/MediaObjects/41592_2025_2910_MOESM1_ESM.pdf"]),

    # ----- batch 1: prior reviews -----
    ("batch1", "01_prior_reviews", "Playout2024_cross_dataset_generalization.pdf",
     "Cross-dataset generalization for retinal lesions segmentation", "arXiv:2405.08329",
     ["https://arxiv.org/pdf/2405.08329.pdf"]),
    ("batch1", "01_prior_reviews", "Sebastian2023_DR_lesion_survey.pdf",
     "A survey on DR lesion detection and segmentation", "10.3390/app13085111",
     ["https://www.mdpi.com/2076-3417/13/8/5111/pdf?version=1681980000",
      "https://www.mdpi.com/2076-3417/13/8/5111/pdf"]),
    ("batch1", "01_prior_reviews", "MA_SLR_2024_microaneurysm_review.pdf",
     "Advances in retinal microaneurysms detection, segmentation and datasets", "10.1007/s11042-023-18089-5",
     ["https://link.springer.com/content/pdf/10.1007/s11042-023-18089-5.pdf"]),

    # ----- batch 1: datasets -----
    ("batch1", "02_datasets", "Porwal2018_IDRiD_dataset.pdf",
     "IDRiD dataset paper", "10.3390/data3030025",
     ["https://www.mdpi.com/2306-5729/3/3/25/pdf"]),
    ("batch1", "02_datasets", "Porwal2020_IDRiD_challenge.pdf",
     "IDRiD segmentation and grading challenge", "10.1016/j.media.2019.101561",
     []),
    ("batch1", "02_datasets", "Li2019_DDR.pdf",
     "DDR diagnostic assessment", "10.1016/j.ins.2019.06.011",
     []),
    ("batch1", "02_datasets", "Zhou2021_FGADR.pdf",
     "FGADR benchmark", "arXiv:2008.09772",
     ["https://arxiv.org/pdf/2008.09772.pdf"]),
    ("batch1", "02_datasets", "MAPLES-DR_2024_ScientificData.pdf",
     "MAPLES-DR", "10.1038/s41597-024-03739-6",
     ["https://www.nature.com/articles/s41597-024-03739-6.pdf"]),
    ("batch1", "02_datasets", "Decenciere2014_MESSIDOR.pdf",
     "MESSIDOR database", "10.5566/ias.1155",
     ["https://www.ias-iss.org/ojs/IAS/article/download/1155/991",
      "https://www.ias-iss.org/ojs/IAS/article/download/1155/959"]),
    ("batch1", "02_datasets", "Wei2020_RetinalLesions_LesionNet.pdf",
     "Learn to segment retinal lesions and beyond", "arXiv:1912.11619",
     ["https://arxiv.org/pdf/1912.11619.pdf"]),
    ("batch1", "02_datasets", "Kauppi2007_DIARETDB1.pdf",
     "DIARETDB1", "BMVC2007",
     ["https://webpages.tuni.fi/vision/public_data/publications/bmvc2007_diaretdb1.pdf",
      "https://www.it.lut.fi/project/imageret/diaretdb1/doc/diaretdb1_v_1_1.pdf"]),
    ("batch1", "02_datasets", "Kauppi2006_DIARETDB0.pdf",
     "DIARETDB0", "DIARETDB0",
     ["https://www.it.lut.fi/project/imageret/diaretdb0/doc/diaretdb0_v_1_1.pdf",
      "https://www.it.lut.fi/project/imageret/diaretdb0/diaretdb0_v_1_1.zip"]),
    ("batch1", "02_datasets", "Decenciere2013_eophtha_TeleOphta.pdf",
     "TeleOphta / e-ophtha", "10.1016/j.irbm.2013.01.010",
     []),
    ("batch1", "02_datasets", "Niemeijer2010_ROC.pdf",
     "Retinopathy Online Challenge", "10.1109/TMI.2009.2037146",
     []),

    # ----- batch 1: fundus methods -----
    ("batch1", "03_fundus_methods", "Guo2019_L-Seg.pdf",
     "L-Seg", "10.1016/j.neucom.2019.04.008",
     []),
    ("batch1", "03_fundus_methods", "Guo2020_bin_loss_exudates.pdf",
     "Bin loss for hard exudates", "10.1016/j.neucom.2018.10.049",
     []),
    ("batch1", "03_fundus_methods", "Huang2022_RTNet.pdf",
     "RTNet", "arXiv:2201.11037",
     ["https://arxiv.org/pdf/2201.11037.pdf"]),
    ("batch1", "03_fundus_methods", "Bo2022_SAA.pdf",
     "SAA scale-aware attention", "10.1109/ISBI52829.2022.9761674",
     []),
    ("batch1", "03_fundus_methods", "Liu2023_M2MRF.pdf",
     "M2MRF", "arXiv:2111.00193",
     ["https://arxiv.org/pdf/2111.00193.pdf"]),
    ("batch1", "03_fundus_methods", "Guo2024_deep_multiscale.pdf",
     "Deep multi-scale framework", "10.1016/j.bspc.2023.105050",
     []),
    ("batch1", "03_fundus_methods", "Xu2024_HACDR-Net.pdf",
     "HACDR-Net", "10.1609/aaai.v38i6.28453",
     ["https://ojs.aaai.org/index.php/AAAI/article/download/28453/28876",
      "https://ojs.aaai.org/index.php/AAAI/article/view/28453/28880",
      "https://cdn.aaai.org/ojs/28453/28453-13-32478-1-2-20240528.pdf"]),
    ("batch1", "03_fundus_methods", "Li2024_WSRFNet.pdf",
     "WSRFNet", "IJCAI2024",
     ["https://www.ijcai.org/proceedings/2024/0115.pdf"]),
    ("batch1", "03_fundus_methods", "Guo2022_CARNet.pdf",
     "CARNet", "10.1007/s40747-021-00630-4",
     ["https://link.springer.com/content/pdf/10.1007/s40747-021-00630-4.pdf"]),
    ("batch1", "03_fundus_methods", "Xiao2020_HEDNet_cGAN.pdf",
     "HEDNet cGAN adversarial lesion segmentation", "arXiv:2007.13854",
     ["https://arxiv.org/pdf/2007.13854.pdf"]),
    ("batch1", "03_fundus_methods", "Bian2024_MLNet.pdf",
     "MLNet", "10.3390/a17040164",
     ["https://www.mdpi.com/1999-4893/17/4/164/pdf"]),
    ("batch1", "03_fundus_methods", "Zhang2022_SS-MAF.pdf",
     "SS-MAF hard exudate", "10.1109/BIBM55620.2022.9995371",
     []),
    ("batch1", "03_fundus_methods", "Dai2018_multisieving_MA.pdf",
     "Multi-sieving MA detection", "10.1109/TMI.2017.2778741",
     []),
    ("batch1", "03_fundus_methods", "Liu2022_dual_branch_exudate.pdf",
     "Dual-branch hard exudate", "10.1109/JBHI.2021.3109302",
     []),

    # ----- batch 1: foundation -----
    ("batch1", "04_foundation_models", "Kirillov2023_SAM.pdf",
     "Segment Anything", "arXiv:2304.02643",
     ["https://arxiv.org/pdf/2304.02643.pdf"]),
    ("batch1", "04_foundation_models", "Ravi2024_SAM2.pdf",
     "SAM 2", "arXiv:2408.00714",
     ["https://arxiv.org/pdf/2408.00714.pdf"]),
    ("batch1", "04_foundation_models", "Ma2024_MedSAM.pdf",
     "MedSAM", "arXiv:2304.12306",
     ["https://arxiv.org/pdf/2304.12306.pdf"]),
    ("batch1", "04_foundation_models", "Li2024_TP-DRSeg.pdf",
     "TP-DRSeg", "arXiv:2406.15764",
     ["https://arxiv.org/pdf/2406.15764.pdf"]),
    ("batch1", "04_foundation_models", "ISBI2025_SAM2_DR_lesion.pdf",
     "Adapting Segment Anything 2 for DR lesion segmentation", "ISBI2025-SAM2-DR",
     []),

    # ----- batch 1: octa -----
    ("batch1", "05_octa", "Qian2023_DRAC.pdf",
     "DRAC challenge", "arXiv:2304.02389",
     ["https://arxiv.org/pdf/2304.02389.pdf"]),

    # ----- batch 1: general backbones -----
    ("batch1", "06_general_backbones", "Ronneberger2015_U-Net.pdf",
     "U-Net", "arXiv:1505.04597",
     ["https://arxiv.org/pdf/1505.04597.pdf"]),
    ("batch1", "06_general_backbones", "Chen2018_DeepLabv3plus.pdf",
     "DeepLabv3+", "arXiv:1802.02611",
     ["https://arxiv.org/pdf/1802.02611.pdf"]),
    ("batch1", "06_general_backbones", "Wang2020_HRNet.pdf",
     "HRNet", "arXiv:1908.07919",
     ["https://arxiv.org/pdf/1908.07919.pdf"]),
    ("batch1", "06_general_backbones", "Liu2021_Swin_Transformer.pdf",
     "Swin Transformer", "arXiv:2103.14030",
     ["https://arxiv.org/pdf/2103.14030.pdf"]),
    ("batch1", "06_general_backbones", "Long2015_FCN.pdf",
     "FCN", "arXiv:1411.4038",
     ["https://arxiv.org/pdf/1411.4038.pdf"]),
    ("batch1", "06_general_backbones", "Xiao2018_UPerNet.pdf",
     "UPerNet", "arXiv:1807.10221",
     ["https://arxiv.org/pdf/1807.10221.pdf"]),

    # ----- batch 2: fundus methods from snowball -----
    ("batch2", "03_fundus_methods", "He2022_PMCNet.pdf",
     "PMCNet", "arXiv:2205.15720",
     ["https://arxiv.org/pdf/2205.15720.pdf"]),
    ("batch2", "03_fundus_methods", "He2023_H2Former.pdf",
     "H2Former", "10.1109/TMI.2023.3264513",
     ["https://oar.a-star.edu.sg/storage/y/y3npzo5jmq/h2former.pdf"]),
    ("batch2", "03_fundus_methods", "Wang2023_VTA_hyperbolic.pdf",
     "VTA hyperbolic embeddings", "10.1038/s41598-023-38320-5",
     ["https://www.nature.com/articles/s41598-023-38320-5.pdf",
      "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10333307/pdf/41598_2023_Article_38320.pdf"]),
    ("batch2", "03_fundus_methods", "Ullah2023_SSMD-UNet.pdf",
     "SSMD-UNet", "10.1038/s41598-023-36311-0",
     ["https://www.nature.com/articles/s41598-023-36311-0.pdf"]),
    ("batch2", "03_fundus_methods", "Wang2023_CLC-Net.pdf",
     "CLC-Net", "10.1016/j.neucom.2023.01.013",
     []),
    ("batch2", "03_fundus_methods", "Zhang2023_TC-Net.pdf",
     "TC-Net", "10.1016/j.compbiomed.2023.106967",
     []),
    ("batch2", "03_fundus_methods", "Ali2023_LezioSeg.pdf",
     "LezioSeg", "10.3390/electronics12234940",
     ["https://www.mdpi.com/2079-9292/12/23/4940/pdf"]),
    ("batch2", "03_fundus_methods", "Zhou2020_collaborative_semi_supervised.pdf",
     "Collaborative learning of semi-supervised segmentation and classification", "CVPR2020",
     ["https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhou_Collaborative_Learning_of_Semi-Supervised_Segmentation_and_Classification_for_Medical_Images_CVPR_2020_paper.pdf"]),

    # ----- batch 2: foundation -----
    ("batch2", "04_foundation_models", "Zhang2023_SAMed.pdf",
     "SAMed", "arXiv:2304.13785",
     ["https://arxiv.org/pdf/2304.13785.pdf"]),
    ("batch2", "04_foundation_models", "Chen2023_SAM-Adapter.pdf",
     "SAM-Adapter", "arXiv:2304.09148",
     ["https://arxiv.org/pdf/2304.09148.pdf"]),
    ("batch2", "04_foundation_models", "Zhang2024_PerSAM.pdf",
     "PerSAM", "arXiv:2305.03048",
     ["https://arxiv.org/pdf/2305.03048.pdf"]),
    ("batch2", "04_foundation_models", "Qiu2023_Learnable_Ophthalmology_SAM.pdf",
     "Learnable Ophthalmology SAM", "arXiv:2304.13425",
     ["https://arxiv.org/pdf/2304.13425.pdf"]),
    ("batch2", "04_foundation_models", "Tragakis2023_FCT.pdf",
     "Fully Convolutional Transformer", "FCT-WACV2023",
     []),
    ("batch2", "04_foundation_models", "Xiong2024_SAM2-UNet.pdf",
     "SAM2-UNet", "arXiv:2408.08870",
     ["https://arxiv.org/pdf/2408.08870.pdf"]),

    # ----- batch 2: general backbones cited as DR baselines -----
    ("batch2", "06_general_backbones", "Cao2022_Swin-Unet.pdf",
     "Swin-Unet", "arXiv:2105.05537",
     ["https://arxiv.org/pdf/2105.05537.pdf"]),
    ("batch2", "06_general_backbones", "Xie2015_HED.pdf",
     "Holistically-nested edge detection", "arXiv:1504.06375",
     ["https://arxiv.org/pdf/1504.06375.pdf"]),
    ("batch2", "06_general_backbones", "Zhou2018_UNet++.pdf",
     "UNet++", "arXiv:1807.10165",
     ["https://arxiv.org/pdf/1807.10165.pdf"]),
    ("batch2", "06_general_backbones", "Badrinarayanan2017_SegNet.pdf",
     "SegNet", "arXiv:1511.00561",
     ["https://arxiv.org/pdf/1511.00561.pdf"]),
]


def fetch(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/pdf,*/*"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
        final = resp.geturl()
        ctype = resp.headers.get("Content-Type", "")
    return data, final, ctype


def looks_like_pdf(data):
    return len(data) > 8000 and data[:5] == b"%PDF-"


def unpaywall(doi):
    if not doi or doi.startswith("arXiv") or ":" in doi and not doi.startswith("10."):
        return []
    if not doi.startswith("10."):
        return []
    url = f"https://api.unpaywall.org/v2/{urllib.parse.quote(doi)}?email={EMAIL}"
    try:
        data, _, _ = fetch(url, timeout=30)
        obj = json.loads(data.decode("utf-8", "replace"))
    except Exception:
        return []
    urls = []
    best = obj.get("best_oa_location") or {}
    if best.get("url_for_pdf"):
        urls.append(best["url_for_pdf"])
    for loc in obj.get("oa_locations") or []:
        if loc.get("url_for_pdf"):
            urls.append(loc["url_for_pdf"])
    return urls


def semantic_scholar_doi(doi):
    if not doi or not doi.startswith("10."):
        return []
    url = (
        "https://api.semanticscholar.org/graph/v1/paper/DOI:"
        + urllib.parse.quote(doi)
        + "?fields=openAccessPdf,title"
    )
    try:
        data, _, _ = fetch(url, timeout=30)
        obj = json.loads(data.decode("utf-8", "replace"))
    except Exception:
        return []
    pdf = (obj.get("openAccessPdf") or {}).get("url")
    return [pdf] if pdf else []


def semantic_scholar_title(title):
    q = urllib.parse.quote(title)
    url = (
        "https://api.semanticscholar.org/graph/v1/paper/search?query="
        + q
        + "&limit=3&fields=title,openAccessPdf,externalIds"
    )
    try:
        data, _, _ = fetch(url, timeout=30)
        obj = json.loads(data.decode("utf-8", "replace"))
    except Exception:
        return []
    urls = []
    for paper in obj.get("data") or []:
        pdf = (paper.get("openAccessPdf") or {}).get("url")
        if pdf:
            urls.append(pdf)
        arxiv = ((paper.get("externalIds") or {}).get("ArXiv"))
        if arxiv:
            urls.append(f"https://arxiv.org/pdf/{arxiv}.pdf")
    return urls


def main():
    rows = []
    for batch, cat, name, title, ident, urls in ITEMS:
        dest_dir = ROOT / batch / cat
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / name
        if dest.exists() and dest.stat().st_size > 8000:
            rows.append((batch, cat, name, title, ident, "already", str(dest), dest.stat().st_size))
            print(f"SKIP {name}")
            continue
        candidates = list(urls)
        candidates.extend(unpaywall(ident))
        candidates.extend(semantic_scholar_doi(ident))
        if not candidates:
            candidates.extend(semantic_scholar_title(title))
        # de-dup
        seen = set()
        uniq = []
        for u in candidates:
            if u and u not in seen:
                seen.add(u)
                uniq.append(u)
        status = "missing"
        used = ""
        size = 0
        for u in uniq:
            try:
                data, final, ctype = fetch(u)
            except Exception as e:
                print(f"  fail {name} {u[:80]} :: {e}")
                time.sleep(0.3)
                continue
            if looks_like_pdf(data):
                dest.write_bytes(data)
                status = "ok"
                used = final
                size = len(data)
                print(f"OK {name} {size}")
                break
            else:
                print(f"  notpdf {name} ctype={ctype} len={len(data)} url={final[:90]}")
            time.sleep(0.2)
        if status != "ok":
            print(f"MISSING {name}")
        rows.append((batch, cat, name, title, ident, status, used, size))
        time.sleep(0.4)

    manifest = ROOT / "manifest.csv"
    with manifest.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["batch", "category", "file", "title", "id", "status", "url", "bytes"])
        w.writerows(rows)
    ok = sum(1 for r in rows if r[5] == "ok" or r[5] == "already")
    miss = sum(1 for r in rows if r[5] == "missing")
    print(f"DONE ok={ok} missing={miss} manifest={manifest}")


if __name__ == "__main__":
    main()
