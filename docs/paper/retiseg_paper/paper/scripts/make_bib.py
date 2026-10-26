"""Write paper/refs.bib from literature/meta/**/*.txt plus a short list of extra entries.

Key = the meta file name in lower case without separators, for example
literature/meta/02_datasets/Porwal2018_IDRiD.txt -> porwal2018idrid.

The meta records hold a shortened author list ("First Author et al."), so
every generated entry carries a note field "authors: complete from the PDF".
Complete the author lists before submission; nothing here invents a name.

    python paper/scripts/make_bib.py
"""

import re
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1]

def _find_repo(start):
    """Nearest ancestor holding bench/configs/models.yaml (the paper lives under docs/paper/)."""
    for p in [start, *start.parents]:
        if (p / "bench" / "configs" / "models.yaml").exists():
            return p
    return start.parent

REPO = _find_repo(PAPER)
META = REPO / "literature" / "meta"

# Works cited in the text that are not in the ledger. nnU-Net Revisited (LNCS 15009, pp. 488-498, DOI) and SAM 3
# (arXiv:2511.16719, author list) were checked against Crossref / arXiv on 2026-10-26; RETFound's author list is the
# first eleven of the Nature paper (the consortium follows).
EXTRA = r"""
@article{isensee2021nnunet,
  author  = {Isensee, Fabian and Jaeger, Paul F. and Kohl, Simon A. A. and Petersen, Jens and Maier-Hein, Klaus H.},
  title   = {{nnU-Net}: a self-configuring method for deep learning-based biomedical image segmentation},
  journal = {Nature Methods}, volume = {18}, pages = {203--211}, year = {2021}, doi = {10.1038/s41592-020-01008-z}
}
@inproceedings{isensee2024revisited,
  author    = {Isensee, Fabian and Wald, Tassilo and Ulrich, Constantin and Baumgartner, Michael and Roy, Saikat and Maier-Hein, Klaus and J{\"a}ger, Paul F.},
  title     = {{nnU-Net} Revisited: A Call for Rigorous Validation in {3D} Medical Image Segmentation},
  booktitle = {Medical Image Computing and Computer Assisted Intervention -- MICCAI 2024},
  series    = {Lecture Notes in Computer Science}, volume = {15009}, pages = {488--498},
  publisher = {Springer}, address = {Cham}, year = {2024}, doi = {10.1007/978-3-031-72114-4_47}
}
@article{zhou2023retfound,
  author  = {Zhou, Yukun and Chia, Mark A. and Wagner, Siegfried K. and Ayhan, Murat S. and Williamson, Dominic J. and Struyven, Robbert R. and Liu, Timing and Xu, Moucheng and Lozano, Mateo G. and Woodward-Court, Peter and Kihara, Yuka and others},
  title   = {A foundation model for generalizable disease detection from retinal images},
  journal = {Nature}, volume = {622}, pages = {156--163}, year = {2023}, doi = {10.1038/s41586-023-06555-x}
}
@article{maierhein2018rankings,
  author  = {Maier-Hein, Lena and others},
  title   = {Why rankings of biomedical image analysis competitions should be interpreted with care},
  journal = {Nature Communications}, volume = {9}, pages = {5217}, year = {2018}, doi = {10.1038/s41467-018-07619-7},
  annote  = {authors: complete from the PDF}
}
@article{demsar2006statistical,
  author  = {Dem{\v{s}}ar, Janez},
  title   = {Statistical comparisons of classifiers over multiple data sets},
  journal = {Journal of Machine Learning Research}, volume = {7}, pages = {1--30}, year = {2006}
}
@article{oquab2024dinov2,
  author  = {Oquab, Maxime and others},
  title   = {{DINOv2}: Learning robust visual features without supervision},
  journal = {Transactions on Machine Learning Research}, year = {2024},
  annote  = {authors: complete from the PDF}
}
@misc{chen2017deeplabv3,
  author = {Chen, Liang-Chieh and Papandreou, George and Schroff, Florian and Adam, Hartwig},
  title  = {Rethinking Atrous Convolution for Semantic Image Segmentation},
  year   = {2017}, howpublished = {arXiv:1706.05587}
}
@misc{carion2025sam3,
  author = {Carion, Nicolas and Gustafson, Laura and Hu, Yuan-Ting and Debnath, Shoubhik and Hu, Ronghang and Suris, Didac and Ryali, Chaitanya and Alwala, Kalyan Vasudev and Khedr, Haitham and Huang, Andrew and others},
  title  = {{SAM 3}: Segment Anything with Concepts},
  year   = {2025}, howpublished = {arXiv:2511.16719}, eprint = {2511.16719}
}
"""


def parse(path):
    rec = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            rec.setdefault(k.strip(), v.strip())
    return rec


def bib_author(text):
    text = text.replace(" et al.", "").replace("等", "").strip()
    names = [n.strip() for n in re.split(r",|，| and ", text) if n.strip()]
    out = []
    for n in names:
        parts = n.split()
        out.append(f"{parts[-1]}, {' '.join(parts[:-1])}" if len(parts) > 1 else n)
    return " and ".join(out), ("et al." in text or len(names) == 1)


def entry(key, rec, truncated_default):
    title = rec.get("题名", key)
    author_raw = rec.get("作者", "")
    author, _ = bib_author(author_raw)
    truncated = "et al" in author_raw or truncated_default
    if truncated:
        author = author + " and others" if author else "others"
    year = re.search(r"\d{4}", rec.get("年份", "") or "")
    venue = rec.get("出处", "")
    ident = rec.get("标识", "")
    fields = [f"  author = {{{author}}}", f"  title = {{{{{title}}}}}", f"  year = {{{year.group(0) if year else ''}}}"]
    kind = "article"
    if "arXiv" in venue or (not venue and "arXiv" in ident):
        kind = "misc"
        fields.append(f"  howpublished = {{{venue or ident}}}")
    elif venue:
        fields.append(f"  journal = {{{venue}}}")
    doi = re.search(r"doi:\s*(\S+)", ident)
    if doi:
        fields.append(f"  doi = {{{doi.group(1)}}}")
    arx = re.search(r"arXiv:\s*(\S+)", ident)
    if arx:
        fields.append(f"  eprint = {{{arx.group(1)}}}")
    if truncated:
        fields.append("  annote = {authors: complete from the PDF}")
    return f"@{kind}{{{key},\n" + ",\n".join(fields) + "\n}\n"


def main():
    out, keys = [], []
    for path in sorted(META.rglob("*.txt")):
        if "00_template" in path.parts and path.stem.endswith("_SI"):
            continue
        rec = parse(path)
        key = re.sub(r"[^a-z0-9]", "", path.stem.lower())
        keys.append(key)
        out.append(entry(key, rec, truncated_default=False))
    text = "% Generated by scripts/make_bib.py from literature/meta. Do not edit by hand.\n\n" + "\n".join(out) + EXTRA
    (PAPER / "refs.bib").write_text(text, encoding="utf-8")
    print(f"wrote refs.bib with {len(keys)} ledger entries and {EXTRA.count('@')} extra entries")
    print(" ".join(keys))


if __name__ == "__main__":
    main()
