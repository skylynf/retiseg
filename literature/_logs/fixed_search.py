#!/usr/bin/env python3
"""Locked search for the DR lesion-segmentation assessment.

The query is fixed in QUERY_* below and in literature/synthesis/06_检索.txt.
This script downloads PubMed, Europe PMC and OpenAlex, applies one keyword
gate, and writes a deduplicated screening table. It does not edit the ledger.
"""

import csv
import json
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "_logs" / "search"
OUT.mkdir(parents=True, exist_ok=True)
UA = "retiseg-search/0.1 (academic literature search; mailto:research@localhost)"
MAILTO = "research@localhost"

PUBMED_TERM = (
    '("diabetic retinopathy"[Title/Abstract] OR "diabetic retinal"[Title/Abstract])'
    " AND (lesion[Title/Abstract] OR lesions[Title/Abstract]"
    " OR microaneurysm[Title/Abstract] OR microaneurysms[Title/Abstract]"
    " OR exudate[Title/Abstract] OR exudates[Title/Abstract]"
    " OR hemorrhage[Title/Abstract] OR hemorrhages[Title/Abstract]"
    " OR haemorrhage[Title/Abstract] OR haemorrhages[Title/Abstract]"
    ' OR "cotton wool"[Title/Abstract] OR "cotton-wool"[Title/Abstract])'
    " AND (segmentation[Title/Abstract] OR segmenting[Title/Abstract] OR segmented[Title/Abstract])"
    ' AND ("2015/01/01"[Date - Publication] : "2026/09/30"[Date - Publication])'
)

EPMC_QUERY = (
    '(TITLE_ABS:"diabetic retinopathy" OR TITLE_ABS:"diabetic retinal")'
    " AND (TITLE_ABS:lesion OR TITLE_ABS:lesions OR TITLE_ABS:microaneurysm OR TITLE_ABS:microaneurysms"
    " OR TITLE_ABS:exudate OR TITLE_ABS:exudates OR TITLE_ABS:hemorrhage OR TITLE_ABS:hemorrhages"
    " OR TITLE_ABS:haemorrhage OR TITLE_ABS:haemorrhages OR TITLE_ABS:\"cotton wool\" OR TITLE_ABS:\"cotton-wool\")"
    " AND (TITLE_ABS:segmentation OR TITLE_ABS:segmenting OR TITLE_ABS:segmented)"
    " AND (FIRST_PDATE:[2015-01-01 TO 2026-09-30])"
)

# OpenAlex title_and_abstract.search is a relevance search, not this Boolean.
# Every OpenAlex hit is passed through the same keyword gate before screening.
OPENALEX_SEARCH = "diabetic retinopathy lesion segmentation"


def get(url, timeout=90):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def get_json(url):
    return json.loads(get(url).decode("utf-8", "replace"))


def norm_doi(value):
    if not value:
        return ""
    text = value.strip().lower()
    text = re.sub(r"^https?://(dx\.)?doi\.org/", "", text)
    text = text.replace("doi:", "")
    return text.strip()


def norm_title(value):
    text = (value or "").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def keyword_gate(title, abstract):
    text = f"{title or ''} {abstract or ''}".lower()
    dr = ("diabetic retinopathy" in text) or ("diabetic retinal" in text)
    lesion = any(
        token in text
        for token in (
            "lesion",
            "microaneurysm",
            "exudate",
            "hemorrhage",
            "haemorrhage",
            "cotton wool",
            "cotton-wool",
        )
    )
    seg = any(token in text for token in ("segmentation", "segmenting", "segmented"))
    return dr and lesion and seg


def load_corpus():
    dois, arxivs, titles = set(), set(), {}
    aliases = []
    for path in (ROOT / "meta").rglob("*.txt"):
        ident = title = ""
        for line in path.read_text(errors="replace").splitlines():
            if line.startswith("标识:"):
                ident = line.split(":", 1)[1].strip()
            elif line.startswith("题名:"):
                title = line.split(":", 1)[1].strip()
        doi = ""
        ax = ""
        low = ident.lower()
        if "10." in ident:
            m = re.search(r"10\.\S+", ident)
            if m:
                doi = norm_doi(m.group(0).rstrip(".;,)"))
                dois.add(doi)
        if "arxiv:" in low:
            m = re.search(r"arxiv:(\d{4}\.\d{4,5})", low)
            if m:
                ax = m.group(1)
                arxivs.add(ax)
        nt = norm_title(title)
        if len(nt) >= 40:
            titles[nt] = path.name
        if title and len(title) <= 40:
            aliases.append(title)
    # Distinctive method names already in the ledger. Used only as a second
    # check so a hit is not listed as new when the meta title is a short name.
    aliases = [
        "l-seg", "lseg", "rtnet", "m2mrf", "hacdr", "wsrfnet", "carnet",
        "hednet", "mlnet", "pmcnet", "h2former", "ssmd-unet", "ssmd unet",
        "lezioseg", "clc-net", "clcnet", "tc-net", "tp-drseg", "bin loss",
        "idrid", "diaretdb", "e-ophtha", "eophtha", "fgadr", "maples-dr",
        "messidor", "sam-adapter", "samed", "persam", "medsam", "swin-unet",
        "unet++", "deeplab", "hrnet",
    ]
    return dois, arxivs, titles, aliases


def corpus_hit(rec, dois, arxivs, titles, aliases):
    doi = norm_doi(rec.get("doi", ""))
    if doi and doi in dois:
        return "doi"
    ax = rec.get("arxiv", "")
    if ax and ax in arxivs:
        return "arxiv"
    nt = norm_title(rec.get("title", ""))
    if nt and nt in titles:
        return "title"
    low = nt
    for alias in aliases:
        if alias in low and len(alias) >= 5:
            # Short aliases only count when the rest of the title is about this task.
            if alias in ("idrid", "hrnet", "deeplab", "medsam", "samed", "persam"):
                continue
            return "alias:" + alias
    return ""


def screen(title, abstract):
    text = f"{title or ''} {abstract or ''}".lower()
    title_l = (title or "").lower()
    if re.search(r"\b(survey|systematic review|scoping review|literature review)\b", title_l) or title_l.startswith("review"):
        return "review", "题名是综述"
    octa = any(k in text for k in ("octa", "optical coherence", "ultra-widefield octa", "swept-source"))
    fundus = any(k in text for k in ("fundus", "colour fundus", "color fundus", "retinal image", "eye fundus"))
    if octa and not fundus:
        return "oct_octa", "只有 OCT 或 OCTA，没有眼底照片"
    anatomy = any(k in text for k in ("optic disc", "optic cup", "vessel segmentation", "blood vessel", "retinal vessel"))
    lesionish = any(k in text for k in ("microaneurysm", "exudate", "cotton wool", "cotton-wool", "lesion"))
    if anatomy and not lesionish and "hemorrhage" not in text and "haemorrhage" not in text:
        return "anatomy", "血管或视盘，没有病灶分割"
    classes = set()
    if "microaneurysm" in text:
        classes.add("MA")
    if "hemorrhage" in text or "haemorrhage" in text:
        classes.add("HE")
    if "exudate" in text:
        classes.add("EX")
    if "cotton wool" in text or "cotton-wool" in text or "soft exudate" in text:
        classes.add("SE")
    multi = any(
        k in text
        for k in (
            "multi-lesion",
            "multiple lesion",
            "multi lesion",
            "four lesion",
            "four types",
            "lesion types",
        )
    )
    grading = any(k in text for k in ("grading", "classification", "severity", "screening"))
    dataset = any(k in title_l for k in ("dataset", "database", "benchmark"))
    if dataset and (classes or "lesion" in text):
        return "dataset", "题名是数据集或基准，" + ",".join(sorted(classes) or ["lesion"])
    if len(classes) >= 2 or multi:
        return "candidate_primary", "题名或摘要提到至少两类或多种病灶：" + ",".join(sorted(classes) or ["multi"])
    if len(classes) == 1:
        return "candidate_single", "只点名一类：" + next(iter(classes))
    if grading and "segmentation" not in text:
        return "grading", "分级或筛查，摘要没有分割"
    if "lesion" in text and any(k in text for k in ("segmentation", "segmenting", "segmented")):
        return "uncertain", "写了病灶分割，但没有点明类别"
    return "uncertain", "关键词通过，类别和任务不够明确"


def fetch_pubmed():
    url = (
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&retmode=json&retmax=10000&term="
        + urllib.parse.quote(PUBMED_TERM)
    )
    ids = get_json(url)["esearchresult"]["idlist"]
    records = []
    for i in range(0, len(ids), 100):
        chunk = ids[i : i + 100]
        fetch = (
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&retmode=xml&id="
            + ",".join(chunk)
        )
        root = ET.fromstring(get(fetch))
        for art in root.findall(".//PubmedArticle"):
            pmid = art.findtext(".//PMID") or ""
            title = "".join(art.find(".//ArticleTitle").itertext()) if art.find(".//ArticleTitle") is not None else ""
            abstract = " ".join("".join(node.itertext()) for node in art.findall(".//AbstractText"))
            year = art.findtext(".//PubDate/Year") or art.findtext(".//PubDate/MedlineDate") or ""
            doi = ""
            for aid in art.findall(".//ArticleId"):
                if aid.get("IdType") == "doi":
                    doi = aid.text or ""
            journal = art.findtext(".//Journal/Title") or ""
            records.append(
                {
                    "source": "pubmed",
                    "pmid": pmid,
                    "doi": doi,
                    "arxiv": "",
                    "title": title,
                    "abstract": abstract,
                    "year": year[:4],
                    "venue": journal,
                }
            )
        time.sleep(0.4)
    return records


def fetch_epmc():
    records = []
    cursor = "*"
    while True:
        url = (
            "https://www.ebi.ac.uk/europepmc/webservices/rest/search?format=json&pageSize=1000"
            f"&cursorMark={urllib.parse.quote(cursor)}&query=" + urllib.parse.quote(EPMC_QUERY)
        )
        obj = get_json(url)
        for hit in obj.get("resultList", {}).get("result", []):
            doi = hit.get("doi") or ""
            ax = ""
            if doi.startswith("10.48550/arxiv."):
                ax = doi.split("arxiv.")[-1]
            records.append(
                {
                    "source": "epmc",
                    "pmid": hit.get("pmid") or "",
                    "doi": doi,
                    "arxiv": ax,
                    "title": hit.get("title") or "",
                    "abstract": hit.get("abstractText") or "",
                    "year": str(hit.get("pubYear") or ""),
                    "venue": hit.get("journalTitle") or hit.get("bookOrReportDetails") or "",
                }
            )
        nxt = obj.get("nextCursorMark")
        if not nxt or nxt == cursor or not obj.get("resultList", {}).get("result"):
            break
        cursor = nxt
        time.sleep(0.3)
    return records


def reconstruct_abstract(index):
    if not index:
        return ""
    pairs = []
    for word, positions in index.items():
        for pos in positions:
            pairs.append((pos, word))
    pairs.sort()
    return " ".join(word for _, word in pairs)


def fetch_openalex():
    records = []
    cursor = "*"
    filt = urllib.parse.quote(
        "from_publication_date:2015-01-01,to_publication_date:2026-09-30,"
        f"title_and_abstract.search:{OPENALEX_SEARCH}",
        safe=":,",
    )
    while True:
        url = (
            "https://api.openalex.org/works?per-page=200"
            f"&cursor={urllib.parse.quote(cursor)}&filter={filt}"
            "&select=id,doi,title,publication_year,publication_date,primary_location,abstract_inverted_index,ids,type"
            f"&mailto={MAILTO}"
        )
        obj = get_json(url)
        works = obj.get("results") or []
        if not works:
            break
        for work in works:
            ids = work.get("ids") or {}
            doi = work.get("doi") or ids.get("doi") or ""
            pmid = ids.get("pmid") or ""
            pmid = pmid.rsplit("/", 1)[-1] if pmid else ""
            ax = ""
            if "10.48550/arxiv." in doi.lower():
                ax = doi.lower().split("arxiv.")[-1]
            loc = (work.get("primary_location") or {}).get("source") or {}
            records.append(
                {
                    "source": "openalex",
                    "pmid": pmid,
                    "doi": doi,
                    "arxiv": ax,
                    "title": work.get("title") or "",
                    "abstract": reconstruct_abstract(work.get("abstract_inverted_index")),
                    "year": str(work.get("publication_year") or ""),
                    "venue": loc.get("display_name") or "",
                    "date": work.get("publication_date") or "",
                    "type": work.get("type") or "",
                }
            )
        cursor = (obj.get("meta") or {}).get("next_cursor")
        if not cursor:
            break
        time.sleep(0.2)
    return records


def dedup_key(rec):
    doi = norm_doi(rec.get("doi", ""))
    if doi:
        return "doi:" + doi
    if rec.get("pmid"):
        return "pmid:" + rec["pmid"]
    if rec.get("arxiv"):
        return "arxiv:" + rec["arxiv"]
    return "title:" + norm_title(rec.get("title", ""))


def main():
    batches = {
        "pubmed": fetch_pubmed(),
        "epmc": fetch_epmc(),
        "openalex": fetch_openalex(),
    }
    counts = {}
    for name, rows in batches.items():
        path = OUT / f"{name}.jsonl"
        with path.open("w") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        counts[name] = len(rows)
        print(name, len(rows))

    dois, arxivs, titles, aliases = load_corpus()
    merged = {}
    keyword_fail = 0
    after_freeze = 0
    for name, rows in batches.items():
        for rec in rows:
            date = rec.get("date") or ""
            if date and date > "2026-09-30":
                after_freeze += 1
                continue
            if not keyword_gate(rec.get("title", ""), rec.get("abstract", "")):
                keyword_fail += 1
                continue
            key = dedup_key(rec)
            slot = merged.setdefault(key, {"sources": set(), "rec": rec})
            slot["sources"].add(name)
            # Prefer a record that has an abstract.
            if len(rec.get("abstract") or "") > len(slot["rec"].get("abstract") or ""):
                slot["rec"] = rec
    print("unique_after_gate", len(merged), "keyword_fail_rows", keyword_fail, "after_freeze", after_freeze)

    out_rows = []
    for key, slot in merged.items():
        rec = slot["rec"]
        how = corpus_hit(rec, dois, arxivs, titles, aliases)
        if how:
            decision, reason = "already_in_corpus", how
        else:
            decision, reason = screen(rec.get("title", ""), rec.get("abstract", ""))
        out_rows.append(
            {
                "decision": decision,
                "reason": reason,
                "sources": ",".join(sorted(slot["sources"])),
                "year": rec.get("year", ""),
                "doi": norm_doi(rec.get("doi", "")),
                "pmid": rec.get("pmid", ""),
                "venue": rec.get("venue", ""),
                "title": re.sub(r"\s+", " ", rec.get("title", "")).strip(),
                "abstract": re.sub(r"\s+", " ", rec.get("abstract", "")).strip()[:1500],
            }
        )
    out_rows.sort(key=lambda r: (r["decision"], r["year"], r["title"]))
    dest = OUT / "screened.tsv"
    with dest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(out_rows)
    from collections import Counter

    print(Counter(r["decision"] for r in out_rows))
    print("wrote", dest)


if __name__ == "__main__":
    main()
