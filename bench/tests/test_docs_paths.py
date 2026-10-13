"""Repository paths cited in the live docs exist and are tracked (docs/ops/21 open issue 34 a).

Review B6 found tracked documents citing untracked notes. The live docs are the ones a reader
follows today: CHECKPOINT, the index, 00, 17, 18, 21 and 16. Historical records (19, 20, 22,
the archive) are not checked. Paths under docs/notes(albert)/ are Albert's private notes and are
cited as such; paths listed in PRODUCED_LATER are written by a later step. A path inside a
submodule (official_code/<name>/...) is checked only when that submodule is checked out, so a
checkout without submodules, or a CI job, is not failed by it (review N4).
"""
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LIVE_DOCS = [
    "docs/CHECKPOINT.md",
    "docs/README.txt",
    "docs/ops/00_进度.txt",
    "docs/ops/17_服务器任务单.txt",
    "docs/design/18_E5推理档实现.txt",
    "docs/ops/21_现存问题.txt",
    "docs/paper/16_范围精简.txt",
]
PRIVATE = ("docs/notes(albert)/",)
PRODUCED_LATER = {
    "docs/paper/retiseg_paper/paper/data/cost_audit.tsv",  # written by bench/scripts/cost_audit.py on the server (17 section 1)
}
PATH = re.compile(r"(?<![\w/])((?:docs|bench|literature|official_code)/[\w\-./()一-鿿]+?\.(?:txt|md|py|yaml|yml|json|sh|tsv|tex|csv))")


def cited_paths(text):
    return sorted({m.group(1).rstrip(".") for m in PATH.finditer(text) if "..." not in m.group(1)})


def submodule_checked_out(path):
    """False for an official_code/<name>/... path whose submodule directory is absent or empty."""
    parts = Path(path).parts
    if parts[0] != "official_code" or len(parts) < 2:
        return True
    root = REPO / parts[0] / parts[1]
    return root.is_dir() and any(root.iterdir())


def test_live_docs_cite_existing_tracked_paths():
    tracked = set(subprocess.run(["git", "-c", "core.quotepath=false", "ls-files"], cwd=REPO, capture_output=True, text=True, check=True).stdout.split("\n"))
    problems = []
    for name in LIVE_DOCS:
        for path in cited_paths((REPO / name).read_text()):
            if path.startswith(PRIVATE) or path in PRODUCED_LATER or not submodule_checked_out(path):
                continue
            if not (REPO / path).exists():
                problems.append(f"{name}: {path} does not exist")
            elif not path.startswith("official_code/") and path not in tracked:
                problems.append(f"{name}: {path} is not tracked")
    assert not problems, "\n".join(problems)
