"""Write the folder a clinician opens: the reader tool, the images, the outlines and heat maps, nothing else.

docs/clinical/28 section 6. Input is selection.json of sample_review.py (the key); the pack holds only what the
reader needs to see, so the source dataset, image id, lesion class and discrepancy kind of a region are not in
it (cases are c01, c02, ...; regions c01-1, c01-2, ...). The key stays with the study team.

    <pack>/index.html          the reader tool (clinical/reader_tool/index.html), opened by double-click
    <pack>/pack.js             window.RETISEG_PACK = {...}: schema, cases, outlines, regions (loaded by index.html)
    <pack>/pack.json           the same object, for the tool's "open folder" fallback
    <pack>/images/<case>.jpg   the photograph (original JPEG bytes when the source is JPEG and not resized)
    <pack>/heat/<case>_<CLS>.png  model probability as a class-coloured RGBA layer (reduced resolution)
    <pack>/READ_ME_FIRST.txt   how to open, save and return the results
    <pack>/SHA256SUMS.txt

Outlines are closed polygons on pixel corners (clinical/scripts/contours.py) of
    dataset annotation   the prepared mask of each class (bench/common/io.py), whole image
    model                probability >= the model's threshold, inside the FOV and outside the class's ignore region
                         (the same binary map the regions were mined from)
    ignore               the class's ignore region where the source has one (drawn dotted, labelled "uncertain")

    python clinical/scripts/build_pack.py --selection clinical/work/pilot01/selection.json \
        --out clinical/packs/pilot01 [--practice clinical/work/practice01/selection.json] [--max-side 0]
"""

import argparse
import io
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import contours  # noqa: E402
from common import CLASSES  # noqa: E402

HERE = Path(__file__).resolve().parent
TOOL = HERE.parent / "reader_tool" / "index.html"
SCHEMA = HERE.parent / "schema" / "review_schema_v1.json"

CLASS_STYLE = {
    "MA": {"name": "Microaneurysm", "color": "#22d3ee"},
    "HE": {"name": "Haemorrhage", "color": "#ff4fd8"},
    "EX": {"name": "Hard exudate", "color": "#ffe14d"},
    "SE": {"name": "Cotton-wool spot", "color": "#7dff6b"},
}


def hex_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def heat_png(prob, color, max_side):
    """Class-coloured RGBA image whose alpha grows with the probability; transparent below 0.05."""
    H, W = prob.shape
    f = max(1, int(np.ceil(max(H, W) / max_side)))
    if f > 1:
        Hc, Wc = H // f * f, W // f * f
        small = prob[:Hc, :Wc].reshape(Hc // f, f, Wc // f, f).max(axis=(1, 3))   # max-pool keeps small lesions
    else:
        small = prob
    a = np.clip((small - 0.05) / 0.95, 0, 1) ** 0.6 * 0.9
    rgba = np.zeros(small.shape + (4,), dtype=np.uint8)
    rgba[..., :3] = hex_rgb(color)
    rgba[..., 3] = (a * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, format="PNG", optimize=True)
    return buf.getvalue(), f


def write_image(src, dst, max_side):
    """Original bytes for an unresized JPEG; otherwise a quality-95 JPEG. Returns (width, height, scale)."""
    im = Image.open(src)
    W, H = im.size
    scale = 1.0
    if max_side and max(W, H) > max_side:
        scale = max_side / max(W, H)
    if scale == 1.0 and src.suffix.lower() in (".jpg", ".jpeg"):
        shutil.copyfile(src, dst)
        return W, H, 1.0
    im = im.convert("RGB")
    if scale != 1.0:
        im = im.resize((round(W * scale), round(H * scale)), Image.LANCZOS)
    im.save(dst, format="JPEG", quality=95, subsampling=0)
    return im.size[0], im.size[1], scale


def loops_of(mask, scale, eps):
    return [contours.flat(lp) for lp in contours.mask_loops(mask, eps=eps, scale=scale)]


def build_case(case, insts, sets, root, out, args, practice=False):
    s = sets[case["set"]]
    meta = s["meta"]
    data_dir = root / meta["data"]
    pred_dir = root / meta["pred"]
    thresholds = meta["thresholds"]
    image_id = case["image_id"]
    cid = case["case_id"]
    src = common.bio.image_path(data_dir, image_id)
    width, height, scale = write_image(src, out / "images" / f"{cid}.jpg", args.max_side)
    fov, gt, ignore = common.read_case(data_dir, image_id)
    overlays = {"annotation": {}, "model": {}, "uncertain": {}}
    heat = {}
    for c in CLASSES:
        prob = common.read_prob(pred_dir, c, image_id)
        valid = fov & ~ignore[c]
        overlays["annotation"][c] = loops_of(gt[c], scale, args.eps)
        overlays["model"][c] = loops_of((prob >= thresholds[c]) & valid, scale, args.eps)
        if ignore[c].any():
            overlays["uncertain"][c] = loops_of(ignore[c], scale, args.eps)
        png, f = heat_png(np.where(fov, prob, 0), CLASS_STYLE[c]["color"], args.heat_max_side)
        (out / "heat" / f"{cid}_{c}.png").write_bytes(png)
        heat[c] = {"src": f"heat/{cid}_{c}.png"}

    regions = []
    for x in insts:
        m = x["marker"]
        regions.append({
            "id": x["inst_id"],
            "marker": {"cx": round(m["cx"] * scale, 1), "cy": round(m["cy"] * scale, 1), "r": round(m["r"] * scale, 1)},
            "crop": [round(v * scale) for v in x["crop"]],
        })
    entry = {"id": cid, "image": f"images/{cid}.jpg", "width": width, "height": height,
             "overlays": overlays, "heat": heat, "regions": regions, "practice": practice}
    scale_name = s.get("reference_grade_scale")
    if scale_name and case.get("grade") is not None:
        entry["reference_grade"] = {"value": case["grade"], "scale": scale_name}
    return entry


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selection", required=True)
    ap.add_argument("--practice", help="a second selection.json whose cases come first as practice (not analysed)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--root", default=".", help="directory the selection's relative data / prediction paths are read from")
    ap.add_argument("--title", help="pack title shown in the tool")
    ap.add_argument("--max-side", type=int, default=0, help="downscale photographs whose long side exceeds this (0 = keep)")
    ap.add_argument("--heat-max-side", type=int, default=1600)
    ap.add_argument("--eps", type=float, default=0.7, help="outline simplification tolerance in pixels")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    root = Path(args.root)
    sel = json.loads(Path(args.selection).read_text())
    out = Path(args.out)
    if out.exists():
        if not args.force:
            raise SystemExit(f"{out} exists; pass --force to rebuild it")
        shutil.rmtree(out)
    (out / "images").mkdir(parents=True)
    (out / "heat").mkdir()

    cases = []
    jobs = []
    if args.practice:
        psel = json.loads(Path(args.practice).read_text())
        for i, case in enumerate(psel["cases"], 1):
            pid = f"p{i:02d}"
            insts = [dict(x, inst_id=x["inst_id"].replace(case["case_id"], pid, 1)) for x in psel["instances"] if x["case_id"] == case["case_id"]]
            jobs.append((dict(case, case_id=pid), insts, psel["sets"], True))
    for case in sel["cases"]:
        jobs.append((case, [x for x in sel["instances"] if x["case_id"] == case["case_id"]], sel["sets"], False))
    t0 = time.time()
    for n, (case, insts, sets, practice) in enumerate(jobs, 1):
        cases.append(build_case(case, insts, sets, root, out, args, practice=practice))
        print(f"[pack] {n}/{len(jobs)} {case['case_id']} ({len(insts)} regions) {time.time() - t0:.0f} s", flush=True)

    schema = json.loads(SCHEMA.read_text())
    pack = {
        "pack_id": sel["design"]["name"] + ("+practice" if args.practice else ""),
        "title": args.title or f"RetiSeg clinical review - {sel['design']['name']}",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "schema": schema,
        "classes": CLASS_STYLE,
        "settings": {"shuffle_cases": True, "shuffle_regions": True, "practice_first": True},
        "cases": cases,
    }
    text = json.dumps(pack, separators=(",", ":"), ensure_ascii=False)
    (out / "pack.json").write_text(text)
    (out / "pack.js").write_text("window.RETISEG_PACK = " + text + ";\n")
    shutil.copyfile(TOOL, out / "index.html")
    (out / "READ_ME_FIRST.txt").write_text(README.format(title=pack["title"], n_cases=len(cases),
                                                         n_regions=sum(len(c["regions"]) for c in cases)))
    sums = []
    for p in sorted(out.rglob("*")):
        if p.is_file() and p.name != "SHA256SUMS.txt":
            sums.append(f"{common.sha256_file(p)}  {p.relative_to(out)}")
    (out / "SHA256SUMS.txt").write_text("\n".join(sums) + "\n")
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(json.dumps({"pack": str(out), "cases": len(cases), "regions": sum(len(c["regions"]) for c in cases),
                      "megabytes": round(size / 1e6, 1)}))


README = """{title}

{n_cases} images, {n_regions} marked regions.

How to open
  1. Keep this folder together (index.html, pack.js, images/, heat/).
  2. Double-click index.html. Chrome, Edge or Firefox are recommended; a laptop screen of 13 inches or more.
     If the page says it cannot find the pack, click "Open pack folder" and choose this folder.
  3. Enter your reader code (given to you by the study team) and follow the steps on the right.

Saving
  Every answer is saved automatically in this browser on this computer. Use "Export results" at the end of
  each session and send the downloaded .json file to the study team. In Chrome or Edge you can also link a
  file once ("Save to file...") so the tool keeps writing to it.
  If you move to another computer, import your last exported file there before continuing.

Please do not share this folder outside the study: some images are licensed for research use by the
study team only. Delete it when the study team tells you the review is closed.
"""


if __name__ == "__main__":
    main()
