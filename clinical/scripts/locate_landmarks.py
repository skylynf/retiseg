"""Optic disc and fovea positions for every image of a test set, for sampling by ETDRS zone (docs/clinical/32).

The formal batch oversamples regions near the fovea (hard exudates within 1 DD, the central subfield), so the
sampler needs a fovea and a scale before anyone has read the images. This script estimates both, writes a QC page on
which a person can correct the misses with two clicks, and merges the corrections. The estimates only decide which
stratum a region falls in; the analysis uses each reader's own fovea and disc marks, so a wrong estimate costs
sampling efficiency, never validity.

Estimate (classical image processing; no model): the image is reduced so the field of view is 512 px across and
divided by its own large-scale background (illumination). The disc is the strongest bright blob at disc scale
(difference of Gaussians, inside the eroded FOV). The fovea is the darkest smoothed point, weighed against a soft
prior on distance and angle, in a band 0.7-1.35 times the expected fovea-to-disc distance from the disc, from 16
degrees above to 28 degrees below the horizontal toward the FOV center (2026-11-07; the first version searched
within 35 degrees and on DDR often took the dark rim of the field, docs/clinical/34). Disc
radius and fovea-to-disc distance scale with the camera's field angle (--field-deg; 45 for most fundus cameras, 50
for IDRiD, about 133 for the ultra-widefield TJDR images). On the four IDRiD trial images the estimates were
80-500 um from the reader's fovea and 185-310 um from the reader's disc marks.
Scale: um_per_px = fovea_disc_um / |fovea - disc| (4,700 um, the adult mean of about 4.6-4.8 mm, as in the reader tool).

    python clinical/scripts/locate_landmarks.py estimate --set DDR --data dataset/prepared/DDR --field-deg 45 \
        --out runs/CLINREV/landmarks/DDR                       # landmarks.json + qc.html
    # open qc.html, click the misses (click = fovea, shift-click = disc), "Export corrections" -> corrections.json
    python clinical/scripts/locate_landmarks.py apply --landmarks runs/CLINREV/landmarks/DDR/landmarks.json \
        --corrections corrections.json                          # merged in place; corrected entries say source=manual
A corrections file may give each entry its own "source" (batch01: clinical/landmarks/batch01/, docs/clinical/34);
the automatic estimate of a corrected image is kept under "estimate". Running apply twice with the same file gives
the same landmarks.
"""

import argparse
import base64
import io
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

FOVEA_DISC_UM = 4700.0
WORK_DIAMETER = 512.0


def estimate(image, fov, field_deg=45.0):
    """{"disc": [x, y], "fovea": [x, y], "um_per_px", "confidence"} in native pixels."""
    H, W = fov.shape
    D = 2.0 * math.sqrt(fov.sum() / math.pi)
    s = WORK_DIAMETER / D
    w, h = max(8, round(W * s)), max(8, round(H * s))
    a = np.asarray(image.convert("RGB").resize((w, h), Image.BILINEAR)).astype(np.float32)
    f = np.asarray(Image.fromarray(fov.astype(np.uint8) * 255).resize((w, h), Image.NEAREST)) > 0
    d = WORK_DIAMETER
    k = 45.0 / float(field_deg)
    r_disc = 0.065 * d * k
    lum = 0.5 * a[..., 0] + 0.5 * a[..., 1]
    lum = np.where(f, lum, lum[f].mean() if f.any() else lum.mean())
    norm = lum / np.maximum(ndimage.gaussian_filter(lum, 0.12 * d), 1.0)
    inner = ndimage.binary_erosion(f, iterations=max(1, int(0.04 * d)))
    dog = ndimage.gaussian_filter(norm, r_disc * 0.5) - ndimage.gaussian_filter(norm, r_disc * 2.0)
    dog_m = np.where(inner, dog, -np.inf)
    dy, dx = np.unravel_index(int(np.argmax(dog_m)), dog_m.shape)
    disc_z = float((dog[dy, dx] - dog[inner].mean()) / (dog[inner].std() + 1e-6))

    yy, xx = np.mgrid[:h, :w]
    dist = np.hypot(xx - dx, yy - dy)
    fd = 0.40 * d * k
    sm = ndimage.gaussian_filter(norm, 0.035 * d)
    # 2026-11-07 (docs/clinical/34): the fovea lies temporal to the disc, about 2.5 disc diameters away and a little
    # below it. The first version searched a band within 35 degrees of the horizontal and took its darkest point; on
    # DDR that point was often the dark rim of the field of view above or below the macula. The search is now limited
    # to -16..+28 degrees (positive = below the disc), keeps 0.06 FOV diameters from the rim, and the darkness is
    # weighed against a soft prior on distance and angle.
    cy, cx = ndimage.center_of_mass(f)
    away = ndimage.binary_erosion(f, iterations=max(1, int(0.06 * d)))
    sides = [1.0, -1.0] if abs(cx - dx) <= 0.05 * d else ([1.0] if cx > dx else [-1.0])
    best = None
    for sgn in sides:
        horiz = (xx - dx) * sgn
        ang = np.degrees(np.arctan2(yy - dy, horiz))
        band = (dist > 0.7 * fd) & (dist < 1.35 * fd) & (np.abs(ang - 6.0) < 22.0) & (horiz > 0) & away
        if band.sum() < 5:
            continue
        z = (sm[band].mean() - sm) / (sm[band].std() + 1e-6)
        prior = ((dist / fd - 1.0) / 0.25) ** 2 + ((ang - 6.0) / 12.0) ** 2
        cost = np.where(band, -z + 0.35 * prior, np.inf)
        iy, ix = np.unravel_index(int(np.argmin(cost)), cost.shape)
        if best is None or cost[iy, ix] < best[0]:
            best = (float(cost[iy, ix]), iy, ix, float(z[iy, ix]))
    if best is None:                       # no room on either side: darkest point well inside the field
        band = ndimage.binary_erosion(f, iterations=max(1, int(0.1 * d)))
        sm_m = np.where(band, sm, np.inf)
        fy, fx = np.unravel_index(int(np.argmin(sm_m)), sm_m.shape)
        fovea_z = float((sm[band].mean() - sm[fy, fx]) / (sm[band].std() + 1e-6))
    else:
        _, fy, fx, fovea_z = best
    disc = [dx / s, dy / s]
    fovea = [fx / s, fy / s]
    sep = math.hypot(fovea[0] - disc[0], fovea[1] - disc[1])
    return {"disc": [round(v, 1) for v in disc], "fovea": [round(v, 1) for v in fovea],
            "um_per_px": round(FOVEA_DISC_UM / sep, 4) if sep > 0 else None,
            "confidence": {"disc_z": round(disc_z, 2), "fovea_z": round(fovea_z, 2)}, "source": "auto", "field_deg": field_deg}


def thumbnail_b64(image, width=320):
    im = image.convert("RGB")
    h = round(im.size[1] * width / im.size[0])
    buf = io.BytesIO()
    im.resize((width, h), Image.BILINEAR).save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode("ascii"), width / im.size[0]


QC_PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Landmark QC</title><style>
:root{color-scheme:light}body{margin:0;font:14px system-ui,sans-serif;background:#ecebe7;color:#1b1c1e}
header{position:sticky;top:0;background:#1d1e20;color:#fff;padding:10px 16px;display:flex;gap:16px;align-items:center;z-index:2}
header button{font:inherit;padding:5px 11px;border-radius:4px;border:0;background:#fff;color:#1b1c1e;font-weight:600;cursor:pointer}
main{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:10px;padding:12px}
figure{margin:0;background:#fff;border:1px solid #cfcdc6;border-radius:6px;padding:5px}figure.fixed{outline:3px solid #a85a07}
canvas{display:block;width:100%;cursor:crosshair}figcaption{font-size:12px;color:#5c5b57;padding-top:4px}
</style></head><body><header><b>Landmark QC: __SET__</b><span>Click = fovea, shift-click = disc on any image whose cross is wrong. Cyan = disc, yellow = fovea, ring = 1 DD (1.5 mm) around the fovea.</span><span id="n"></span><button id="exp">Export corrections</button></header>
<main id="grid"></main><script>
const DATA = __DATA__; const fixes = {};
function drawOne(cv, item){const c=cv.getContext("2d"),im=cv._img,s=item.s;c.drawImage(im,0,0);
 const L=Object.assign({},item.lm,fixes[item.id]||{});const cross=(p,col)=>{c.strokeStyle="#000";c.lineWidth=4;c.beginPath();c.moveTo(p[0]*s-9,p[1]*s);c.lineTo(p[0]*s+9,p[1]*s);c.moveTo(p[0]*s,p[1]*s-9);c.lineTo(p[0]*s,p[1]*s+9);c.stroke();c.strokeStyle=col;c.lineWidth=2;c.stroke();};
 cross(L.disc,"#22d3ee");cross(L.fovea,"#ffe14d");const sep=Math.hypot(L.fovea[0]-L.disc[0],L.fovea[1]-L.disc[1]);
 c.strokeStyle="#ffe14d";c.lineWidth=1.5;c.beginPath();c.arc(L.fovea[0]*s,L.fovea[1]*s,sep/3*s,0,7);c.stroke();}
for(const item of DATA){const fig=document.createElement("figure"),cv=document.createElement("canvas"),cap=document.createElement("figcaption");
 const im=new Image();im.onload=()=>{cv.width=im.width;cv.height=im.height;cv._img=im;drawOne(cv,item)};im.src="data:image/jpeg;base64,"+item.b64;
 cv.addEventListener("click",e=>{const r=cv.getBoundingClientRect(),x=(e.clientX-r.left)*cv.width/r.width/item.s,y=(e.clientY-r.top)*cv.height/r.height/item.s;
  fixes[item.id]=Object.assign({},fixes[item.id]||{},e.shiftKey?{disc:[+x.toFixed(1),+y.toFixed(1)]}:{fovea:[+x.toFixed(1),+y.toFixed(1)]});fig.classList.add("fixed");drawOne(cv,item);count();});
 cap.textContent=item.id+"  disc z "+item.lm.confidence.disc_z+", fovea z "+item.lm.confidence.fovea_z;fig.append(cv,cap);document.getElementById("grid").append(fig);}
function count(){document.getElementById("n").textContent=Object.keys(fixes).length+" corrected";}count();
document.getElementById("exp").onclick=()=>{const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([JSON.stringify({set:"__SET__",corrections:fixes},null,1)],{type:"application/json"}));a.download="corrections___SET__.json";a.click();};
</script></body></html>"""


def cmd_estimate(args):
    data = Path(args.data)
    ids = args.ids or common.bio.load_split(data, args.split)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lms, qc = {}, []
    t0 = time.time()
    for n, iid in enumerate(ids, 1):
        image = Image.open(common.bio.image_path(data, iid))
        fov = common.bio.read_fov(data, iid)
        lm = estimate(image, fov, args.field_deg)
        lms[iid] = lm
        b64, s = thumbnail_b64(image)
        qc.append({"id": iid, "lm": lm, "b64": b64, "s": s})
        if n % 25 == 0 or n == len(ids):
            print(f"[landmarks] {args.set} {n}/{len(ids)} {time.time() - t0:.0f} s", flush=True)
    payload = {"set": args.set, "data": str(data), "split": args.split, "field_deg": args.field_deg,
               "fovea_disc_um": FOVEA_DISC_UM, "written": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "landmarks": lms}
    (out / "landmarks.json").write_text(json.dumps(payload, indent=1) + "\n")
    (out / "qc.html").write_text(QC_PAGE.replace("__SET__", args.set).replace("__DATA__", json.dumps(qc)))
    print(json.dumps({"set": args.set, "images": len(ids), "landmarks": str(out / "landmarks.json"), "qc": str(out / "qc.html")}))


def cmd_apply(args):
    path = Path(args.landmarks)
    payload = json.loads(path.read_text())
    corr = json.loads(Path(args.corrections).read_text())
    if corr.get("set") and payload.get("set") and corr["set"] != payload["set"]:
        raise SystemExit(f"{args.corrections} is for {corr['set']}, {path} for {payload['set']}")
    fixes = corr.get("corrections", {})
    for iid, fix in fixes.items():
        lm = payload["landmarks"].get(iid)
        if lm is None:
            raise SystemExit(f"{iid} is not in {path}")
        lm.setdefault("estimate", {"disc": lm["disc"], "fovea": lm["fovea"]})   # the automatic estimate, kept once
        lm.update({k: v for k, v in fix.items() if k in ("fovea", "disc")})
        sep = math.hypot(lm["fovea"][0] - lm["disc"][0], lm["fovea"][1] - lm["disc"][1])
        lm["um_per_px"] = round(FOVEA_DISC_UM / sep, 4) if sep > 0 else None
        lm["source"] = fix.get("source", "manual")
    payload.setdefault("corrections_applied", []).append({"file": str(args.corrections), "n": len(fixes), "at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    path.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"[landmarks] {len(fixes)} corrections merged into {path}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("estimate")
    e.add_argument("--set", required=True)
    e.add_argument("--data", required=True)
    e.add_argument("--split", default="test")
    e.add_argument("--ids", nargs="*")
    e.add_argument("--field-deg", type=float, default=45.0)
    e.add_argument("--out", required=True)
    a = sub.add_parser("apply")
    a.add_argument("--landmarks", required=True)
    a.add_argument("--corrections", required=True)
    args = ap.parse_args(argv)
    (cmd_estimate if args.cmd == "estimate" else cmd_apply)(args)


def load_landmarks(path):
    """{image id: landmark dict} from a landmarks.json, or {} when path is empty."""
    if not path:
        return {}
    return json.loads(Path(path).read_text())["landmarks"]


if __name__ == "__main__":
    main()
