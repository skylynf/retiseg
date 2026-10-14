"""One style for every figure: sizes, fonts, palettes, panel letters, output.

Sizes follow Nature Methods: 180 mm full width, 5-7 pt text, 8 pt bold panel
letters. Colors are assigned by the job they do:

  identity    FAMILY (model family), LESION (lesion class)   categorical
  order       TIER (annotation granularity, fine to region)  one-hue ordinal
  magnitude   CMAP_SEQ (AUPR, recall, any 0-1 score)         sequential
  difference  CMAP_DIV (deltas around zero)                  diverging, gray midpoint

The template (Lu 2026) colors AUPRC with a blue-yellow-red map centred on the
0.5 chance level. Pixel AUPR has no such midpoint (chance is the lesion
prevalence, about 1e-3), so magnitudes use a sequential map here and the
diverging map is kept for differences.

Both categorical sets pass the all-pairs colour-vision check
(dataviz validate_palette.js, white surface): worst CVD dE 9.2 / 13.0.
"""

import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

OUT = Path(__file__).resolve().parents[1] / "out"
MM = 1 / 25.4
FULL_W = 180 * MM
HALF_W = 88 * MM

# Set RETISEG_FINAL=1 to drop the "simulated" tags once every row is measured.
FINAL = os.environ.get("RETISEG_FINAL", "0") == "1"

INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#b9b8ae"
PAPER = "#ffffff"
SOFT = "#f4f3ef"

FAMILY = {
    "general": "#2a78d6",
    "auto": "#1baf7a",
    "fundus": "#eb6834",
    "foundation": "#4a3aa7",
}
FAMILY_LABEL = {
    "general": "General segmentation",
    "auto": "Self-configuring",
    "fundus": "Fundus-specific",
    "foundation": "Foundation model",
}
FAMILY_SHORT = {"general": "General", "auto": "Self-config.", "fundus": "Fundus-spec.", "foundation": "Foundation"}
FAMILY_ORDER = ["general", "auto", "fundus", "foundation"]

LESION = {"MA": "#e87ba4", "HE": "#b3261e", "EX": "#eda100", "SE": "#2a78d6"}
LESION_NAME = {"MA": "Microaneurysm", "HE": "Hemorrhage", "EX": "Hard exudate", "SE": "Soft exudate"}

TIER = {"fine": "#1d3f5e", "mixed": "#3f6f9c", "coarse": "#7aa3c7", "region": "#aabfd3"}
TIER_LABEL = {"fine": "Fine", "mixed": "Mixed", "coarse": "Coarse", "region": "Region consensus"}
DATASET_TIER = {
    "IDRiD": "fine",
    "DDR": "fine",
    "TJDR-std": "fine",
    "TJDR-uwf": "fine",
    "MAPLES-DR": "fine",
    "FGADR": "mixed",
    "Retinal-Lesions": "coarse",
    "DiaRetDB1": "region",
}
DATASET_SHORT = {
    "IDRiD": "IDRiD",
    "DDR": "DDR",
    "TJDR-std": "TJDR\nstd",
    "TJDR-uwf": "TJDR\nUWF",
    "MAPLES-DR": "Maples",
    "FGADR": "FGADR",
    "Retinal-Lesions": "Ret.-\nLesions",
    "DiaRetDB1": "DiaRet\nDB1",
}

CMAP_SEQ = LinearSegmentedColormap.from_list(
    "retiseg_seq", ["#fbfaf4", "#e2f1e6", "#aeddd0", "#62bfc1", "#2f93b6", "#255f9c", "#1d3572"]
)
CMAP_DIV = LinearSegmentedColormap.from_list(
    "retiseg_div", ["#b3261e", "#e8867c", "#f0efec", "#86b6ef", "#1c5cab"]
)
CMAP_WARM = LinearSegmentedColormap.from_list(
    "retiseg_warm", ["#fbfaf4", "#fde6c4", "#f9c178", "#ee8f45", "#d3562c", "#9c2a1d"]
)


def apply():
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
            "font.size": 6.5,
            "axes.titlesize": 7,
            "axes.titleweight": "regular",
            "axes.labelsize": 6.5,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
            "legend.frameon": False,
            "legend.handlelength": 1.2,
            "legend.handletextpad": 0.5,
            "legend.borderaxespad": 0.2,
            "axes.linewidth": 0.5,
            "axes.edgecolor": INK2,
            "axes.labelcolor": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlepad": 4,
            "axes.labelpad": 2.5,
            "xtick.color": INK2,
            "ytick.color": INK2,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 2.2,
            "ytick.major.size": 2.2,
            "xtick.major.pad": 2,
            "ytick.major.pad": 2,
            "lines.linewidth": 1.0,
            "lines.markersize": 3.5,
            "patch.linewidth": 0.5,
            "hatch.linewidth": 0.4,
            "figure.facecolor": PAPER,
            "axes.facecolor": PAPER,
            "savefig.facecolor": PAPER,
            "figure.dpi": 150,
            "savefig.dpi": 400,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "mathtext.default": "regular",
        }
    )


def figure(height_mm, width=FULL_W):
    apply()
    return plt.figure(figsize=(width, height_mm * MM))


def panel_label(ax, letter, dx=-0.0, dy=0.0, x=None, y=None):
    """Bold lowercase panel letter at the top-left of the axes, in figure coordinates."""
    fig = ax.figure
    box = ax.get_position()
    px = box.x0 + dx if x is None else x
    py = box.y1 + dy if y is None else y
    fig.text(px, py, letter, fontsize=9, fontweight="bold", color=INK, ha="left", va="bottom")


def tag_simulated(ax, loc="tr", text="simulated", dx=0.0, dy=0.0):
    """Mark a panel whose numbers come from the placeholder generator."""
    if FINAL:
        return
    x, ha = (0.995, "right") if loc.endswith("r") else (0.005, "left")
    y, va = (1.012, "bottom") if loc.startswith("t") else (0.01, "bottom")
    ax.text(
        x + dx,
        y + dy,
        text,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=4.6,
        color="#9a5b00",
        style="italic",
        bbox=dict(boxstyle="round,pad=0.22", fc="#fff3d6", ec="#e8c36a", lw=0.3),
        zorder=50,
        clip_on=False,
    )


def light_grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, lw=0.4, zorder=0)
    ax.set_axisbelow(True)


def bare(ax):
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])


def text_on(rgba):
    """Black or white ink for a label sitting on a filled cell."""
    r, g, b = rgba[:3]
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "#ffffff" if lum < 0.52 else INK


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=300)
    plt.close(fig)
    print(f"wrote {OUT / name}.pdf / .png")


def axes_mm(fig, left, top, width, height, **kw):
    """Axes placed in millimetres from the top-left corner of the figure."""
    W, H = fig.get_size_inches() / MM
    return fig.add_axes([left / W, 1 - (top + height) / H, width / W, height / H], **kw)


def rect_mm(fig, left, top, width, height):
    """Figure-fraction rectangle for a box given in millimetres from the top-left."""
    W, H = fig.get_size_inches() / MM
    return [left / W, 1 - (top + height) / H, width / W, height / H]


def label_mm(fig, letter, left, top):
    """Panel letter at a position in millimetres from the top-left corner."""
    W, H = fig.get_size_inches() / MM
    fig.text(left / W, 1 - top / H, letter, fontsize=9, fontweight="bold", color=INK, ha="left", va="top")


def text_mm(fig, s, left, top, **kw):
    W, H = fig.get_size_inches() / MM
    kw.setdefault("ha", "left")
    kw.setdefault("va", "top")
    kw.setdefault("fontsize", 6.5)
    kw.setdefault("color", INK)
    return fig.text(left / W, 1 - top / H, s, **kw)


def family_chips(ax, families, x=-0.035, size=9):
    """Small family-coloured squares between the row labels and the matrix."""
    import numpy as np

    trans = ax.get_yaxis_transform()
    y = np.arange(len(families)) + 0.5
    ax.scatter(np.full(len(families), x), y, s=size, marker="s", c=[FAMILY[f] for f in families],
               transform=trans, clip_on=False, linewidths=0, zorder=6)
    ax.tick_params(axis="y", pad=7)


# The five representative models of the factor experiments: family colour plus a marker.
REP_STYLE = {
    "U-Net": dict(color=FAMILY["general"], marker="o", ls="-"),
    "nnU-Net": dict(color=FAMILY["auto"], marker="s", ls="-"),
    "M2MRF": dict(color=FAMILY["fundus"], marker="^", ls="-"),
    "SAM2-UNet": dict(color=FAMILY["foundation"], marker="D", ls="-"),
    "RETFound + UPerNet": dict(color="#9085e9", marker="v", ls=(0, (3, 1.5))),
}


def rep_legend(target, loc="upper left", bbox=None, ncol=5, fs=5.6):
    from matplotlib.lines import Line2D

    h = [Line2D([], [], color=st["color"], marker=st["marker"], ls=st["ls"], lw=1.0, ms=3.4, mec=PAPER, mew=0.4, label=name)
         for name, st in REP_STYLE.items()]
    return target.legend(handles=h, loc=loc, bbox_to_anchor=bbox, ncol=ncol, fontsize=fs, columnspacing=1.3, handletextpad=0.4, handlelength=2.0)
