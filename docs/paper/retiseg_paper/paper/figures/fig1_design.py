"""Fig. 1 | Data collection and study design. Every number is from the repository.

a  workflow: data, models, four evaluations, two test scenarios, metrics
b  the seven test sources: splits, role, annotation granularity
c  the 38 registered model variants by family and reproducibility level
d  literature search and screening
e  the factors varied one at a time (Q3)
"""

import numpy as np
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, PathPatch, Rectangle
from matplotlib.path import Path as MPath

from retifig import data as D
from retifig import panels as P
from retifig import style as S

REPRO = ["R0", "R1", "R2", "R3", "R4"]
REPRO_COLOR = dict(zip(REPRO, ["#e4e3dc", "#c9d9e8", "#8fb0cc", "#4f7ea8", "#1d3f5e"]))
REPRO_TEXT = {"R0": "no code", "R1": "model definition", "R2": "inference + weights", "R3": "training code", "R4": "code + weights + splits + env."}
FAMILY_KEY = {"general": "general", "fundus_multiscale": "fundus", "relational_transformer": "fundus", "semi_supervised": "fundus", "foundation": "foundation"}
FAMILY_ROWS = [
    ("general", "General segmentation"),
    ("fundus_multiscale", "Fundus-specific: multi-scale CNN"),
    ("relational_transformer", "Fundus-specific: relation / transformer"),
    ("semi_supervised", "Fundus-specific: semi-supervised"),
    ("foundation", "Foundation models"),
]


def box(ax, x, y, w, h, fc, ec="none", r=1.2, lw=0.5, z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, zorder=z))


def arrow(ax, x0, y0, x1, y1, color=S.MUTED):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=5, lw=0.7, color=color, zorder=5, shrinkA=0, shrinkB=0))


def fundus(ax, cx, cy, r):
    """Stylised fundus with the four lesion types; a drawing, not data."""
    rng = np.random.default_rng(7)
    ax.add_patch(Circle((cx, cy), r, fc="#f3b58a", ec="#d98d5f", lw=0.5, zorder=3))
    ax.add_patch(Circle((cx + r * 0.48, cy), r * 0.17, fc="#fff1c9", ec="#f0c98a", lw=0.4, zorder=4))
    ax.add_patch(Circle((cx - r * 0.1, cy), r * 0.14, fc="#d98d5f", ec="none", alpha=0.45, zorder=4))
    for sgn in (-1, 1):
        for bend, lw in ((0.95, 0.9), (0.55, 0.6)):
            verts = [(cx + r * 0.48, cy), (cx + r * 0.15, cy + sgn * r * bend), (cx - r * 0.75, cy + sgn * r * bend * 0.7)]
            ax.add_patch(PathPatch(MPath(verts, [MPath.MOVETO, MPath.CURVE3, MPath.CURVE3]), fc="none", ec="#b6452c", lw=lw, zorder=4, capstyle="round"))
    def inside(n, spread=0.78):
        a = rng.uniform(0, 2 * np.pi, n)
        d = r * spread * np.sqrt(rng.uniform(0.03, 1, n))
        return cx + d * np.cos(a), cy + d * np.sin(a)
    x, y = inside(16)
    ax.scatter(x, y, s=1.6, color=S.LESION["MA"], zorder=6, linewidths=0)
    x, y = inside(5, 0.7)
    for xi, yi in zip(x, y):
        ax.add_patch(Ellipse((xi, yi), r * rng.uniform(0.12, 0.2), r * rng.uniform(0.08, 0.14), angle=rng.uniform(0, 180), fc=S.LESION["HE"], ec="none", zorder=6))
    x, y = inside(22, 0.6)
    ax.scatter(x * 0.55 + (cx - r * 0.25) * 0.45, y * 0.55 + (cy + r * 0.3) * 0.45, s=2.4, color=S.LESION["EX"], zorder=6, linewidths=0, marker="o")
    for dx, dy in ((-0.5, -0.45), (0.1, -0.6)):
        ax.add_patch(Ellipse((cx + r * dx, cy + r * dy), r * 0.24, r * 0.16, angle=20, fc="#ffffff", ec=S.LESION["SE"], lw=0.7, zorder=6))


def panel_a(fig):
    ax = S.axes_mm(fig, 0, 0, 180, 62)
    ax.set_xlim(0, 180)
    ax.set_ylim(62, 0)
    S.bare(ax)
    reg = D.registry()
    n_img = sum(d["n"] for d in D.DATASETS)
    cols = [
        (4, 34, "Task"),
        (40, 32, "Data collection"),
        (74, 32, "Model collection"),
        (108, 34, "Four evaluations"),
        (144, 32, "Tests and metrics"),
    ]
    top, h = 8, 51
    for i, (x, w, title) in enumerate(cols):
        box(ax, x, top, w, h, S.SOFT, r=1.6, z=1)
        ax.text(x + 2, top + 2.2, title, fontsize=6.8, fontweight="bold", va="top")
        if i:
            arrow(ax, x - 2.0, top + h / 2, x - 0.1, top + h / 2)
    # --- task
    ax.text(6, top + 7.4, "Colour fundus photograph in,\nfour lesion probability maps out", fontsize=5.3, va="top", color=S.INK2, linespacing=1.2)
    fundus(ax, 21, 34.5, 10.5)
    for k, cls in enumerate(D.CLASSES):
        xx = 6.8 + (k % 2) * 15.5
        y0 = 49.5 + (k // 2) * 4.6
        ax.scatter([xx], [y0], s=11, color=S.LESION[cls], linewidths=0, zorder=6)
        ax.text(xx + 1.8, y0 - 0.9, cls, fontsize=5.2, va="center", fontweight="bold")
        ax.text(xx + 1.8, y0 + 1.3, S.LESION_NAME[cls].lower(), fontsize=4.5, va="center", color=S.INK2)
    # --- data
    x = 42
    ax.text(x, top + 7.4, f"7 test sources · {n_img:,} images", fontsize=5.6, color=S.INK2, va="top")
    groups = [("Training sources", [d for d in D.DATASETS if d["role"] == "train"], "#3d3d3a"),
              ("Independent sources", [d for d in D.DATASETS if d["role"] != "train"], "#8f8d86")]
    y = top + 12.5
    for title, items, fc in groups:
        box(ax, x, y, 28, 3.6, fc, r=0.8)
        ax.text(x + 14, y + 1.8, title, fontsize=5.2, color="#ffffff", ha="center", va="center")
        y += 4.6
        for d in items:
            ax.add_patch(Rectangle((x, y + 0.5), 1.6, 2.2, fc=S.TIER[d["tier"]], ec="none", zorder=3))
            ax.text(x + 2.6, y + 1.6, d["name"], fontsize=5.4, va="center")
            ax.text(x + 28, y + 1.6, f"{d['n']:,}", fontsize=5.2, va="center", ha="right", color=S.INK2)
            y += 3.55
        y += 1.2
    # --- models
    x = 76
    n_all = len(reg)
    ax.text(x, top + 7.4, f"{n_all} variants in the registry", fontsize=5.6, color=S.INK2, va="top")
    tiers = [(1, "Tier 1 · main table", "#1d3f5e"), (2, "Tier 2 · if time allows", "#4f7ea8"), (3, "Tier 3 · paper audit only", "#8fb0cc")]
    y = top + 12.5
    for t, label, fc in tiers:
        n = int((reg.tier == t).sum())
        box(ax, x, y, 28, 6.4, fc, r=0.8)
        ax.text(x + 2, y + 3.2, label, fontsize=5.3, color="#ffffff", va="center")
        ax.text(x + 26, y + 3.2, str(n), fontsize=7.5, color="#ffffff", va="center", ha="right", fontweight="bold")
        y += 7.6
    ax.text(x, y + 0.8, "Reproducibility level R0–R4\nrecorded for every variant;\nfailures stay in the denominator", fontsize=5.1, color=S.INK2, va="top", linespacing=1.25)
    # --- evaluations
    x = 110
    evals = [
        ("Paper audit", "numbers as reported,\nand where they came from", "L1", "#e4e3dc", S.INK),
        ("Author weights", "released checkpoints,\nthree numbers each", "E1", "#f6c9b3", S.INK),
        ("Author recipe", "retrained with the\nauthors' own pipeline", "E1r", "#f2a37f", S.INK),
        ("Unified retraining", "shared data, budget,\naugmentation, selection", "E2", "#eb6834", "#ffffff"),
    ]
    y = top + 7.6
    for title, sub, tag, fc, ink in evals:
        box(ax, x, y, 30, 9.6, fc, r=0.9)
        ax.text(x + 1.8, y + 2.5, title, fontsize=5.7, fontweight="bold", color=ink, va="center")
        ax.text(x + 1.8, y + 6.3, sub, fontsize=4.8, color=ink, va="center", linespacing=1.15)
        ax.text(x + 28.4, y + 2.5, tag, fontsize=5.0, color=ink, va="center", ha="right", style="italic")
        y += 10.55
    # --- scenarios and metrics
    x = 146
    box(ax, x, top + 7.6, 28, 9.2, "#3d3d3a", r=0.9)
    ax.text(x + 14, top + 10.4, "Internal test", fontsize=5.7, color="#fff", ha="center", va="center", fontweight="bold")
    ax.text(x + 14, top + 14.0, "official test set of the\ntraining source", fontsize=4.8, color="#fff", ha="center", va="center", linespacing=1.15)
    box(ax, x, top + 18.0, 28, 9.2, "#d9d7cf", r=0.9)
    ax.text(x + 14, top + 20.8, "Independent test", fontsize=5.7, ha="center", va="center", fontweight="bold")
    ax.text(x + 14, top + 24.4, "sources never used\nfor training", fontsize=4.8, ha="center", va="center", linespacing=1.15)
    lines = [
        ("Primary", "exact pooled AUPR inside\nthe field of view; mAUPR"),
        ("Secondary", "Dice, IoU; lesion-level\nrecall by lesion size"),
        ("Statistics", "3 seeds; image bootstrap;\nFriedman + Nemenyi"),
    ]
    y = top + 29.0
    for head, body in lines:
        ax.text(x, y, head, fontsize=5.2, fontweight="bold", va="top")
        ax.text(x + 0.0, y + 2.4, body, fontsize=4.8, color=S.INK2, va="top", linespacing=1.15)
        y += 7.0


def panel_b(fig, top):
    ax = S.axes_mm(fig, 27, top, 48, 40)
    ds = D.DATASETS
    y = np.arange(len(ds))
    shades = {"train": 1.0, "val": 0.55, "test": 0.28}
    left = np.zeros(len(ds))
    xmax = 1900
    for part in ("train", "val", "test"):
        vals = np.array([d[part] for d in ds], float)
        for i, d in enumerate(ds):
            base = S.TIER[d["tier"]]
            ax.barh(y[i], vals[i], left=left[i], height=0.62, color=base, alpha=shades[part], lw=0, zorder=2)
            if vals[i] > 0 and part != "test":
                ax.plot([left[i] + vals[i]] * 2, [y[i] - 0.31, y[i] + 0.31], color=S.PAPER, lw=0.8, zorder=3)
        left += vals
    for i, d in enumerate(ds):
        split = f"{d['train']:,}/{d['val']}/{d['test']}" if d["train"] else f"–/{d['val']}/{d['test']}"
        ax.text(left[i] + 35, y[i], split, va="center", fontsize=5.2, color=S.INK)
    ax.set_yticks(y)
    ax.set_yticklabels([d["name"] for d in ds])
    ax.set_ylim(len(ds) - 0.45, -0.55)
    ax.set_xlim(0, xmax + 1700)
    ax.set_xticks([0, 500, 1000, 1500])
    ax.set_xlabel("Images (train / validation / test)")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    S.light_grid(ax, "x")
    ax.axhline(1.5, color=S.INK2, lw=0.5, ls=(0, (2, 2)))
    ax.text(xmax + 1680, 0.5, "training\nsources", ha="right", va="center", fontsize=5.2, color=S.INK2)
    ax.text(xmax + 1680, 4.0, "independent\nsources", ha="right", va="center", fontsize=5.2, color=S.INK2)
    # tier key
    x0 = 27
    for k, tier in enumerate(("fine", "mixed", "coarse", "region")):
        axk = S.axes_mm(fig, x0, top + 49.5, 2.2, 2.2)
        axk.add_patch(Rectangle((0, 0), 1, 1, fc=S.TIER[tier], ec="none"))
        S.bare(axk)
        S.text_mm(fig, S.TIER_LABEL[tier], x0 + 3, top + 49.4, fontsize=5.3, color=S.INK2)
        x0 += 6 + 1.25 * len(S.TIER_LABEL[tier])


def panel_c(fig, top):
    reg = D.registry()
    ax = S.axes_mm(fig, 121, top, 38, 28)
    rows = FAMILY_ROWS
    y = np.arange(len(rows))
    left = np.zeros(len(rows))
    for level in REPRO:
        vals = np.array([((reg.family == fam) & (reg.repro_level == level)).sum() for fam, _ in rows], float)
        ax.barh(y, vals, left=left, height=0.62, color=REPRO_COLOR[level], lw=0, zorder=2)
        for i, v in enumerate(vals):
            if v > 0:
                ax.plot([left[i] + v] * 2, [y[i] - 0.31, y[i] + 0.31], color=S.PAPER, lw=0.8, zorder=3)
                ax.text(left[i] + v / 2, y[i], str(int(v)), ha="center", va="center", fontsize=5.0,
                        color="#ffffff" if level in ("R3", "R4") else S.INK)
        left += vals
    for i, (fam, _) in enumerate(rows):
        t1 = int(((reg.family == fam) & (reg.tier == 1)).sum())
        ax.text(left[i] + 0.4, y[i], f"{int(left[i])} · {t1} in tier 1" if t1 else f"{int(left[i])}", va="center", fontsize=5.1, color=S.INK2)
    ax.set_yticks(y)
    ax.set_yticklabels([name for _, name in rows])
    ax.set_ylim(len(rows) - 0.45, -0.55)
    ax.set_xlim(0, 17.5)
    ax.set_xticks([0, 4, 8, 12])
    ax.set_xlabel("Model variants")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    S.light_grid(ax, "x")
    # key
    yk = top + 37.5
    S.text_mm(fig, "Reproducibility level", 84, yk, fontsize=5.4, color=S.INK2)
    for k, level in enumerate(REPRO):
        col, row = k % 2, k // 2
        xk = 84 + col * 30
        axk = S.axes_mm(fig, xk, yk + 3.4 + row * 3.4, 2.2, 2.2)
        axk.add_patch(Rectangle((0, 0), 1, 1, fc=REPRO_COLOR[level], ec="none"))
        S.bare(axk)
        S.text_mm(fig, f"{level}  {REPRO_TEXT[level]}", xk + 3, yk + 3.3 + row * 3.4, fontsize=5.1, color=S.INK2)


def panel_d(fig, top):
    ax = S.axes_mm(fig, 2, top, 84, 37)
    labels = [s[0] for s in D.SEARCH_FLOW]
    values = [s[1] for s in D.SEARCH_FLOW]
    notes = [s[2] for s in D.SEARCH_FLOW]
    ramp = ["#c9d9e8", "#a9c3da", "#8fb0cc", "#6a94ba", "#4f7ea8", "#1d3f5e"]
    P.funnel(ax, labels, values, ramp, notes=notes, fs=5.6)
    ax.set_xlim(-1.45, 0.62)
    for t in ax.texts:
        if t.get_ha() == "right":
            t.set_x(-0.6)
    S.text_mm(fig, "Fixed search 2015-01-01 to 2026-09-30; with the snowballed corpus the ledger holds 86 records.", 4, top + 38.5, fontsize=5.0, color=S.INK2)


def panel_e(fig, top):
    ax = S.axes_mm(fig, 92, top, 86, 38)
    ax.set_xlim(0, 86)
    ax.set_ylim(38, 0)
    S.bare(ax)
    cards = [
        ("Training source", "E3b", "IDRiD · DDR ·\nTJDR-std · FGADR"),
        ("Label granularity", "E4", "fine · mixed · coarse,\nalone and combined"),
        ("Training amount", "E8", "10–100% of DDR,\nmerged sources"),
        ("Input resolution", "E7", "512 · 1,024 · 1,440 ·\nnative patches"),
        ("Loss", "E9", "CE · Dice ·\nCE + Dice · bin loss"),
        ("Extra input", "E6", "image alone or\nwith vessel map"),
        ("Foundation models", "E5", "zero-shot · oracle\nprompt · PEFT ·\nencoder + decoder"),
        ("Training recipe", "E2b, E2c", "source recipe (B1);\nlearning rate ×⅓, 1, 3"),
    ]
    w, h, gx, gy = 20.3, 16.5, 1.6, 1.8
    for k, (title, tag, body) in enumerate(cards):
        col, row = k % 4, k // 4
        x, y = col * (w + gx), 0.5 + row * (h + gy)
        box(ax, x, y, w, h, S.SOFT, r=1.0)
        ax.add_patch(Rectangle((x, y + 1.2), 0.9, 5.2, fc=S.FAMILY["fundus"] if tag != "E5" else S.FAMILY["foundation"], ec="none", zorder=3))
        ax.text(x + 2.0, y + 2.4, title, fontsize=5.5, fontweight="bold", va="center")
        ax.text(x + 2.0, y + 5.4, tag, fontsize=4.9, style="italic", color=S.INK2, va="center")
        ax.text(x + 2.0, y + 11.6, body, fontsize=4.9, color=S.INK2, va="center", linespacing=1.2)
    S.text_mm(fig, "One factor changes at a time; everything else stays fixed. E3b–E9 use five representative models: U-Net,\nnnU-Net, M2MRF, SAM2-UNet, RETFound + UPerNet.",
              92, top + 38.5, fontsize=5.0, color=S.INK2, linespacing=1.25)


def main():
    fig = S.figure(176)
    S.label_mm(fig, "a", 2, 2)
    panel_a(fig)
    S.label_mm(fig, "b", 2, 66)
    S.text_mm(fig, "Test sources and frozen splits", 8, 66.6, fontsize=6.8, fontweight="bold")
    panel_b(fig, 73)
    S.label_mm(fig, "c", 78, 66)
    S.text_mm(fig, "Registered models by family and reproducibility level", 84, 66.6, fontsize=6.8, fontweight="bold")
    panel_c(fig, 73)
    S.label_mm(fig, "d", 2, 127)
    S.text_mm(fig, "Literature search and screening", 8, 127.6, fontsize=6.8, fontweight="bold")
    panel_d(fig, 132)
    S.label_mm(fig, "e", 88, 127)
    S.text_mm(fig, "Factors assessed", 94, 127.6, fontsize=6.8, fontweight="bold")
    panel_e(fig, 132)
    S.save(fig, "fig1_design")


if __name__ == "__main__":
    main()
