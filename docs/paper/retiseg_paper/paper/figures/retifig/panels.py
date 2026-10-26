"""Reusable panels. Each takes an Axes and plain arrays; none reads files.

heat            numbered heat map (template Fig. 2e, left)
score_bars      horizontal bars with the value written at the end (template Fig. 2e, right)
dot_matrix      coloured dots on a model x condition grid (template Fig. 3b)
strip_mean_sd   one dot per model, black mean and s.d. (template Fig. 3c)
paired_slope    the same models under two or three settings, joined by lines (template Fig. 4)
bracket         P-value bracket above two x positions
cd_diagram      critical-difference diagram (Demsar 2006)
bump            rank under one condition against rank under another
dumbbell        two or three values per row on one scale
funnel          centred shrinking bars
colorbar        slim horizontal colour bar with a title
"""

import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle

from . import style as S


def heat(ax, M, rows=None, cols=None, cmap=None, vmin=0.0, vmax=1.0, fmt="{:.2f}", fs=5.2, show_values=True,
         col_rotation=0, gap=0.6, na="–", bold_max=None):
    """Numbered heat map. `bold_max='col'` bolds the best value of each column."""
    M = np.asarray(M, dtype=float)
    cmap = cmap or S.CMAP_SEQ
    n, m = M.shape
    masked = np.ma.masked_invalid(M)
    ax.pcolormesh(np.arange(m + 1), np.arange(n + 1), masked, cmap=cmap, vmin=vmin, vmax=vmax,
                  edgecolors=S.PAPER, linewidth=gap, rasterized=False)
    ax.set_xlim(0, m)
    ax.set_ylim(n, 0)
    best = set()
    if bold_max == "col":
        for j in range(m):
            if np.isfinite(M[:, j]).any():
                best.add((int(np.nanargmax(M[:, j])), j))
    if show_values:
        for i in range(n):
            for j in range(m):
                v = M[i, j]
                if not np.isfinite(v):
                    ax.text(j + 0.5, i + 0.5, na, ha="center", va="center", fontsize=fs, color=S.MUTED)
                    continue
                rgba = cmap((v - vmin) / (vmax - vmin) if vmax > vmin else 0.5)
                ax.text(j + 0.5, i + 0.5, fmt.format(v), ha="center", va="center", fontsize=fs,
                        color=S.text_on(rgba), fontweight="bold" if (i, j) in best else "regular")
    ax.set_xticks(np.arange(m) + 0.5)
    ax.set_yticks(np.arange(n) + 0.5)
    ax.set_xticklabels(cols if cols is not None else [""] * m, rotation=col_rotation,
                       ha="right" if col_rotation else "center", rotation_mode="anchor" if col_rotation else None)
    ax.set_yticklabels(rows if rows is not None else [""] * n)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)


def score_bars(ax, values, colors, xmax=1.0, fmt="{:.2f}", fs=5.2, err=None, xlabel=None, height=0.62):
    """Horizontal bars aligned with the rows of a neighbouring heat map (row 0 at the top)."""
    values = np.asarray(values, dtype=float)
    y = np.arange(len(values)) + 0.5
    ax.barh(y, values, height=height, color=colors, lw=0, zorder=2)
    if err is not None:
        ax.errorbar(values, y, xerr=err, fmt="none", ecolor=S.INK, elinewidth=0.5, capsize=0, zorder=3)
    pad = xmax * 0.03
    for yi, v, e in zip(y, values, err if err is not None else np.zeros_like(values)):
        ax.text(v + (e if np.isfinite(e) else 0) + pad, yi, fmt.format(v), va="center", ha="left", fontsize=fs, color=S.INK)
    ax.set_xlim(0, xmax)
    ax.set_ylim(len(values), 0)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="x", length=2)
    if xlabel:
        ax.set_xlabel(xlabel)


def dot_matrix(ax, M, rows=None, cols=None, cmap=None, vmin=0.0, vmax=1.0, size=24, frame=True, col_fs=5.5):
    """Coloured dots, one per cell; a thin frame groups the block like the template."""
    M = np.asarray(M, dtype=float)
    cmap = cmap or S.CMAP_SEQ
    n, m = M.shape
    xx, yy = np.meshgrid(np.arange(m) + 0.5, np.arange(n) + 0.5)
    ok = np.isfinite(M)
    ax.scatter(xx[ok], yy[ok], c=M[ok], cmap=cmap, vmin=vmin, vmax=vmax, s=size, edgecolors=S.AXIS, linewidths=0.25, zorder=2)
    ax.set_xlim(0, m)
    ax.set_ylim(n, 0)
    ax.set_xticks(np.arange(m) + 0.5)
    ax.set_xticklabels(cols if cols is not None else [""] * m, fontsize=col_fs)
    ax.set_yticks(np.arange(n) + 0.5)
    ax.set_yticklabels(rows if rows is not None else [""] * n)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(frame)
        s.set_linewidth(0.6)
        s.set_color(S.INK)


def strip_mean_sd(ax, groups, colors, labels=None, jitter=0.12, seed=3, size=9, point_colors=None):
    """groups: list of arrays. Coloured dots, black mean dot, error bar = mean +- s.d."""
    rng = np.random.default_rng(seed)
    for i, vals in enumerate(groups):
        vals = np.asarray(vals, dtype=float)
        x = i + rng.uniform(-jitter, jitter, len(vals))
        c = point_colors[i] if point_colors is not None else colors[i]
        ax.scatter(x, vals, s=size, color=c, alpha=0.8, linewidths=0, zorder=2)
        if len(vals) > 1:
            ax.errorbar([i], [vals.mean()], yerr=[vals.std(ddof=1)], fmt="o", color=S.INK, ms=3.6, elinewidth=0.7, capsize=1.8, capthick=0.7, zorder=4)
        else:
            ax.plot([i], [vals.mean()], "o", color=S.INK, ms=3.6, zorder=4)
    ax.set_xticks(range(len(groups)))
    if labels is not None:
        ax.set_xticklabels(labels)
    ax.set_xlim(-0.6, len(groups) - 0.4)


def paired_slope(ax, columns, colors, xlabels, line_color="#c9c8c0", size=11, highlight=None):
    """columns: list of equal-length arrays (one per x position); row i is one model."""
    cols = [np.asarray(c, dtype=float) for c in columns]
    n = len(cols[0])
    xs = np.arange(len(cols))
    for i in range(n):
        hl = highlight is not None and i in highlight
        ax.plot(xs, [c[i] for c in cols], color=S.INK2 if hl else line_color, lw=0.9 if hl else 0.45, zorder=3 if hl else 1)
    for x, c in zip(xs, cols):
        ax.scatter(np.full(n, x), c, s=size, c=colors, linewidths=0.3, edgecolors=S.PAPER, zorder=4)
    ax.set_xticks(xs)
    ax.set_xticklabels(xlabels)
    ax.set_xlim(-0.45, len(cols) - 0.55)


def bracket(ax, x0, x1, y, text, h=0.012, fs=5.2):
    """P-value bracket in data coordinates."""
    ax.plot([x0, x0, x1, x1], [y - h, y, y, y - h], color=S.INK, lw=0.5, clip_on=False)
    ax.text((x0 + x1) / 2, y + h * 0.4, text, ha="center", va="bottom", fontsize=fs, color=S.INK)


def p_text(p):
    if not np.isfinite(p):
        return "P = n/a"
    if p < 0.001:
        return "P < 0.001"
    return f"P = {p:.3f}" if p < 0.01 else f"P = {p:.2f}"


def cd_diagram(ax, names, avg_ranks, cd, colors, fs=5.6, title=None):
    """Critical-difference diagram: rank axis on top, best on the left, bars join
    models whose average ranks differ by less than the Nemenyi critical difference."""
    names = list(names)
    r = np.asarray(avg_ranks, dtype=float)
    order = np.argsort(r)
    k = len(names)
    lo, hi = 1, k
    ax.set_xlim(lo - 0.6, hi + 0.6)
    n_left = (k + 1) // 2
    depth = max(n_left, k - n_left)
    ax.set_ylim(-(depth + 1.6), 2.9)
    ax.hlines(0, lo, hi, color=S.INK, lw=0.7)
    for t in range(lo, hi + 1):
        ax.vlines(t, 0, 0.18, color=S.INK, lw=0.6)
        ax.text(t, 0.3, str(t), ha="center", va="bottom", fontsize=fs)
    ax.text((lo + hi) / 2, 2.2, title or "Average rank over test sets (1 = best)", ha="center", va="bottom", fontsize=fs + 0.4)
    # critical difference ruler
    ax.hlines(1.35, lo, lo + cd, color=S.INK, lw=0.9)
    ax.vlines([lo, lo + cd], 1.22, 1.48, color=S.INK, lw=0.6)
    ax.text(lo + cd + 0.15, 1.35, f"CD = {cd:.2f}", ha="left", va="center", fontsize=fs)
    # cliques: maximal runs of consecutive models within one CD
    sorted_r = r[order]
    cliques = []
    for i in range(k):
        j = i
        while j + 1 < k and sorted_r[j + 1] - sorted_r[i] <= cd:
            j += 1
        if j > i and not any(a <= i and j <= b for a, b in cliques):
            cliques.append((i, j))
    for level, (i, j) in enumerate(cliques):
        y = -0.32 - 0.26 * level
        ax.hlines(y, sorted_r[i] - 0.06, sorted_r[j] + 0.06, color=S.INK2, lw=1.6, alpha=0.85)
    base = -0.32 - 0.26 * max(len(cliques), 1) - 0.35
    for pos, idx in enumerate(order):
        left = pos < n_left
        row = pos if left else k - 1 - pos
        y = base - row * 0.95
        x_text = lo - 0.45 if left else hi + 0.45
        ax.plot([r[idx], r[idx], x_text], [0, y, y], color=colors[idx], lw=0.7, solid_joinstyle="miter")
        ax.plot([r[idx]], [0], "o", ms=2.6, color=colors[idx], zorder=5)
        from decimal import ROUND_HALF_UP, Decimal
        rank_text = str(Decimal(repr(round(float(r[idx]), 10))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))   # half up, as the tables (1.625 -> 1.63)
        label = f"{names[idx]}  {rank_text}" if left else f"{rank_text}  {names[idx]}"
        ax.text(x_text - (0.08 if left else -0.08), y, label, ha="right" if left else "left", va="center", fontsize=fs, color=S.INK)
    S.bare(ax)


def bump(ax, names, rank_a, rank_b, colors, labels=("A", "B"), fs=5.4, lw=1.0):
    """Rank under A (left) against rank under B (right); rank 1 at the top."""
    ra, rb = np.asarray(rank_a, float), np.asarray(rank_b, float)
    k = len(names)
    for i in range(k):
        moved = abs(ra[i] - rb[i]) >= 1
        ax.plot([0, 1], [ra[i], rb[i]], color=colors[i], lw=lw, alpha=0.95 if moved else 0.55, solid_capstyle="round", zorder=2)
        ax.scatter([0, 1], [ra[i], rb[i]], s=14, color=colors[i], edgecolors=S.PAPER, linewidths=0.4, zorder=3)
        ax.text(-0.07, ra[i], f"{names[i]}", ha="right", va="center", fontsize=fs)
        ax.text(1.07, rb[i], f"{names[i]}", ha="left", va="center", fontsize=fs)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(k + 0.6, 0.4)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels)
    ax.tick_params(length=0)
    ax.xaxis.tick_top()
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def dumbbell(ax, rows, series, colors, labels, markers=None, fs=5.6, xlim=None, annotate=None, size=20):
    """rows: row labels. series: list of arrays (one per marker type), NaN allowed."""
    y = np.arange(len(rows))
    arr = np.array(series, dtype=float)
    for i in range(len(rows)):
        vals = arr[:, i][np.isfinite(arr[:, i])]
        if len(vals) > 1:
            ax.plot([vals.min(), vals.max()], [y[i], y[i]], color=S.AXIS, lw=1.2, zorder=1, solid_capstyle="round")
    markers = markers or ["o"] * len(series)
    for s, c, lab, mk in zip(arr, colors, labels, markers):
        ax.scatter(s, y, s=size, color=c, marker=mk, edgecolors=S.PAPER, linewidths=0.5, zorder=3, label=lab)
    ax.set_yticks(y)
    ax.set_yticklabels(rows, fontsize=fs)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    if xlim:
        ax.set_xlim(*xlim)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    S.light_grid(ax, "x")
    if annotate is not None:
        for i, t in enumerate(annotate):
            if t:
                ax.text(ax.get_xlim()[1], y[i], t, ha="right", va="center", fontsize=fs - 0.4, color=S.INK2)


def funnel(ax, labels, values, colors, notes=None, fs=5.8, width_scale="sqrt", value_fmt="{:,}"):
    """Centred bars that shrink from top to bottom; value inside, label on the left."""
    v = np.asarray(values, dtype=float)
    w = np.sqrt(v / v.max()) if width_scale == "sqrt" else v / v.max()
    n = len(v)
    for i in range(n):
        ax.add_patch(FancyBboxPatch((-w[i] / 2, i + 0.14), w[i], 0.72, boxstyle="round,pad=0,rounding_size=0.08",
                                    fc=colors[i], ec="none", zorder=2))
        if i + 1 < n:
            ax.fill([-w[i] / 2, w[i] / 2, w[i + 1] / 2, -w[i + 1] / 2], [i + 0.86, i + 0.86, i + 1.14, i + 1.14],
                    color=colors[i], alpha=0.18, lw=0, zorder=1)
        rgba = __import__("matplotlib.colors", fromlist=["to_rgba"]).to_rgba(colors[i])
        inside = w[i] > 0.26
        ax.text(0 if inside else w[i] / 2 + 0.03, i + 0.5, value_fmt.format(int(v[i])), ha="center" if inside else "left",
                va="center", fontsize=fs + 0.4, fontweight="bold", color=S.text_on(rgba) if inside else S.INK)
        ax.text(-0.56, i + 0.5 - (0.13 if notes else 0), labels[i], ha="right", va="center", fontsize=fs, color=S.INK)
        if notes and notes[i]:
            ax.text(-0.56, i + 0.5 + 0.2, notes[i], ha="right", va="center", fontsize=fs - 1.2, color=S.MUTED)
    ax.set_xlim(-0.58, 0.58)
    ax.set_ylim(n, 0)
    S.bare(ax)


def colorbar(fig, rect, cmap, vmin, vmax, label, ticks=None, fs=5.2, orientation="horizontal"):
    """Slim colour bar in figure coordinates rect = [x, y, w, h]."""
    import matplotlib as mpl

    cax = fig.add_axes(rect)
    cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=mpl.colors.Normalize(vmin, vmax), orientation=orientation)
    cb.outline.set_visible(False)
    if ticks is not None:
        cb.set_ticks(ticks)
    cb.ax.tick_params(labelsize=fs, length=1.5, width=0.4, pad=1.5)
    cax.set_title(label, fontsize=fs + 0.3, pad=2.5, loc="center" if orientation == "horizontal" else "left")
    return cax


def header_band(ax, spans, y=1.02, h=0.045, fs=5.4):
    """Coloured bands above a matrix. spans: list of (x0, x1, text, facecolor) in data x."""
    trans = ax.get_xaxis_transform()
    for x0, x1, text, fc in spans:
        ax.add_patch(Rectangle((x0 + 0.04, y), x1 - x0 - 0.08, h, transform=trans, fc=fc, ec="none", clip_on=False))
        rgba = __import__("matplotlib.colors", fromlist=["to_rgba"]).to_rgba(fc)
        ax.text((x0 + x1) / 2, y + h / 2, text, transform=trans, ha="center", va="center", fontsize=fs, color=S.text_on(rgba))


def family_legend(ax_or_fig, families, loc="lower center", ncol=4, bbox=None, fs=5.8, marker="s"):
    from matplotlib.lines import Line2D

    handles = [Line2D([], [], marker=marker, ls="", ms=4.2, color=S.FAMILY[f], label=S.FAMILY_LABEL[f]) for f in families]
    return ax_or_fig.legend(handles=handles, loc=loc, ncol=ncol, bbox_to_anchor=bbox, fontsize=fs, columnspacing=1.2, handletextpad=0.3)
