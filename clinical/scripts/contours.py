"""Outlines of binary masks as closed polygons on pixel corners (no scikit-image or OpenCV needed).

A pixel (row r, column c) covers the square [c, c+1] x [r, r+1] in image coordinates (x to the right, y down),
the same convention as drawing the image on an HTML canvas, so the reader tool draws the polygons exactly over
the pixels they enclose. Every boundary edge between an inside and an outside pixel is a directed unit segment
with the inside on its right; the edges are chained into closed loops (outer boundaries clockwise on screen,
holes counter-clockwise). Where two inside pixels touch only at a corner the walk turns right, so the two
pixels get separate loops that share the corner. Straight runs are merged and long loops are simplified with
Ramer-Douglas-Peucker at a tolerance below one pixel.
"""

import numpy as np

# direction codes: 0 = +x, 1 = +y, 2 = -x, 3 = -y (screen coordinates, y down)
_DX = (1, 0, -1, 0)
_DY = (0, 1, 0, -1)


def _edges(mask):
    m = np.pad(np.asarray(mask, dtype=bool), 1)
    starts, dirs = [], []
    # Horizontal grid lines. above/below have shape (H+1, W+2): index (i, j) is the line y = i (unpadded)
    # under padded row i, at padded column j, i.e. the segment x = j-1 .. j.
    above, below = m[:-1, :], m[1:, :]
    i, j = np.nonzero(below & ~above)           # top edge of an inside pixel: walk +x from (j-1, i)
    starts.append(np.stack([j - 1, i], 1)); dirs.append(np.zeros(i.size, np.int8))
    i, j = np.nonzero(above & ~below)           # bottom edge of an inside pixel: walk -x from (j, i)
    starts.append(np.stack([j, i], 1)); dirs.append(np.full(i.size, 2, np.int8))
    # Vertical grid lines. left/right have shape (H+2, W+1): index (i, j) is the line x = j (unpadded)
    # beside padded row i, i.e. the segment y = i-1 .. i.
    left, right = m[:, :-1], m[:, 1:]
    i, j = np.nonzero(right & ~left)            # left edge of an inside pixel: walk -y from (j, i)
    starts.append(np.stack([j, i], 1)); dirs.append(np.full(i.size, 3, np.int8))
    i, j = np.nonzero(left & ~right)            # right edge of an inside pixel: walk +y from (j, i-1)
    starts.append(np.stack([j, i - 1], 1)); dirs.append(np.ones(i.size, np.int8))
    pts = np.concatenate(starts).astype(np.int64)
    return pts, np.concatenate(dirs)


def _chain(pts, dirs):
    """Closed loops of vertex coordinates from directed unit edges."""
    if pts.size == 0:
        return []
    W = int(pts[:, 0].max()) + 3
    key = (pts[:, 1] + 1) * W + (pts[:, 0] + 1)
    out_edges = {}
    for i, k in enumerate(key.tolist()):
        out_edges.setdefault(k, []).append(i)
    used = np.zeros(len(dirs), dtype=bool)
    loops = []
    for first in range(len(dirs)):
        if used[first]:
            continue
        loop = []
        e = first
        while True:
            used[e] = True
            x, y, d = int(pts[e, 0]), int(pts[e, 1]), int(dirs[e])
            loop.append((x, y))
            nx, ny = x + _DX[d], y + _DY[d]
            cands = [j for j in out_edges.get((ny + 1) * W + (nx + 1), ()) if not used[j]]
            if not cands:
                break
            if len(cands) > 1:   # corner-touching pixels: prefer the right turn, then straight, then left
                pref = {(d + 1) % 4: 0, d: 1, (d + 3) % 4: 2}
                cands.sort(key=lambda j: pref.get(int(dirs[j]), 3))
            e = cands[0]
        loops.append(loop)
    return loops


def _merge_straight(loop):
    n = len(loop)
    if n < 4:
        return loop
    keep = []
    for i in range(n):
        x0, y0 = loop[i - 1]
        x1, y1 = loop[i]
        x2, y2 = loop[(i + 1) % n]
        if (x1 - x0) * (y2 - y1) - (y1 - y0) * (x2 - x1) != 0:
            keep.append(loop[i])
    return keep if len(keep) >= 3 else loop


def _rdp(points, eps):
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 3:
        return pts
    keep = np.zeros(len(pts), dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        a, b = stack.pop()
        if b <= a + 1:
            continue
        seg = pts[b] - pts[a]
        norm = np.hypot(*seg)
        rel = pts[a + 1:b] - pts[a]
        if norm == 0:
            dist = np.hypot(rel[:, 0], rel[:, 1])
        else:
            dist = np.abs(seg[0] * rel[:, 1] - seg[1] * rel[:, 0]) / norm
        i = int(np.argmax(dist))
        if dist[i] > eps:
            keep[a + 1 + i] = True
            stack.append((a, a + 1 + i))
            stack.append((a + 1 + i, b))
    return pts[keep]


def simplify_loop(loop, eps=0.7, min_points=24):
    loop = _merge_straight(loop)
    if eps <= 0 or len(loop) < min_points:
        return [tuple(map(float, p)) for p in loop]
    pts = np.asarray(loop, dtype=np.float64)
    far = int(np.argmax(np.hypot(pts[:, 0] - pts[0, 0], pts[:, 1] - pts[0, 1])))
    first = _rdp(np.vstack([pts[: far + 1]]), eps)
    second = _rdp(np.vstack([pts[far:], pts[:1]]), eps)
    merged = np.vstack([first[:-1], second[:-1]])
    if len(merged) < 3:
        return [tuple(map(float, p)) for p in loop]
    return [tuple(map(float, p)) for p in merged]


def mask_loops(mask, eps=0.7, offset=(0, 0), scale=1.0):
    """Closed outlines of a boolean mask as lists of (x, y) pixel-corner coordinates.

    offset is added before scaling (use it when the mask is a crop of a larger image); scale maps the result to
    a resized display image.
    """
    pts, dirs = _edges(mask)
    loops = []
    for loop in _chain(pts, dirs):
        simple = simplify_loop(loop, eps=eps)
        loops.append([((x + offset[0]) * scale, (y + offset[1]) * scale) for x, y in simple])
    return loops


def flat(loop, ndigits=1):
    """[x0, y0, x1, y1, ...] rounded for the pack file; integers stay integers."""
    out = []
    for x, y in loop:
        for v in (x, y):
            r = round(float(v), ndigits)
            out.append(int(r) if r == int(r) else r)
    return out


def polygon_area(loop):
    pts = np.asarray(loop, dtype=np.float64)
    x, y = pts[:, 0], pts[:, 1]
    return float(0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
