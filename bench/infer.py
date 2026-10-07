"""Network logits on one normalized canvas: whole canvas or overlapping windows.

``window`` None pads the canvas to card.pad_multiple and runs it once.
A ``window`` of (height, width) pads the canvas to at least that size, runs
windows on a grid with stride window * (1 - overlap), the last window flush
with the far edge, and averages the logits where windows overlap, as mmseg
slide inference does. The result is cut back to the canvas size.
"""

import torch

from bench.data.b1_input import pad_to_multiple


def window_starts(length, window, stride):
    if length <= window:
        return [0]
    starts = list(range(0, length - window + 1, stride))
    if starts[-1] != length - window:
        starts.append(length - window)
    return starts


@torch.no_grad()
def canvas_logits(model, tensor, card, window=None, overlap=0.5, batch=8):
    """``tensor`` is (C, H, W) on the model's device. Returns logits (K, H, W)."""
    height, width = tensor.shape[-2:]
    if window is None:
        padded = pad_to_multiple(tensor, card.pad_multiple).unsqueeze(0)
        return model(padded)[0, :, :height, :width]
    win_h, win_w = int(window[0]), int(window[1])
    if not 0.0 <= float(overlap) < 1.0:
        raise ValueError(f"overlap must be in [0, 1), got {overlap}")
    full_h, full_w = max(height, win_h), max(width, win_w)
    padded = torch.nn.functional.pad(tensor, (0, full_w - width, 0, full_h - height), value=0.0)
    stride_h = max(1, int(round(win_h * (1.0 - float(overlap)))))
    stride_w = max(1, int(round(win_w * (1.0 - float(overlap)))))
    boxes = [
        (top, left)
        for top in window_starts(full_h, win_h, stride_h)
        for left in window_starts(full_w, win_w, stride_w)
    ]
    total = None
    count = torch.zeros((1, full_h, full_w), device=tensor.device, dtype=torch.float32)
    for start in range(0, len(boxes), int(batch)):
        chunk = boxes[start : start + int(batch)]
        crops = torch.stack([padded[:, top : top + win_h, left : left + win_w] for top, left in chunk])
        out = model(crops).float()
        if total is None:
            total = torch.zeros((out.shape[1], full_h, full_w), device=tensor.device, dtype=torch.float32)
        for (top, left), logits in zip(chunk, out):
            total[:, top : top + win_h, left : left + win_w] += logits
            count[:, top : top + win_h, left : left + win_w] += 1.0
    return (total / count)[:, :height, :width]
