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
    margin = int(getattr(card, "output_margin", 0) or 0)
    # A network with an output margin predicts the (win - 2m) centre of each window: the canvas is padded
    # by m on every side, the windows step over the centres, and each centre lands on the unpadded canvas.
    out_h, out_w = win_h - 2 * margin, win_w - 2 * margin
    full_h, full_w = max(height, out_h), max(width, out_w)
    padded = torch.nn.functional.pad(tensor, (margin, full_w - width + margin, margin, full_h - height + margin), value=0.0)
    stride_h = max(1, int(round(out_h * (1.0 - float(overlap)))))
    stride_w = max(1, int(round(out_w * (1.0 - float(overlap)))))
    boxes = [
        (top, left)
        for top in window_starts(full_h, out_h, stride_h)
        for left in window_starts(full_w, out_w, stride_w)
    ]
    total = None
    count = torch.zeros((1, full_h, full_w), device=tensor.device, dtype=torch.float32)
    for start in range(0, len(boxes), int(batch)):
        chunk = boxes[start : start + int(batch)]
        crops = torch.stack([padded[:, top : top + win_h, left : left + win_w] for top, left in chunk])
        out = model(crops).float()
        if out.shape[-2:] != (out_h, out_w):
            raise RuntimeError(f"{card.name}: window {win_h}x{win_w} with margin {margin} must give {out_h}x{out_w} logits, got {tuple(out.shape[-2:])}")
        if total is None:
            total = torch.zeros((out.shape[1], full_h, full_w), device=tensor.device, dtype=torch.float32)
        for (top, left), logits in zip(chunk, out):
            total[:, top : top + out_h, left : left + out_w] += logits
            count[:, top : top + out_h, left : left + out_w] += 1.0
    return (total / count)[:, :height, :width]
