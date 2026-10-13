"""Tier a, SAM 3 text prompts (design/15 section 5.4; e5_frozen.yaml ``zero_shot.sam3``).

    python -m bench.e5 sam3 --set DDR --split val                   # the three phrasings of every class, G1 windows
    python -m bench.e5 sam3-select                                  # best phrasing per class on DDR val -> YAML to append
    python -m bench.e5 sam3-compose --set IDRiD [--split test]      # the frozen selection -> sam3/selected/<set>_<split>

Three phrasings per class are frozen. Every phrasing runs on every image: the
pixel score is the maximum over the 200 queries of (query score x mask
probability) with the processor's confidence threshold at 0 so that the PR
curve is not cut. Windows of 1,008 with overlap 0.5 on the canvas, merged by
maximum. The main phrasing is chosen per class on DDR val by AUPR and written
to e5_frozen.yaml (``selected``) before any test image is run; all three are
reported (wording sensitivity, figure 5c).

SAM 3 needs its own environment (Python >= 3.12, PyTorch >= 2.7, CUDA >= 12.6;
retiseg-sam3) and the gated weights pretrained/sam3.pt. The API below follows
official_code/sam3 at the pinned commit as read in design/15 (sam3/model_builder.py,
sam3/model/sam3_image_processor.py); the first server run checks the names
in ``Sam3Adapter`` against that commit (17 section 6) and records them in done.json.

Output: runs/E5/sam3/phrase<i>/<set>_<split>/{<CLS>/<id>.png, paper.json (variant_by_class text:<phrase>), done.json};
sam3/selected/<set>_<split>/ with variant text:selected.
"""

import shutil
import time

import numpy as np
from PIL import Image

from bench.common.io import LESION_CLASSES
from bench.e5 import common


class Sam3Adapter:
    """Text-prompted SAM 3 image model; one call per (window, phrase) -> canvas-window probability map."""

    name = "sam3"
    window = common.WINDOW["sam3"]

    def __init__(self, checkpoint="sam3.pt", confidence_threshold=0.0):
        import torch

        from bench.models.foundation import official, weights

        self.confidence_threshold = float(confidence_threshold)
        with official("sam3", purge=("sam3",)):
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor

            self.model = build_sam3_image_model(checkpoint_path=str(weights(checkpoint)))
            self.model = self.model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
            self.processor = Sam3Processor(self.model, confidence_threshold=self.confidence_threshold)
        self.api = {"set_image": "set_image", "set_text": None, "masks": "masks_logits", "scores": "scores"}
        self._state = None

    def set_image(self, rgb):
        # the processor reads the size of a numpy input wrongly (sam3_image_processor.py:49-50): always a PIL image
        self._state = self.processor.set_image(Image.fromarray(np.ascontiguousarray(rgb, dtype=np.uint8)))

    def text_map(self, phrase):
        """Pixelwise max over queries of (query score x mask probability), window resolution, float32 in [0, 1]."""
        import torch

        state = self._state
        for name in ("set_text_prompt", "set_text", "add_text_prompt"):
            fn = getattr(self.processor, name, None)
            if fn is not None:
                self.api["set_text"] = name
                out = fn(phrase, state) if name != "add_text_prompt" else fn(state, phrase)
                state = out if isinstance(out, dict) else state
                break
        else:
            raise AttributeError("Sam3Processor has none of set_text_prompt/set_text/add_text_prompt; check official_code/sam3")
        probs = state.get("masks_logits")
        scores = state.get("scores")
        if probs is None or scores is None:
            raise KeyError(f"SAM 3 state keys {sorted(state)}; expected masks_logits and scores (sam3_image_processor.py:195-219)")
        probs = torch.as_tensor(probs).float()
        scores = torch.as_tensor(scores).float().reshape(-1)
        if probs.ndim == 4:
            probs = probs[:, 0]
        if probs.shape[0] == 0:
            return np.zeros(probs.shape[-2:], dtype=np.float32)
        if float(probs.min()) < 0.0 or float(probs.max()) > 1.0:
            probs = torch.sigmoid(probs)   # logits after all; the design read them as probabilities, record it
            self.api["masks"] = "masks_logits(sigmoid applied)"
        weighted = probs * scores[:, None, None]
        return weighted.max(dim=0).values.cpu().numpy().astype(np.float32)


def phrases(frozen):
    return {cls: list(v) for cls, v in frozen["zero_shot"]["sam3"]["phrases"].items()}


def out_dir(phrase_index, test_set, split):
    return common.OUT_ROOT / "sam3" / f"phrase{phrase_index}" / common.set_dirname(test_set, split)


def run(test_set, split="test", geometry="G1", limit=None, adapter=None):
    frozen = common.load_frozen()
    table = phrases(frozen)
    n_phr = len(next(iter(table.values())))
    dirs = {i: out_dir(i, test_set, split) for i in range(n_phr)}
    pending = [i for i in dirs if not common.is_done(dirs[i])]
    if not pending:
        print("sam3: every phrasing done", flush=True)
        return dirs
    ids = common.split_ids(test_set, split)
    if limit:
        ids = ids[: int(limit)]
    if adapter is None:
        adapter = Sam3Adapter(frozen["zero_shot"]["sam3"]["checkpoint"], frozen["zero_shot"]["sam3"]["confidence_threshold"])
    started = time.perf_counter()
    for image_id in ids:
        if all((dirs[i] / "SE" / f"{image_id}.png").is_file() for i in pending):
            continue
        canvas = common.load_canvas(test_set, image_id, masks=False)
        image = canvas["image"]
        canvas_hw = tuple(canvas["canvas_hw"])
        wins = common.windows(canvas_hw[0], canvas_hw[1], adapter.window) if geometry == "G1" else [(0, 0, *canvas_hw)]
        maps = {i: {cls: np.zeros(canvas_hw, dtype=np.float32) for cls in LESION_CLASSES} for i in pending}
        for top, left, h, w in wins:
            crop = common.window_crop(image, top, left, h, w) if geometry == "G1" else image
            adapter.set_image(crop)
            for cls in LESION_CLASSES:
                for i in pending:
                    common.paste_max(maps[i][cls], adapter.text_map(table[cls][i]), top, left)
        for i in pending:
            common.write_maps(dirs[i], image_id, canvas, maps[i])
        print(f"sam3 {test_set} {image_id}: {len(wins)} windows x {len(pending) * len(LESION_CLASSES)} phrases", flush=True)
    if limit:
        return dirs
    for i in pending:
        by_class = {cls: f"text:{table[cls][i]}" for cls in LESION_CLASSES}
        common.write_json(dirs[i] / "paper.json", common.paper_record("sam3", f"text:phrase{i}", test_set, by_class))
        common.write_json(common.done_marker(dirs[i]), {
            "experiment": "E5", "tier": "a", "model": "sam3", "phrase_index": i, "phrases": {cls: table[cls][i] for cls in LESION_CLASSES},
            "confidence_threshold": adapter.confidence_threshold, "pixel_score": "max over queries of score x mask probability",
            "api": adapter.api, "geometry": geometry, "window": adapter.window, "test_set": test_set, "split": split,
            "n_images": len(ids), "seconds": round(time.perf_counter() - started, 1), "environment": common.environment(),
        })
    return dirs


def select(split_source=common.THRESHOLD_SOURCE):
    """Best phrasing per class by AUPR on DDR val (metrics_test.json of every phrase directory) -> YAML to append."""
    frozen = common.load_frozen()
    table = phrases(frozen)
    test_set, split = split_source
    chosen = {}
    for cls in LESION_CLASSES:
        best = None
        for i, phrase in enumerate(table[cls]):
            path = out_dir(i, test_set, split) / "metrics_test.json"
            if not path.is_file():
                raise FileNotFoundError(f"{path}: evaluate the DDR val phrasing directories first (python -m bench.e5 evaluate --all)")
            aupr = common.read_json(path)["per_class"][cls]["aupr"]
            print(f"{cls:3s} {phrase:22s} AUPR {aupr:.4f}")
            if best is None or aupr > best[1]:
                best = (phrase, aupr)
        chosen[cls] = best[0]
    print("append to bench/configs/e5_frozen.yaml under zero_shot.sam3 (replace the empty selected: {}), then commit:")
    print("    selected: {" + ", ".join(f"{cls}: {phrase}" for cls, phrase in chosen.items()) + "}")
    return chosen


def compose(test_set, split="test"):
    """The frozen per-class selection -> sam3/selected/<set>_<split> (copies of the chosen phrase's class maps)."""
    frozen = common.load_frozen()
    selected = frozen["zero_shot"]["sam3"].get("selected") or {}
    if set(selected) != set(LESION_CLASSES):
        raise ValueError("e5_frozen.yaml zero_shot.sam3.selected is not filled for the four classes; run sam3-select on DDR val first")
    table = phrases(frozen)
    target = common.OUT_ROOT / "sam3" / "selected" / common.set_dirname(test_set, split)
    if common.is_done(target):
        return target
    for cls in LESION_CLASSES:
        index = table[cls].index(selected[cls])
        source = out_dir(index, test_set, split)
        if not common.is_done(source):
            raise FileNotFoundError(f"{source} is not finished")
        (target / cls).mkdir(parents=True, exist_ok=True)
        for png in (source / cls).glob("*.png"):
            shutil.copyfile(png, target / cls / png.name)
    common.write_json(target / "paper.json", common.paper_record("sam3", "text:selected", test_set))
    common.write_json(common.done_marker(target), {"experiment": "E5", "tier": "a", "model": "sam3", "selected": selected,
                                                   "test_set": test_set, "split": split, "composed_from": "phrase<i> directories"})
    return target
