"""bench/scripts/label_style_audit.py on a toy prepared source (figure 6a's counts and sizes; docs/ops/21 item 90)."""

import json

import numpy as np
from PIL import Image

from bench.scripts import label_style_audit as audit


def _disk(shape, cy, cx, r):
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r


def test_audit_counts_components_and_measures_them_against_the_fov(tmp_path):
    shape = (200, 300)
    data = tmp_path / "Toy"
    fov = _disk(shape, 100, 150, 90)                     # FOV box long side 181 (2 * 90 + 1)
    for image_id in ("a", "b", "c"):
        (data / "fov").mkdir(parents=True, exist_ok=True)
        Image.fromarray(fov.astype(np.uint8) * 255).save(data / "fov" / f"{image_id}.png")
    masks = {
        "a": {"MA": _disk(shape, 80, 120, 3) | _disk(shape, 120, 180, 3), "EX": _disk(shape, 100, 150, 10)},
        "b": {"MA": _disk(shape, 90, 140, 3)},
        "c": {},                                          # no lesion at all
    }
    for image_id, per_class in masks.items():
        for cls, mask in per_class.items():
            (data / "masks" / cls).mkdir(parents=True, exist_ok=True)
            Image.fromarray(mask.astype(np.uint8) * 255).save(data / "masks" / cls / f"{image_id}.png")
    # an ignore region is not a lesion: a grey band that would otherwise be one more HE component
    (data / "masks" / "HE").mkdir(parents=True, exist_ok=True)
    band = np.zeros(shape, bool)
    band[50:60, 100:140] = True
    Image.fromarray(band.astype(np.uint8) * 255).save(data / "masks" / "HE" / "a.png")
    (data / "ignore" / "HE").mkdir(parents=True, exist_ok=True)
    Image.fromarray(band.astype(np.uint8) * 255).save(data / "ignore" / "HE" / "a.png")
    (data / "splits.json").write_text(json.dumps({"train": ["a", "b"], "val": ["b"], "test": ["c"]}))

    result = audit.audit_source(data, log=lambda *a, **k: None)
    assert result["n_images"] == 3 and result["splits"] == {"train": 2, "val": 1, "test": 1}, "every id once"
    ma = result["per_class"]["MA"]
    assert ma["n_components"] == 3 and ma["images_positive"] == 2
    assert ma["per_image_mean"] == 1.0 and ma["per_image_mean_positive"] == 1.5
    # a radius-3 disk has 29 pixels: equivalent diameter 2 sqrt(29 / pi) = 6.08, over the FOV long side 181
    assert abs(ma["rel_diameter"]["p50"] - 2 * np.sqrt(29 / np.pi) / 181) < 1e-6
    ex = result["per_class"]["EX"]
    assert ex["n_components"] == 1 and ex["rel_diameter"]["p50"] > ma["rel_diameter"]["p50"]
    assert result["per_class"]["HE"]["n_components"] == 0 and result["per_class"]["HE"]["rel_diameter"] is None
    assert result["per_class"]["SE"]["n_components"] == 0 and result["per_class"]["SE"]["per_image_mean"] == 0.0
