import numpy as np
from PIL import Image, ImageDraw

from bench.scripts import leak_audit


def _fundus(seed, size=400):
    rng = np.random.default_rng(seed)
    image = Image.new("RGB", (size, size), (0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.ellipse((10, 10, size - 10, size - 10), fill=(170, 90, 40))
    for _ in range(14):
        points = [tuple(int(v) for v in rng.integers(60, size - 60, 2)) for _ in range(4)]
        draw.line(points, fill=(110, 40, 20), width=int(rng.integers(2, 6)))
    fov = np.zeros((size, size), bool)
    yy, xx = np.ogrid[:size, :size]
    fov[(yy - size / 2) ** 2 + (xx - size / 2) ** 2 <= (size / 2 - 10) ** 2] = True
    return image, fov


def _score(a, b):
    pa, _ = leak_audit.vessel_signature(*a)
    pb, mb = leak_audit.vessel_signature(*b)
    return max(float(pa @ pb), float(pa @ mb))


def test_vessel_signature_finds_the_same_eye_and_its_mirror():
    eye, fov = _fundus(1)
    other, other_fov = _fundus(2)
    brighter = Image.fromarray(np.clip(np.asarray(eye, dtype=np.float32) * 1.3 + 10, 0, 255).astype(np.uint8))
    mirror = eye.transpose(Image.FLIP_LEFT_RIGHT)
    assert _score((eye, fov), (brighter, fov)) > 0.9
    assert _score((eye, fov), (mirror, fov[:, ::-1])) > 0.9
    assert _score((eye, fov), (other, other_fov)) < 0.4
