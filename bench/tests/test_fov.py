import numpy as np

from bench.data.fov import crop_box, field_of_view
from bench.data.lesion_size import freeze_edges, relative_diameters


def test_deep_outside_ignores_a_one_pixel_rim():
    from bench.data.fov import lesion_outside

    fov = np.zeros((20, 20), dtype=bool)
    fov[2:18, 2:18] = True
    mask = np.zeros_like(fov)
    mask[1:18, 2:18] = True
    lesion, outside, deep = lesion_outside(mask, fov, margin=2)
    assert lesion == int(mask.sum())
    assert outside > 0
    assert deep == 0


def test_fov_keeps_a_dim_half_of_the_retina():
    image = np.zeros((80, 120, 3), dtype=np.uint8)
    yy, xx = np.ogrid[:80, :120]
    disk = (yy - 40) ** 2 + (xx - 60) ** 2 <= 34**2
    image[disk] = (40, 30, 20)
    image[disk & (xx < 60)] = (220, 180, 160)
    fov = field_of_view(image)
    assert fov[40, 30]
    assert fov[40, 90]
    assert not fov[0, 0]


def test_fov_keeps_the_bright_disk_and_drops_the_black_border():
    image = np.zeros((80, 100, 3), dtype=np.uint8)
    yy, xx = np.ogrid[:80, :100]
    disk = (yy - 40) ** 2 + (xx - 50) ** 2 <= 30**2
    image[disk] = (180, 40, 40)
    fov = field_of_view(image)
    assert fov[40, 50]
    assert not fov[0, 0]
    assert fov.sum() > 2000
    y0, y1, x0, x1 = crop_box(fov)
    assert y0 > 0 and x0 > 0 and y1 < 80 and x1 < 100


def test_size_edges_follow_the_training_medians():
    per_class = {
        "MA": {"n": 10, "p50": 0.0042},
        "HE": {"n": 10, "p50": 0.0214, "p75": 0.04},
    }
    edges, source = freeze_edges(per_class)
    assert edges == [0.0, 0.005, 0.022, float("inf")]
    assert "MA median" in source


def test_relative_diameter_is_scale_free():
    mask = np.zeros((40, 40), dtype=bool)
    mask[10:14, 10:14] = True
    fov = np.ones((40, 40), dtype=bool)
    small = relative_diameters(mask, fov)
    mask2 = np.zeros((80, 80), dtype=bool)
    mask2[20:28, 20:28] = True
    large = relative_diameters(mask2, np.ones((80, 80), dtype=bool))
    assert small.shape == (1,)
    assert abs(float(small[0]) - float(large[0])) < 1e-6
