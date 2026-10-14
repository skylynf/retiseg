"""MAPLES-DR (Lepetit-Aimon et al., Scientific Data 2024): pixel labels for 198 MESSIDOR images.

The labels (figshare, doi 10.6084/m9.figshare.24328660) were drawn at
1500 x 1500 on the field-of-view crop; the fundus images are MESSIDOR's
(adcis.net licence, requested separately). The ``maples_dr`` package (pip)
joins the two and, with ``resize=False``, returns every mask at the original
MESSIDOR resolution (1440 x 960, 2240 x 1488 or 2304 x 1536), which is what
the prepared layout keeps. Official split: 138 train, 60 test; the two
duplicated training images stay out, as in the package's default subsets.

Classes: microaneurysms -> MA, hemorrhages -> HE, exudates -> EX, cotton wool
spots -> SE. The dataset also marks red lesions whose type was uncertain
(``redUncertains``: MA or HE) and bright ones (``brightUncertains``: EX, SE
or drusen); those pixels go to the ignore layer of the two classes they could
belong to, so a prediction there counts neither way. Drusen and
neovascularization go to masks_ext. The MA, HE, EX and vessel labels were
AI-pre-segmented and corrected by the retinologists; cotton wool spots were
drawn from scratch (paper, section Methods); meta.json records it.
"""

from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from bench.common.io import LESION_CLASSES

RAW = "dataset/maples-dr"       # the figshare archive (zip) or its unpacked directory; absent: the package downloads it
MESSIDOR = "dataset/messidor"   # Base11 ... Base34 directories or zips, as released by adcis.net
BIOMARKERS = {"MA": "microaneurysms", "HE": "hemorrhages", "EX": "exudates", "SE": "cottonWoolSpots"}
UNCERTAIN = {"MA": "redUncertains", "HE": "redUncertains", "EX": "brightUncertains", "SE": "brightUncertains"}
EXTRA = {"DRUSEN": "drusens", "NV": "neovascularization"}
AI_ASSISTED = ("MA", "HE", "EX")   # pre-segmented by a network, then corrected; SE drawn from scratch
SPLIT_RULE = "official MAPLES-DR train (138) and test (60) lists; the two duplicated training images are left out"
GRADES = ("dr", "me")


def case_from_sample(name, fundus, biomarker, grades=None):
    """One prepared case from a sample's masks.

    ``fundus`` is the RGB uint8 array at MESSIDOR resolution; ``biomarker(field)``
    returns that field's boolean mask at the same resolution (the package's
    read_biomarker with resize=False). Pure, so the mapping is testable without
    the package.
    """
    fundus = np.asarray(fundus)
    if fundus.ndim != 3 or fundus.shape[2] != 3 or fundus.dtype != np.uint8:
        raise ValueError(f"{name}: fundus must be RGB uint8, got {fundus.shape} {fundus.dtype}")
    shape = fundus.shape[:2]

    def read(field):
        mask = np.asarray(biomarker(field)).astype(bool)
        if mask.shape != shape:
            raise ValueError(f"{name} {field}: mask {mask.shape} != image {shape}")
        return mask

    masks = {cls: read(field) for cls, field in BIOMARKERS.items()}
    uncertain = {field: read(field) for field in sorted(set(UNCERTAIN.values()))}
    ignore = {cls: uncertain[UNCERTAIN[cls]] & ~masks[cls] for cls in LESION_CLASSES}
    extra = {cls: read(field) for cls, field in EXTRA.items()}
    buffer = BytesIO()
    Image.fromarray(fundus).save(buffer, format="PNG", optimize=False)
    stats = {}
    for key in GRADES:
        value = (grades or {}).get(key)
        if value:
            stats[key] = {str(value): 1}
    return {
        "bytes": buffer.getvalue(),
        "ext": "png",
        "image": fundus,
        "masks": masks,
        "binary": {"masks_ext": extra, "ignore": ignore},
        "stats": stats,
    }


class MaplesDR:
    """The dataset through the maples_dr package, configured for MESSIDOR-resolution RGB arrays."""

    extra_classes = tuple(EXTRA)
    granularity = "fine"

    def __init__(self, repo_root, raw=None, messidor=None):
        try:
            import maples_dr
        except ImportError as exc:  # pragma: no cover - needs the package
            raise ImportError("pip install maples_dr (the MAPLES-DR loader) in the retiseg environment") from exc
        root = Path(repo_root)
        raw = Path(raw) if raw is not None else root / RAW
        messidor = Path(messidor) if messidor is not None else root / MESSIDOR
        if not messidor.exists():
            raise FileNotFoundError(f"{messidor} is missing: the MESSIDOR images (Base11 ... Base34) come from adcis.net")
        maples_dr.configure(
            maples_dr_path=str(raw) if raw.exists() else None,
            messidor_path=str(messidor),
            cache=False,
            resize=False,
            image_format="rgb",
            preprocessing="none",
        )
        self._api = maples_dr
        self._train = maples_dr.load_train_set()
        self._test = maples_dr.load_test_set()
        self._samples = {}
        for subset in (self._train, self._test):
            for sample in subset:
                self._samples[sample.name] = sample

    def splits(self):
        train = [sample.name for sample in self._train]
        test = [sample.name for sample in self._test]
        if len(train) != 138 or len(test) != 60:
            raise RuntimeError(f"MAPLES-DR official split moved: {len(train)} train, {len(test)} test (expected 138, 60)")
        if set(train) & set(test):
            raise RuntimeError("MAPLES-DR train and test lists overlap")
        return {"train": train, "val": [], "test": test}

    def read_case(self, image_id):
        sample = self._samples[image_id]
        fundus = np.asarray(sample.read_fundus(preprocess=False, image_format="rgb", resize=False))
        grades = {}
        for key in GRADES:
            try:
                grades[key] = str(sample[key])
            except Exception:  # a sample without a grade field
                pass
        return case_from_sample(
            image_id,
            fundus,
            lambda field: sample.read_biomarker(field, image_format="rgb", resize=False),
            grades,
        )


def open_maples(repo_root):
    return MaplesDR(repo_root)
