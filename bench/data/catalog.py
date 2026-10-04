"""Local dataset inventory. Facts that need prose live in literature/synthesis/07_本地数据.txt."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetRecord:
    name: str
    role: str
    status: str
    path: str
    note: str


def dataset_root(repo_root):
    return Path(repo_root) / "dataset"


def catalog(repo_root):
    root = dataset_root(repo_root)
    return (
        DatasetRecord(
            "IDRiD",
            "fine",
            "ready",
            "dataset/iDRID/A. Segmentation",
            "81 images, official train 54 / test 27, pixel masks for MA HE EX SE",
        ),
        DatasetRecord(
            "DDR-grading",
            "grading",
            "not_for_lesion_segmentation",
            "dataset/DDR/archive.zip",
            "12524 grading images, no lesion masks and no 383/149/225 split",
        ),
        DatasetRecord(
            "DDR-lesion",
            "fine",
            "ready_split_zip",
            "dataset/DDR/OIA-DDR",
            "ten parts, one zip; segmentation 383/149/225; valid masks are in 'segmentation label'",
        ),
        DatasetRecord(
            "DIARETDB1",
            "coarse",
            "ready_regions",
            "dataset/DiaRetDB1 V2.1/archive/ddb1_v02_01",
            "89 images, 28/61 official split, four-expert polygons and circles",
        ),
        DatasetRecord(
            "DIARETDB0",
            "screening",
            "not_for_pixel_leaderboard",
            "dataset/DIARETDB1/archive (1)/diaretdb0_v_1_1",
            "folder name says DIARETDB1; the archive is DIARETDB0, 130 images",
        ),
        DatasetRecord(
            "MAPLES-DR",
            "fine",
            "absent",
            "",
            "labels via the maples-dr package; fundus images are MESSIDOR",
        ),
        DatasetRecord(
            "MESSIDOR",
            "grading",
            "absent",
            "",
            "needed only as the images underneath MAPLES-DR",
        ),
        DatasetRecord("e-ophtha", "fine_partial", "applied", "", "application sent 2026-10-03; MA and EX only"),
        DatasetRecord("FGADR", "coarse", "applied", "", "application sent 2026-10-03"),
        DatasetRecord("Retinal-Lesions", "coarse", "applied", "", "application sent 2026-10-03"),
        DatasetRecord(
            "TJDR",
            "fine_mixed_fov",
            "ready_zip",
            "dataset/TJDR",
            "three independent zips; train 448 / test 113; EX=1 HE=2 MA=3 SE=4; 3912px is 133-degree",
        ),
        DatasetRecord(
            "Refined-IDRiD",
            "fine_resized",
            "absent",
            "",
            "revised IDRiD labels at 1024; zenodo 17615903; does not replace official IDRiD",
        ),
    )


def require(repo_root, name):
    for record in catalog(repo_root):
        if record.name == name:
            if record.path and not (Path(repo_root) / record.path).exists():
                raise FileNotFoundError(record.path)
            return record
    raise KeyError(name)
