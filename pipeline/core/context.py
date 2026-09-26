from dataclasses import dataclass
import csv
from pathlib import Path

import numpy as np

from utils import DEFAULT_DATA_ROOT, load_img


@dataclass
class PatientContext:
    patient_id: str
    ct: np.ndarray
    seg: np.ndarray
    affine: np.ndarray
    ct_path: Path
    seg_path: Path
    modality: str = "CT"
    sequence: str = ""

    @property
    def image(self) -> np.ndarray:
        return self.ct

    @property
    def image_path(self) -> Path:
        return self.ct_path

    @classmethod
    def load(cls, patient_id, root_dir=DEFAULT_DATA_ROOT, canonical=True):
        root = Path(root_dir)
        patient_id = str(patient_id)
        patient_dir = root / patient_id
        candidates = [
            (patient_dir / f"{patient_id}_ct.nii.gz", "CT"),
            (patient_dir / f"{patient_id}_mr.nii.gz", "MR"),
        ]
        matches = [(path, modality) for path, modality in candidates if path.exists()]
        seg_path = patient_dir / f"{patient_id}_seg-1.nii.gz"
        if not matches:
            raise FileNotFoundError(f"CT or MR image not found in: {patient_dir}")
        if len(matches) > 1:
            raise ValueError(f"Multiple modality images found in: {patient_dir}")
        if not seg_path.exists():
            raise FileNotFoundError(f"Segmentation not found: {seg_path}")

        image_path, modality = matches[0]
        image, affine = load_img(str(image_path), canonical=canonical)
        seg, seg_affine = load_img(str(seg_path), canonical=canonical)
        if image.shape != seg.shape or not np.allclose(affine, seg_affine, atol=1e-5):
            raise ValueError(
                f"Image and segmentation grids do not match for {patient_id}: "
                f"image={image.shape}, segmentation={seg.shape}"
            )
        return cls(
            patient_id=patient_id,
            ct=image,
            seg=seg,
            affine=affine,
            ct_path=image_path,
            seg_path=seg_path,
            modality=modality,
            sequence=_load_sequence_metadata(root, patient_id),
        )


def _load_sequence_metadata(root: Path, patient_id: str) -> str:
    manifest_path = root / "manifest.csv"
    if not manifest_path.exists():
        return ""
    with manifest_path.open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            if row.get("scan_id") == patient_id:
                return row.get("sequence", "")
    return ""
