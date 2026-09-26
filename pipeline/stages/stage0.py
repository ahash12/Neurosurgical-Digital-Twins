from dataclasses import dataclass, field
from pathlib import Path
import subprocess

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to

from pipeline.stages.base import PipelineStage, register_stage
from utils import DEFAULT_DATA_ROOT, STAGE0_VERTEBRA_LABELS


DEFAULT_VERTEBRA_ROIS = [f"vertebrae_{name}" for name in STAGE0_VERTEBRA_LABELS]


@dataclass
class Stage0Config:
    output_root: str = "output/stage0"
    device: str = "gpu"
    rois: list[str] = field(default_factory=lambda: list(DEFAULT_VERTEBRA_ROIS))
    executable: str = "TotalSegmentator"
    ct_task: str = "total"
    mr_task: str = "vertebrae_mr"
    write_combined_segmentation: bool = True


@register_stage("stage0")
class Stage0Segmentation(PipelineStage):
    def __init__(self, config=None):
        self.config = config or Stage0Config()

    def output_dir(self, patient_id):
        return Path(self.config.output_root) / str(patient_id)

    def has_vertebra_masks(self, patient_id: str) -> bool:
        output_path = self.output_dir(patient_id)
        return any(
            (output_path / f"vertebrae_{vertebra}.nii.gz").is_file()
            for vertebra in STAGE0_VERTEBRA_LABELS
        )

    def build_command(self, input_path, output_path, modality="CT"):
        modality = str(modality).upper()
        if modality not in {"CT", "MR"}:
            raise ValueError(f"Unsupported modality: {modality}")
        task = self.config.ct_task if modality == "CT" else self.config.mr_task
        command = [
            self.config.executable,
            "-i",
            str(input_path),
            "-o",
            str(output_path),
            "--device",
            self.config.device,
            "--task",
            task,
        ]
        if modality == "CT":
            command.extend(["--roi_subset", *self.config.rois])
        return command

    def run_file(self, input_path, output_path, modality="CT"):
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        cmd = self.build_command(input_path, output_path, modality=modality)
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)
        return output_path

    @staticmethod
    def resolve_patient_image(patient_dir, patient_id):
        candidates = [
            (patient_dir / f"{patient_id}_ct.nii.gz", "CT"),
            (patient_dir / f"{patient_id}_mr.nii.gz", "MR"),
        ]
        matches = [(path, modality) for path, modality in candidates if path.exists()]
        if not matches:
            raise FileNotFoundError(f"CT or MR image not found in: {patient_dir}")
        if len(matches) > 1:
            raise ValueError(f"Multiple modality images found in: {patient_dir}")
        return matches[0]

    def combine_vertebra_masks(self, masks_dir, reference_path, output_path):
        reference = nib.as_closest_canonical(nib.load(str(reference_path)))
        combined = np.zeros(reference.shape, dtype=np.int16)
        found = []
        for vertebra, label in STAGE0_VERTEBRA_LABELS.items():
            mask_path = Path(masks_dir) / f"vertebrae_{vertebra}.nii.gz"
            if not mask_path.exists():
                continue
            mask = nib.as_closest_canonical(nib.load(str(mask_path)))
            if mask.shape != reference.shape or not np.allclose(
                mask.affine, reference.affine, atol=1e-5
            ):
                mask = resample_from_to(mask, reference, order=0)
            mask_data = np.asanyarray(mask.dataobj) > 0.5
            if not np.any(mask_data):
                continue
            overlap = mask_data & (combined != 0)
            if np.any(overlap):
                raise ValueError(
                    f"Overlapping vertebra masks in {masks_dir}: vertebrae_{vertebra}"
                )
            combined[mask_data] = label
            found.append(vertebra)
        if not found:
            raise RuntimeError(f"No vertebra masks found in: {masks_dir}")
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        header = reference.header.copy()
        header.set_data_dtype(np.int16)
        nib.save(nib.Nifti1Image(combined, reference.affine, header), str(output_path))
        return output_path, found

    def run_patient(self, patient_id, root_dir=DEFAULT_DATA_ROOT, skip_existing=False):
        patient_id = str(patient_id)
        patient_dir = Path(root_dir) / patient_id
        image_path, modality = self.resolve_patient_image(patient_dir, patient_id)
        output_path = self.output_dir(patient_id)
        seg_path = patient_dir / f"{patient_id}_seg-1.nii.gz"
        if skip_existing and self.has_vertebra_masks(patient_id):
            print(f"Reusing existing vertebra masks for {patient_id}; no model run.")
        elif skip_existing and seg_path.is_file():
            print(f"Reusing combined segmentation for {patient_id}; no model run.")
            return output_path
        else:
            self.run_file(image_path, output_path, modality=modality)
        if self.config.write_combined_segmentation:
            _, found = self.combine_vertebra_masks(output_path, image_path, seg_path)
            print(
                f"Combined {len(found)} vertebra masks ({', '.join(found)}): {seg_path}"
            )
        return output_path

    def run_directory(self, root_dir=DEFAULT_DATA_ROOT, skip_existing=False):
        root_dir = Path(root_dir)
        outputs = []
        for patient_dir in sorted(root_dir.iterdir()):
            if not patient_dir.is_dir():
                continue
            try:
                outputs.append(
                    self.run_patient(
                        patient_dir.name, root_dir=root_dir, skip_existing=skip_existing
                    )
                )
                print(f"Finished {patient_dir.name}")
            except subprocess.CalledProcessError:
                print(f"Error processing {patient_dir.name}")
        return outputs
