import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import numpy as np
import torch
from monai.networks.nets import resnet10
from monai.transforms import Resize
from torch import nn
from torch.utils.data import Dataset

from dataset_class import extract_fixed_physical_crop, load_aligned_ct_seg_with_spacing
from pipeline.scripts.stage3_postlateral import mri_sequence_family
from utils import STAGE0_VERTEBRA_LABELS, VERTEBRA_LABELS


ARCHITECTURE = "resnet10_ct_or_mr_binary_v1"
MODALITIES = ("CT", "MR")
CLASS_NAMES = ["none", "cancer"]


@dataclass(frozen=True)
class MultimodalPatchConfig:
    patch_size: tuple[int, int, int] = (96, 96, 64)
    physical_fov_mm: tuple[float, float, float] = (128.0, 128.0, 96.0)
    ct_window: tuple[float, float] = (-200.0, 1000.0)
    mri_percentiles: tuple[float, float] = (1.0, 99.0)

    def __post_init__(self) -> None:
        if len(self.patch_size) != 3 or any(
            int(v) != v or v < 16 for v in self.patch_size
        ):
            raise ValueError(
                "Patch size needs three integer dimensions of at least 16."
            )
        if len(self.physical_fov_mm) != 3 or any(
            not np.isfinite(v) or v <= 0 for v in self.physical_fov_mm
        ):
            raise ValueError(
                "Physical field of view needs three finite positive dimensions."
            )
        if (
            len(self.ct_window) != 2
            or not np.isfinite(sum(self.ct_window))
            or self.ct_window[0] >= self.ct_window[1]
        ):
            raise ValueError("CT window must contain finite ascending bounds.")
        if (
            len(self.mri_percentiles) != 2
            or not 0 <= self.mri_percentiles[0] < self.mri_percentiles[1] <= 100
        ):
            raise ValueError("MRI percentiles must be ascending within 0-100.")


@dataclass(frozen=True)
class VertebraSample:
    source: str
    patient_id: str
    group_id: str
    scan_id: str
    vertebra: str
    modality: str
    sequence: str
    label: int
    image_path: Path
    seg_path: Path


def load_ct_samples(
    csv_path: Path, root: Path, source: str, excluded_patients: set[str]
) -> list[VertebraSample]:
    samples = []
    with csv_path.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            patient = row["patient_id"].strip()
            if patient in excluded_patients:
                continue
            label = int(row["label"])
            if label not in range(4):
                raise ValueError(f"Invalid CT lesion label: {label}")
            group = row.get("source_group", "").strip() or patient
            samples.append(
                VertebraSample(
                    source,
                    patient,
                    row.get("group_id", "").strip() or f"{source}:{group}",
                    patient,
                    row["vertebra"].upper(),
                    "CT",
                    "",
                    int(label > 0),
                    root / patient / f"{patient}_ct.nii.gz",
                    root / patient / f"{patient}_seg-1.nii.gz",
                )
            )
    return samples


def load_mri_samples(csv_path: Path, root: Path) -> list[VertebraSample]:
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"MRI classification GT is required: {csv_path}. "
            "Run --write-label-template, then supply reviewed binary labels."
        )
    samples = []
    with csv_path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {
            "patient_id",
            "scan_id",
            "vertebra",
            "label",
            "sequence",
            "spine_coverage_reviewed",
        }
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"MRI label CSV requires columns: {sorted(required)}")
        for row in reader:
            if not row["label"].strip():
                continue
            label = int(row["label"])
            if label not in {0, 1}:
                raise ValueError(
                    "MRI labels must be 0=none or 1=cancer, not lesion subtypes."
                )
            if row["spine_coverage_reviewed"].strip().lower() not in {
                "true",
                "1",
                "yes",
            }:
                raise ValueError(
                    f"Review spine coverage for labeled scan: {row['scan_id']}"
                )
            sequence = row["sequence"].strip()
            if mri_sequence_family(sequence) in {"unknown", "unsupported"}:
                raise ValueError(f"Specify a supported MRI sequence: {row['scan_id']}")
            patient, scan_id = row["patient_id"].strip(), row["scan_id"].strip()
            if not patient or not scan_id:
                raise ValueError("Labeled MRI rows need patient_id and scan_id.")
            samples.append(
                VertebraSample(
                    "trial_b",
                    patient,
                    row.get("group_id", "").strip() or f"trial_b:{patient}",
                    scan_id,
                    row["vertebra"].upper(),
                    "MR",
                    sequence,
                    label,
                    root / scan_id / f"{scan_id}_mr.nii.gz",
                    root / scan_id / f"{scan_id}_seg-1.nii.gz",
                )
            )
    if not samples:
        raise ValueError(
            "No reviewed MRI classification labels; blank labels are not negatives."
        )
    return samples


def validate_samples(samples: list[VertebraSample]) -> None:
    seen = set()
    scan_groups: dict[tuple[str, str], str] = {}
    patient_groups: dict[tuple[str, str], str] = {}
    scan_sequences: dict[tuple[str, str], str] = {}
    for sample in samples:
        if sample.label not in {0, 1} or sample.modality not in MODALITIES:
            raise ValueError("Samples require binary labels and CT/MR modality.")
        patient_key = (sample.source, sample.patient_id)
        if (
            patient_key in patient_groups
            and patient_groups[patient_key] != sample.group_id
        ):
            raise ValueError(
                f"Conflicting group IDs across scans for patient {sample.patient_id}"
            )
        patient_groups[patient_key] = sample.group_id
        if sample.vertebra not in VERTEBRA_LABELS:
            raise ValueError(
                f"Training currently supports T1-L5 only: {sample.vertebra}"
            )
        key = (str(sample.image_path.resolve()), sample.vertebra)
        if key in seen:
            raise ValueError(
                f"Duplicate labeled scan/vertebra: {sample.scan_id}/{sample.vertebra}"
            )
        seen.add(key)
        scan_key = (sample.source, sample.scan_id)
        if scan_key in scan_groups and scan_groups[scan_key] != sample.group_id:
            raise ValueError(f"Conflicting patient groups for {sample.scan_id}")
        if scan_key in scan_sequences and scan_sequences[scan_key] != sample.sequence:
            raise ValueError(f"Conflicting sequence metadata for {sample.scan_id}")
        scan_groups[scan_key] = sample.group_id
        scan_sequences[scan_key] = sample.sequence
        for path in (sample.image_path, sample.seg_path):
            if not path.is_file():
                raise FileNotFoundError(
                    f"Labeled scan needs image and segmentation: {path}"
                )


def build_multimodal_patch(
    image: np.ndarray,
    seg: np.ndarray,
    spacing: tuple[float, float, float],
    vertebra: str,
    modality: str,
    config: MultimodalPatchConfig,
) -> torch.Tensor:
    if modality not in MODALITIES:
        raise ValueError(f"Unsupported modality: {modality}")
    if image.ndim != 3 or image.shape != seg.shape:
        raise ValueError("Expected aligned scalar 3D image and segmentation.")
    if any(value <= 0 or not np.isfinite(value) for value in spacing):
        raise ValueError("Voxel spacing must be finite and positive.")
    target = seg == VERTEBRA_LABELS[vertebra]
    coordinates = np.argwhere(target)
    if not coordinates.size:
        raise ValueError(f"Target vertebra absent from segmentation: {vertebra}")
    if modality == "CT":
        lower, upper = config.ct_window
        fill = -1000.0
    else:
        reference = np.isin(seg, list(STAGE0_VERTEBRA_LABELS.values()))
        values = image[reference & np.isfinite(image)]
        if not values.size:
            raise ValueError("MRI has no finite vertebral reference intensities.")
        lower, upper = np.percentile(values, config.mri_percentiles)
        fill = float(lower)
    if not np.isfinite(lower + upper) or upper <= lower:
        raise ValueError(f"Invalid intensity reference/window for {modality}.")
    center = (coordinates.min(axis=0) + coordinates.max(axis=0)) / 2.0
    crop = extract_fixed_physical_crop(
        image, center, spacing, config.physical_fov_mm, fill_value=fill
    )
    mask = extract_fixed_physical_crop(
        target.astype(np.float32),
        center,
        spacing,
        config.physical_fov_mm,
        fill_value=0.0,
    )
    normalized = np.clip(
        (np.nan_to_num(crop, nan=fill, posinf=fill, neginf=fill) - lower)
        / (upper - lower),
        0.0,
        1.0,
    )
    intensity = Resize(config.patch_size, mode="trilinear", align_corners=False)(
        torch.from_numpy(normalized.astype(np.float32)[None])
    ).as_subclass(torch.Tensor)
    resized_mask = Resize(config.patch_size, mode="nearest")(
        torch.from_numpy(mask[None])
    ).as_subclass(torch.Tensor)
    if not torch.any(resized_mask > 0.5):
        raise ValueError(f"Target vertebra vanished after resizing: {vertebra}")
    return torch.cat((intensity, (resized_mask > 0.5).float()), dim=0)


class MultimodalVertebraDataset(Dataset):
    def __init__(
        self,
        samples: list[VertebraSample],
        config: MultimodalPatchConfig,
        cache_dir: Path,
    ) -> None:
        self.samples = samples
        self.config = config
        self.cache_dir = cache_dir

    def __len__(self) -> int:
        return len(self.samples)

    def cache_path(self, sample: VertebraSample) -> Path:
        signature = {
            "version": ARCHITECTURE,
            "config": asdict(self.config),
            "modality": sample.modality,
            "sequence": sample.sequence,
            "vertebra": sample.vertebra,
            "files": [
                (str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns)
                for p in (sample.image_path, sample.seg_path)
            ],
        }
        digest = hashlib.sha256(
            json.dumps(signature, sort_keys=True).encode()
        ).hexdigest()
        return self.cache_dir / f"{digest}.npy"

    def precompute_cache(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        scans: dict[tuple[Path, Path], list[VertebraSample]] = {}
        for sample in self.samples:
            scans.setdefault((sample.image_path, sample.seg_path), []).append(sample)
        for index, ((image_path, seg_path), samples) in enumerate(
            scans.items(), start=1
        ):
            pending = [s for s in samples if not self.cache_path(s).is_file()]
            if pending:
                image, seg, spacing = load_aligned_ct_seg_with_spacing(
                    image_path, seg_path
                )
                for sample in pending:
                    patch = build_multimodal_patch(
                        image,
                        seg,
                        spacing,
                        sample.vertebra,
                        sample.modality,
                        self.config,
                    )
                    np.save(self.cache_path(sample), patch.numpy(), allow_pickle=False)
                del image, seg
            print(f"Patch cache {index}/{len(scans)} | built={len(pending)}")

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int, int]:
        sample = self.samples[index]
        patch = torch.from_numpy(np.load(self.cache_path(sample), allow_pickle=False))
        return patch, sample.label, MODALITIES.index(sample.modality), index


class ModalityGroupNorm(nn.Module):
    def __init__(self, original: nn.BatchNorm3d) -> None:
        super().__init__()
        channels = original.num_features
        groups = min(8, channels)
        while channels % groups:
            groups -= 1
        self.layers = nn.ModuleList(
            [nn.GroupNorm(groups, channels) for _ in MODALITIES]
        )
        self.domain = 0
        if original.affine:
            for layer in self.layers:
                normalizer = cast(nn.GroupNorm, layer)
                with torch.no_grad():
                    normalizer.weight.copy_(original.weight)
                    normalizer.bias.copy_(original.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers[self.domain](inputs)


def replace_normalization(module: nn.Module) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm3d):
            setattr(module, name, ModalityGroupNorm(child))
        else:
            replace_normalization(child)


class CTOrMRIResNet10(nn.Module):
    def __init__(self, pretrained: bool = False) -> None:
        super().__init__()
        self.backbone = resnet10(
            pretrained=pretrained,
            progress=True,
            spatial_dims=3,
            n_input_channels=1 if pretrained else 2,
            feed_forward=False,
            shortcut_type="B",
            bias_downsample=False,
        )
        if pretrained:
            original = self.backbone.conv1
            conv = nn.Conv3d(
                2,
                original.out_channels,
                cast(tuple[int, int, int], original.kernel_size),
                cast(tuple[int, int, int], original.stride),
                cast(tuple[int, int, int], original.padding),
                bias=False,
            )
            with torch.no_grad():
                conv.weight.zero_()
                conv.weight[:, :1].copy_(original.weight)
            self.backbone.conv1 = conv
        replace_normalization(self.backbone)
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(512, 2))

    def forward(self, inputs: torch.Tensor, domains: torch.Tensor) -> torch.Tensor:
        if (
            inputs.ndim != 5
            or inputs.shape[1] != 2
            or domains.shape != (inputs.shape[0],)
        ):
            raise ValueError("Expected [batch,2,x,y,z] and one modality ID per sample.")
        if torch.any((domains != 0) & (domains != 1)):
            raise ValueError("Modality IDs must be 0=CT or 1=MR.")
        outputs = inputs.new_zeros((inputs.shape[0], 2))
        # Each sample supplies one image, never a paired CT+MRI input.
        for domain in range(len(MODALITIES)):
            selected = domains == domain
            if not torch.any(selected):
                continue
            for layer in self.backbone.modules():
                if isinstance(layer, ModalityGroupNorm):
                    layer.domain = domain
            outputs[selected] = self.classifier(self.backbone(inputs[selected]))
        return outputs
