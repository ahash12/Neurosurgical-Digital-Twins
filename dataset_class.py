import os
import re
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import xml.etree.ElementTree as ET

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from nibabel.processing import resample_from_to
from monai.transforms import Resize
from torch.utils.data import ConcatDataset, Dataset
from utils import (
    VERTEBRA_LABELS,
    VERTEBRAE,
    extract_centered_label_cube,
)


ARCHIVE_GT_WORKBOOK = "../../datasets/radiel/Archive/SINS Model Ground Truth.xlsx"
ARCHIVE_PREPARED_ROOT = "../../datasets/radiel/archive_nifti"
ARCHIVE_STAGE1_LABELS = "../../datasets/radiel/archive_stage1_labels.csv"
ARCHIVE_SHEET_PATTERN = re.compile(r"^PAT(\d+)$", re.IGNORECASE)
EXCEL_CELL_REF_PATTERN = re.compile(r"([A-Z]+)([0-9]+)")
EXCEL_NS = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
STAGE1_CACHE_VERSION = "aligned_v2"
PHYSICAL_CONTEXT_CACHE_VERSION = "physical_context_v1"
PATIENT_RELATIVE_CACHE_VERSION = "patient_relative_context_v1"


def resolve_dataset_root(default="../../datasets/radiel"):
    env_root = os.environ.get("RADIEL_DATA_ROOT")
    if env_root:
        return Path(env_root).expanduser()
    return Path(default).expanduser()


def extract_levels(text):
    if pd.isna(text):
        return []
    text = str(text)
    return re.findall(r"[TL]\d+", text)


def build_labels(row):
    labels = {v: 0 for v in VERTEBRAE}
    blastic = extract_levels(row["Blastic"])
    lytic = extract_levels(row["Lytic"])
    mixed = extract_levels(row["Mixed"])

    for v in blastic:
        if v in labels:
            labels[v] = 1
    for v in lytic:
        if v in labels:
            labels[v] = 2
    for v in mixed:
        if v in labels:
            labels[v] = 3
    return labels


def build_dataset_csv(metadata_path=None, output_path=None):
    data_root = resolve_dataset_root()
    metadata_path = metadata_path or data_root / "patient_metadata.csv"
    output_path = output_path or data_root / "vertebra_dataset.csv"
    df = pd.read_csv(metadata_path)
    rows = []

    for _, row in df.iterrows():
        pid = row["Case"]
        labels = build_labels(row)

        for v in VERTEBRAE:
            rows.append(
                {
                    "patient_id": pid,
                    "vertebra": v,
                    "label": labels[v],
                }
            )

    out = pd.DataFrame(rows, columns=["patient_id", "vertebra", "label"])
    out.to_csv(output_path, index=False)
    counts = dict(Counter(out["label"]))
    print(f"Saved: {output_path}")
    print(f"Label counts: {counts}")
    return out


def normalize_archive_patient_id(sheet_name):
    match = ARCHIVE_SHEET_PATTERN.match(str(sheet_name).strip())
    if match is None:
        raise ValueError(f"Unexpected Archive GT sheet name: {sheet_name}")
    return f"PAT_{int(match.group(1)):04d}"


def normalize_archive_lesion_type(value):
    text = str(value).strip().lower()
    if not text or "unable" in text or "?" in text:
        return None
    if "clear" in text or text == "0":
        return 0
    if text == "1":
        return 1
    if text == "2":
        return 2
    if text == "3":
        return 3
    return None


def excel_column_index(cell_ref):
    match = EXCEL_CELL_REF_PATTERN.match(str(cell_ref))
    if match is None:
        return 0
    index = 0
    for char in match.group(1):
        index = index * 26 + ord(char) - ord("A") + 1
    return index - 1


def read_xlsx_rows(path):
    archive = zipfile.ZipFile(path)
    shared_strings = []
    if "xl/sharedStrings.xml" in archive.namelist():
        root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
        for item in root.findall("a:si", EXCEL_NS):
            shared_strings.append(
                "".join(text.text or "" for text in item.findall(".//a:t", EXCEL_NS))
            )

    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    relmap = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels}
    for sheet in workbook.findall(".//a:sheet", EXCEL_NS):
        sheet_name = sheet.attrib["name"]
        rel_id = sheet.attrib[
            "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
        ]
        target = relmap[rel_id]
        if not target.startswith("xl/"):
            target = "xl/" + target
        worksheet = ET.fromstring(archive.read(target))
        rows = []
        for row in worksheet.findall(".//a:sheetData/a:row", EXCEL_NS):
            values = [""] * 8
            for cell in row.findall("a:c", EXCEL_NS):
                col_idx = excel_column_index(cell.attrib.get("r", "A1"))
                value_node = cell.find("a:v", EXCEL_NS)
                value = "" if value_node is None else value_node.text
                if cell.attrib.get("t") == "s" and value:
                    value = shared_strings[int(value)]
                if col_idx >= len(values):
                    values.extend([""] * (col_idx - len(values) + 1))
                values[col_idx] = value.strip() if isinstance(value, str) else value
            rows.append(values)
        yield sheet_name, rows


def default_archive_series_mapping(archive_root, require_segmentation=False):
    manifest_path = Path(archive_root) / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Archive manifest not found: {manifest_path}")
    manifest = pd.read_csv(manifest_path)
    manifest = manifest[
        (manifest["status"] == "converted")
        & (manifest["modality"] == "CT")
        & manifest["source_group"].astype(str).str.match(r"PAT_000[0-4]$")
    ].copy()
    if require_segmentation:
        manifest = manifest[
            manifest["scan_id"].astype(str).map(
                lambda scan_id: (
                    Path(archive_root) / scan_id / f"{scan_id}_seg-1.nii.gz"
                ).exists()
            )
        ]
    if manifest.empty:
        return {}
    manifest["slice_count"] = manifest["slice_count"].astype(int)
    manifest = manifest.sort_values(
        ["source_group", "slice_count", "scan_id"],
        ascending=[True, False, True],
    )
    return {
        row.source_group: row.scan_id
        for row in manifest.drop_duplicates("source_group").itertuples(index=False)
    }


def load_archive_series_mapping(
    mapping_path,
    archive_root,
    require_segmentation=False,
):
    if mapping_path is None:
        return default_archive_series_mapping(
            archive_root,
            require_segmentation=require_segmentation,
        )
    mapping_df = pd.read_csv(mapping_path)
    required = {"source_group", "scan_id"}
    missing = required.difference(mapping_df.columns)
    if missing:
        raise ValueError(f"Archive mapping missing columns: {sorted(missing)}")
    return {
        str(row.source_group): str(row.scan_id)
        for row in mapping_df.itertuples(index=False)
    }


def build_archive_stage1_labels(
    gt_path=ARCHIVE_GT_WORKBOOK,
    archive_root=ARCHIVE_PREPARED_ROOT,
    output_path=ARCHIVE_STAGE1_LABELS,
    mapping_path=None,
    require_segmentation=True,
):
    mapping = load_archive_series_mapping(
        mapping_path,
        archive_root,
        require_segmentation=require_segmentation,
    )
    rows = []
    skipped = Counter()
    archive_root = Path(archive_root)

    for sheet_name, sheet_rows in read_xlsx_rows(gt_path):
        source_group = normalize_archive_patient_id(sheet_name)
        scan_id = mapping.get(source_group)
        if scan_id is None:
            skipped["no_scan_mapping"] += 1
            continue
        if require_segmentation:
            seg_path = archive_root / scan_id / f"{scan_id}_seg-1.nii.gz"
            if not seg_path.exists():
                skipped["missing_segmentation"] += 1
                continue

        for row in sheet_rows[1:]:
            vertebra = str(row[0]).strip().upper()
            if vertebra not in VERTEBRA_LABELS:
                skipped["unsupported_vertebra"] += 1
                continue
            label = normalize_archive_lesion_type(row[2])
            if label is None:
                skipped["unsupported_label"] += 1
                continue
            rows.append(
                {
                    "patient_id": scan_id,
                    "vertebra": vertebra,
                    "label": label,
                    "source": "archive_gt",
                    "source_group": source_group,
                    "gt_sheet": sheet_name,
                }
            )

    out = pd.DataFrame(rows)
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(output_path, index=False)
        print(f"Saved: {output_path}")
    print(f"Archive Stage 1 labels: {len(out)} rows")
    print(f"Label counts: {dict(Counter(out['label'])) if len(out) else {}}")
    print(f"Skipped: {dict(skipped)}")
    return out


class Stage1PatchAugmentation:
    def __init__(
        self,
        flip_probability=0.5,
        noise_std=0.02,
        intensity_scale=0.08,
        intensity_shift=0.04,
    ):
        self.flip_probability = float(flip_probability)
        self.noise_std = float(noise_std)
        self.intensity_scale = float(intensity_scale)
        self.intensity_shift = float(intensity_shift)

    def __call__(self, patch):
        out = patch.clone()
        for dim in (1, 2):
            if torch.rand(1).item() < self.flip_probability:
                out = torch.flip(out, dims=(dim,))
        intensity_channels = out.shape[0] - 1 if out.shape[0] > 1 else 1
        intensities = out[:intensity_channels]
        foreground = intensities > 0
        if self.intensity_scale > 0:
            scale = 1.0 + (
                torch.rand(1).item() * 2.0 - 1.0
            ) * self.intensity_scale
            intensities[foreground] = intensities[foreground] * scale
        if self.intensity_shift > 0:
            shift = (torch.rand(1).item() * 2.0 - 1.0) * self.intensity_shift
            intensities[foreground] = intensities[foreground] + shift
        if self.noise_std > 0:
            noise = torch.randn_like(intensities) * self.noise_std
            intensities[foreground] = (
                intensities[foreground] + noise[foreground]
            )
        out[:intensity_channels] = torch.clamp(intensities, 0.0, 1.0)
        return out


class Stage1CTWindowAugmentation:
    def __init__(
        self,
        flip_probability=0.5,
        hu_shift=150.0,
        hu_scale=0.15,
        noise_std=0.01,
    ):
        self.flip_probability = float(flip_probability)
        self.hu_shift = max(float(hu_shift), 0.0)
        self.hu_scale = max(float(hu_scale), 0.0)
        self.noise_std = max(float(noise_std), 0.0)

    def __call__(self, patch):
        out = patch.clone()
        for dim in (1, 2):
            if torch.rand(1).item() < self.flip_probability:
                out = torch.flip(out, dims=(dim,))

        intensities = out[:2]
        foreground = torch.any(intensities > 0, dim=0)
        scale = 1.0 + (
            torch.rand(1).item() * 2.0 - 1.0
        ) * self.hu_scale
        shift = (
            torch.rand(1).item() * 2.0 - 1.0
        ) * self.hu_shift
        windows = ((-200.0, 1000.0), (-150.0, 400.0))
        for channel, (lower, upper) in enumerate(windows):
            width = upper - lower
            hu = intensities[channel] * width + lower
            hu[foreground] = hu[foreground] * scale + shift
            normalized = torch.clamp((hu - lower) / width, 0.0, 1.0)
            if self.noise_std > 0:
                noise = torch.randn_like(normalized) * self.noise_std
                normalized[foreground] += noise[foreground]
            normalized = torch.clamp(normalized, 0.0, 1.0)
            normalized[~foreground] = 0.0
            out[channel] = normalized
        return out


class TransformedDataset(Dataset):
    def __init__(self, base_dataset, transform=None):
        self.base = base_dataset
        self.transform = transform
        if hasattr(base_dataset, "df"):
            self.df = base_dataset.df

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        patch, label = self.base[idx]
        if self.transform is not None:
            patch = self.transform(patch)
        return patch, label


class CombinedVertebraDataset(ConcatDataset):
    def __init__(self, datasets):
        super().__init__(datasets)
        self.df = pd.concat(
            [dataset.df for dataset in datasets],
            ignore_index=True,
            sort=False,
        )


def load_aligned_ct_seg(
    ct_path: str | Path,
    seg_path: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    ct, seg, _ = load_aligned_ct_seg_with_spacing(ct_path, seg_path)
    return ct, seg


def load_aligned_ct_seg_with_spacing(
    ct_path: str | Path,
    seg_path: str | Path,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
    ct_img = nib.as_closest_canonical(nib.load(str(ct_path)))
    seg_img = nib.as_closest_canonical(nib.load(str(seg_path)))
    if seg_img.shape != ct_img.shape or not np.allclose(
        seg_img.affine,
        ct_img.affine,
        atol=1e-4,
    ):
        seg_img = resample_from_to(seg_img, ct_img, order=0)

    ct = np.asarray(ct_img.dataobj, dtype=np.float32)
    seg = np.rint(np.asarray(seg_img.dataobj)).astype(np.int16)
    spacing = tuple(float(value) for value in ct_img.header.get_zooms()[:3])
    return ct, seg, spacing


def stage1_cache_path(
    cache_dir,
    pid,
    vname,
    patch_size,
    norm_mode,
    zscore_scale,
    foreground_floor,
    patch_mode="isolated",
    physical_fov_mm=(128.0, 128.0, 96.0),
):
    sx, sy, sz = patch_size
    zscore_token = int(round(float(zscore_scale) * 100))
    foreground_token = int(round(float(foreground_floor) * 100))
    if patch_mode in {"physical_context", "patient_relative_context"}:
        fx, fy, fz = (int(round(float(value))) for value in physical_fov_mm)
        cache_version = (
            PHYSICAL_CONTEXT_CACHE_VERSION
            if patch_mode == "physical_context"
            else PATIENT_RELATIVE_CACHE_VERSION
        )
        return Path(cache_dir) / (
            f"{pid}_{vname}_{sx}x{sy}x{sz}_{cache_version}_"
            f"fov{fx}x{fy}x{fz}.npy"
        )
    return Path(cache_dir) / (
        f"{pid}_{vname}_{sx}x{sy}x{sz}_{STAGE1_CACHE_VERSION}_{norm_mode}_"
        f"zs{zscore_token}_fg{foreground_token}.npy"
    )


def build_patient_patch_cache(job):
    pid = str(job["patient_id"])
    root = Path(job["root"])
    cache_dir = Path(job["cache_dir"])
    patch_size = tuple(int(x) for x in job["patch_size"])
    norm_mode = str(job["norm_mode"])
    zscore_scale = max(float(job["zscore_scale"]), 1e-6)
    foreground_floor = float(np.clip(job["foreground_floor"], 0.0, 0.95))
    patch_mode = str(job["patch_mode"])
    physical_fov_mm = tuple(float(value) for value in job["physical_fov_mm"])
    force = bool(job["force"])
    vertebrae = [str(v) for v in job["vertebrae"]]

    patient_dir = root / pid
    ct_path = patient_dir / f"{pid}_ct.nii.gz"
    seg_path = patient_dir / f"{pid}_seg-1.nii.gz"
    ct, seg, spacing = load_aligned_ct_seg_with_spacing(ct_path, seg_path)
    patient_reference = compute_patient_vertebral_reference(ct, seg)

    built = 0
    skipped = 0
    sx, sy, sz = patch_size
    cache_dir.mkdir(parents=True, exist_ok=True)
    for vname in vertebrae:
        out_path = stage1_cache_path(
            cache_dir,
            pid,
            vname,
            (sx, sy, sz),
            norm_mode,
            zscore_scale,
            foreground_floor,
            patch_mode=patch_mode,
            physical_fov_mm=physical_fov_mm,
        )
        if out_path.exists() and not force:
            skipped += 1
            continue
        patch_np = build_stage1_patch_from_arrays(
            ct,
            seg,
            vname,
            patch_size=patch_size,
            norm_mode=norm_mode,
            zscore_scale=zscore_scale,
            foreground_floor=foreground_floor,
            spacing=spacing,
            patch_mode=patch_mode,
            physical_fov_mm=physical_fov_mm,
            patient_reference=patient_reference,
        )
        np.save(out_path, patch_np, allow_pickle=False)
        built += 1
    return {"patient_id": pid, "built": built, "skipped": skipped, "seen": len(vertebrae)}


def normalize_stage1_patch(
    ct_patch,
    mask,
    norm_mode="zscore_sigmoid",
    zscore_scale=1.5,
    foreground_floor=0.45,
):
    out = np.zeros_like(ct_patch, dtype=np.float32)
    if not np.any(mask):
        return out

    vals = ct_patch[mask]
    if norm_mode == "robust_minmax":
        lo = float(np.percentile(vals, 1.0))
        hi = float(np.percentile(vals, 99.0))
        if hi <= lo:
            lo = float(vals.min())
            hi = float(vals.max())
        if hi > lo:
            out = (ct_patch - lo) / (hi - lo)
            out = np.clip(out, 0.0, 1.0).astype(np.float32)
    elif norm_mode == "zscore_sigmoid":
        mu = float(np.mean(vals))
        sigma = max(float(np.std(vals)), 1e-6)
        z = (ct_patch - mu) / (sigma * max(float(zscore_scale), 1e-6))
        z = np.clip(z, -12.0, 12.0)
        out = (1.0 / (1.0 + np.exp(-z))).astype(np.float32)
    elif norm_mode == "ct_hu_window":
        out = ((ct_patch + 200.0) / 1200.0).astype(np.float32)
        out = np.clip(out, 0.0, 1.0)
    else:
        raise ValueError(f"Unknown norm_mode: {norm_mode}")

    out = foreground_floor + out * (1.0 - foreground_floor)
    out[~mask] = 0.0
    return out.astype(np.float32)


def extract_fixed_physical_crop(
    volume,
    center,
    spacing,
    physical_fov_mm,
    fill_value,
):
    source_size = np.maximum(
        1,
        np.rint(np.asarray(physical_fov_mm) / np.asarray(spacing)).astype(int),
    )
    start = np.floor(np.asarray(center) - source_size / 2.0).astype(int)
    end = start + source_size
    crop = np.full(tuple(source_size), fill_value, dtype=volume.dtype)

    source_start = np.maximum(start, 0)
    source_end = np.minimum(end, np.asarray(volume.shape))
    if np.any(source_end <= source_start):
        return crop
    destination_start = source_start - start
    destination_end = destination_start + (source_end - source_start)
    crop[
        destination_start[0]:destination_end[0],
        destination_start[1]:destination_end[1],
        destination_start[2]:destination_end[2],
    ] = volume[
        source_start[0]:source_end[0],
        source_start[1]:source_end[1],
        source_start[2]:source_end[2],
    ]
    return crop


def compute_patient_vertebral_reference(ct, seg):
    vertebral_mask = (seg >= min(VERTEBRA_LABELS.values())) & (
        seg <= max(VERTEBRA_LABELS.values())
    )
    values = ct[vertebral_mask & np.isfinite(ct)]
    if values.size == 0:
        return 0.0, 100.0
    lower, center, upper = np.percentile(values, [25.0, 50.0, 75.0])
    robust_scale = max(float((upper - lower) / 1.349), 50.0)
    return float(center), robust_scale


def resize_physical_context(
    ct,
    seg,
    vname,
    spacing,
    patch_size=(96, 96, 64),
    physical_fov_mm=(128.0, 128.0, 96.0),
):
    target_mask = seg == VERTEBRA_LABELS[vname]
    if not np.any(target_mask):
        return None, None

    coordinates = np.argwhere(target_mask)
    center = (coordinates.min(axis=0) + coordinates.max(axis=0)) / 2.0
    ct_crop = extract_fixed_physical_crop(
        ct,
        center,
        spacing,
        physical_fov_mm,
        fill_value=-1000.0,
    )
    mask_crop = extract_fixed_physical_crop(
        target_mask.astype(np.float32),
        center,
        spacing,
        physical_fov_mm,
        fill_value=0.0,
    )

    resize_image = Resize(
        spatial_size=patch_size,
        mode="trilinear",
        align_corners=False,
        anti_aliasing=False,
    )
    resize_mask = Resize(
        spatial_size=patch_size,
        mode="nearest",
        anti_aliasing=False,
    )
    ct_resized = resize_image(torch.from_numpy(ct_crop[None])).numpy()[0]
    mask_resized = resize_mask(torch.from_numpy(mask_crop[None])).numpy()[0]
    mask_resized = (mask_resized > 0.5).astype(np.float32)
    return ct_resized, mask_resized


def build_physical_context_patch(
    ct,
    seg,
    vname,
    spacing,
    patch_size=(96, 96, 64),
    physical_fov_mm=(128.0, 128.0, 96.0),
):
    ct_resized, mask_resized = resize_physical_context(
        ct,
        seg,
        vname,
        spacing,
        patch_size=patch_size,
        physical_fov_mm=physical_fov_mm,
    )
    if ct_resized is None:
        return np.zeros((3, *patch_size), dtype=np.float32)

    bone_window = np.clip((ct_resized + 200.0) / 1200.0, 0.0, 1.0)
    marrow_window = np.clip((ct_resized + 150.0) / 550.0, 0.0, 1.0)
    return np.stack(
        [bone_window, marrow_window, mask_resized],
        axis=0,
    ).astype(np.float32)


def build_patient_relative_context_patch(
    ct,
    seg,
    vname,
    spacing,
    patient_reference,
    patch_size=(96, 96, 64),
    physical_fov_mm=(128.0, 128.0, 96.0),
):
    ct_resized, mask_resized = resize_physical_context(
        ct,
        seg,
        vname,
        spacing,
        patch_size=patch_size,
        physical_fov_mm=physical_fov_mm,
    )
    if ct_resized is None:
        return np.zeros((3, *patch_size), dtype=np.float32)

    center, scale = patient_reference
    relative_context = 0.5 + (ct_resized - center) / (6.0 * scale)
    relative_context = np.clip(relative_context, 0.0, 1.0)
    relative_context[ct_resized <= -500.0] = 0.0
    target_relative = relative_context * mask_resized
    return np.stack(
        [relative_context, target_relative, mask_resized],
        axis=0,
    ).astype(np.float32)


def build_stage1_patch_from_arrays(
    ct,
    seg,
    vname,
    patch_size=(96, 96, 64),
    norm_mode="zscore_sigmoid",
    zscore_scale=1.5,
    foreground_floor=0.45,
    spacing=(1.0, 1.0, 1.0),
    patch_mode="isolated",
    physical_fov_mm=(128.0, 128.0, 96.0),
    patient_reference=None,
):
    if patch_mode == "physical_context":
        return build_physical_context_patch(
            ct,
            seg,
            vname,
            spacing=spacing,
            patch_size=patch_size,
            physical_fov_mm=physical_fov_mm,
        )
    if patch_mode == "patient_relative_context":
        if patient_reference is None:
            patient_reference = compute_patient_vertebral_reference(ct, seg)
        return build_patient_relative_context_patch(
            ct,
            seg,
            vname,
            spacing=spacing,
            patient_reference=patient_reference,
            patch_size=patch_size,
            physical_fov_mm=physical_fov_mm,
        )
    if patch_mode != "isolated":
        raise ValueError(f"Unknown patch_mode: {patch_mode}")

    vid = VERTEBRA_LABELS[vname]
    ct_patch = extract_centered_label_cube(
        ct,
        seg,
        vid,
        size=patch_size,
    ).astype(np.float32)
    seg_bin = np.where(seg == vid, 1.0, 0.0)
    seg_patch = extract_centered_label_cube(
        seg_bin,
        seg,
        vid,
        size=patch_size,
    ).astype(np.float32)
    mask = seg_patch > 0.5

    ct_patch = np.clip(ct_patch, -200.0, 1000.0)
    return normalize_stage1_patch(
        ct_patch,
        mask,
        norm_mode=norm_mode,
        zscore_scale=zscore_scale,
        foreground_floor=foreground_floor,
    )


class VertebraDataset(Dataset):

    def __init__(
        self,
        csv_path,
        root_dir,
        use_patch_cache=True,
        cache_dir=None,
        patient_cache_size=8,
        patch_size=(96, 96, 64),
        norm_mode="zscore_sigmoid",
        zscore_scale=1.5,
        foreground_floor=0.45,
        source="original",
        patch_mode="isolated",
        physical_fov_mm=(128.0, 128.0, 96.0),
    ):
        self.df = pd.read_csv(csv_path)
        if "source" not in self.df.columns:
            self.df["source"] = source
        self.root = root_dir
        self.use_patch_cache = use_patch_cache
        self.cache_dir = Path(cache_dir or (resolve_dataset_root() / "vertebra_patch_cache"))
        self.patient_cache_size = max(int(patient_cache_size), 0)
        self.norm_mode = str(norm_mode)
        self.zscore_scale = max(float(zscore_scale), 1e-6)
        self.foreground_floor = float(np.clip(foreground_floor, 0.0, 0.95))
        self.patch_size = tuple(int(x) for x in patch_size)
        self.patch_mode = str(patch_mode)
        self.physical_fov_mm = tuple(float(value) for value in physical_fov_mm)
        self._patient_cache = {}
        self._patient_cache_order = []

        if self.use_patch_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self):
        return len(self.df)

    def exclude_patients(self, patient_ids: set[str]) -> int:
        excluded = {str(patient_id) for patient_id in patient_ids}
        keep = ~self.df["patient_id"].astype(str).isin(excluded)
        removed = int((~keep).sum())
        self.df = self.df.loc[keep].reset_index(drop=True)
        return removed

    def _cache_path(self, pid, vname):
        return stage1_cache_path(
            self.cache_dir,
            pid,
            vname,
            self.patch_size,
            self.norm_mode,
            self.zscore_scale,
            self.foreground_floor,
            patch_mode=self.patch_mode,
            physical_fov_mm=self.physical_fov_mm,
        )

    def _get_patient_arrays(self, pid):
        if pid in self._patient_cache:
            if pid in self._patient_cache_order:
                self._patient_cache_order.remove(pid)
            self._patient_cache_order.append(pid)
            return self._patient_cache[pid]

        patient_dir = Path(self.root) / pid
        ct_path = patient_dir / f"{pid}_ct.nii.gz"
        seg_path = patient_dir / f"{pid}_seg-1.nii.gz"
        ct, seg, spacing = load_aligned_ct_seg_with_spacing(ct_path, seg_path)
        patient_reference = compute_patient_vertebral_reference(ct, seg)

        if self.patient_cache_size > 0:
            self._patient_cache[pid] = (
                ct,
                seg,
                spacing,
                patient_reference,
            )
            self._patient_cache_order.append(pid)
            while len(self._patient_cache_order) > self.patient_cache_size:
                evict_pid = self._patient_cache_order.pop(0)
                self._patient_cache.pop(evict_pid, None)

        return ct, seg, spacing, patient_reference

    def _build_patch(self, pid, vname):
        ct, seg, spacing, patient_reference = self._get_patient_arrays(pid)
        return self._build_patch_from_arrays(
            ct,
            seg,
            vname,
            spacing,
            patient_reference,
        )

    def _normalize_foreground(self, ct_patch, mask):
        return normalize_stage1_patch(
            ct_patch,
            mask,
            norm_mode=self.norm_mode,
            zscore_scale=self.zscore_scale,
            foreground_floor=self.foreground_floor,
        )

    def _build_patch_from_arrays(
        self,
        ct,
        seg,
        vname,
        spacing,
        patient_reference,
    ):
        return build_stage1_patch_from_arrays(
            ct,
            seg,
            vname,
            patch_size=self.patch_size,
            norm_mode=self.norm_mode,
            zscore_scale=self.zscore_scale,
            foreground_floor=self.foreground_floor,
            spacing=spacing,
            patch_mode=self.patch_mode,
            physical_fov_mm=self.physical_fov_mm,
            patient_reference=patient_reference,
        )

    def _cache_jobs(self, force=False):
        jobs = []
        for pid, group in self.df.groupby("patient_id", sort=False):
            jobs.append(
                {
                    "patient_id": str(pid),
                    "root": str(self.root),
                    "cache_dir": str(self.cache_dir),
                    "patch_size": self.patch_size,
                    "norm_mode": self.norm_mode,
                    "zscore_scale": self.zscore_scale,
                    "foreground_floor": self.foreground_floor,
                    "patch_mode": self.patch_mode,
                    "physical_fov_mm": self.physical_fov_mm,
                    "force": bool(force),
                    "vertebrae": [str(row.vertebra) for row in group.itertuples(index=False)],
                }
            )
        return jobs

    def precompute_cache(self, force=False, verbose=True, num_workers=1):
        if not self.use_patch_cache:
            if verbose:
                print("Patch cache disabled; skipping precompute.")
            return

        total = len(self.df)
        built = 0
        skipped = 0
        seen = 0
        num_workers = max(int(num_workers), 1)
        jobs = self._cache_jobs(force=force)

        if num_workers > 1 and len(jobs) > 1:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(build_patient_patch_cache, job) for job in jobs]
                for future in as_completed(futures):
                    result = future.result()
                    built += int(result["built"])
                    skipped += int(result["skipped"])
                    seen += int(result["seen"])
                    if verbose:
                        print(
                            f"[cache] {seen}/{total} | built={built} "
                            f"skipped={skipped} | patient={result['patient_id']}"
                        )
            return

        # Memory-safe precompute: keep only one patient's CT/seg in memory at a time.
        for pid, group in self.df.groupby("patient_id", sort=False):
            pid = str(pid)
            patient_dir = Path(self.root) / pid
            ct_path = patient_dir / f"{pid}_ct.nii.gz"
            seg_path = patient_dir / f"{pid}_seg-1.nii.gz"
            ct, seg, spacing = load_aligned_ct_seg_with_spacing(ct_path, seg_path)
            patient_reference = compute_patient_vertebral_reference(ct, seg)

            for row in group.itertuples(index=False):
                vname = row.vertebra
                out_path = self._cache_path(pid, vname)
                if out_path.exists() and not force:
                    skipped += 1
                else:
                    patch_np = self._build_patch_from_arrays(
                        ct,
                        seg,
                        vname,
                        spacing,
                        patient_reference,
                    )
                    np.save(out_path, patch_np, allow_pickle=False)
                    built += 1

                seen += 1
                if verbose and (seen % 100 == 0 or seen == total):
                    print(f"[cache] {seen}/{total} | built={built} skipped={skipped}")

            del ct, seg

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        pid = str(row["patient_id"])
        vname = row["vertebra"]
        label = int(row["label"])

        patch_np = None
        if self.use_patch_cache:
            cache_path = self._cache_path(pid, vname)
            if cache_path.exists():
                patch_np = np.load(cache_path, allow_pickle=False).astype(np.float32)
            else:
                patch_np = self._build_patch(pid, vname)
                np.save(cache_path, patch_np, allow_pickle=False)
        else:
            patch_np = self._build_patch(pid, vname)

        patch = torch.from_numpy(patch_np)
        return patch, torch.tensor(label, dtype=torch.long)
    

if __name__ == "__main__":
    data_root = resolve_dataset_root()
    build_dataset_csv(
        metadata_path=data_root / "patient_metadata.csv",
        output_path=data_root / "vertebra_dataset.csv",
    )
    dataset = VertebraDataset(
        csv_path=data_root / "vertebra_dataset.csv",
        root_dir=data_root / "Spine-Mets-CT-SEG-Nifti",
    )
    print(f"Dataset size: {len(dataset)}")
    patch, label = dataset[0]
    print(f"Patch shape: {patch.shape}, Label: {label}")
    
    
    
    
    
    
