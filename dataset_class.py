import os
import numpy as np
import pandas as pd
import torch
import re
from collections import Counter
from pathlib import Path

from torch.utils.data import Dataset
from utils import (
    VERTEBRA_LABELS,
    VERTEBRAE,
    load_img,
    extract_centered_label_cube,
)


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


def build_dataset_csv(metadata_path="data/patient_metadata.csv", output_path="data/vertebra_dataset.csv"):
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

    out = pd.DataFrame(rows)
    out.to_csv(output_path, index=False)
    counts = dict(Counter(out["label"]))
    print(f"Saved: {output_path}")
    print(f"Label counts: {counts}")
    return out

class VertebraDataset(Dataset):

    def __init__(
        self,
        csv_path,
        root_dir,
        use_patch_cache=True,
        cache_dir="data/vertebra_patch_cache",
        patient_cache_size=8,
        patch_size=(96, 96, 96),
        norm_mode="zscore_sigmoid",
        zscore_scale=1.5,
        foreground_floor=0.45,
    ):
        self.df = pd.read_csv(csv_path)
        self.root = root_dir
        self.use_patch_cache = use_patch_cache
        self.cache_dir = Path(cache_dir)
        self.patient_cache_size = max(int(patient_cache_size), 0)
        self.norm_mode = str(norm_mode)
        self.zscore_scale = max(float(zscore_scale), 1e-6)
        self.foreground_floor = float(np.clip(foreground_floor, 0.0, 0.95))
        self.patch_size = tuple(int(x) for x in patch_size)
        self._patient_cache = {}
        self._patient_cache_order = []

        if self.use_patch_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self):
        return len(self.df)

    def _cache_path(self, pid, vname):
        return self.cache_dir / f"{pid}_{vname}.npy"

    def _get_patient_arrays(self, pid):
        if pid in self._patient_cache:
            if pid in self._patient_cache_order:
                self._patient_cache_order.remove(pid)
            self._patient_cache_order.append(pid)
            return self._patient_cache[pid]

        patient_dir = os.path.join(self.root, pid)
        ct_path = os.path.join(patient_dir, f"{pid}_ct.nii.gz")
        seg_path = os.path.join(patient_dir, f"{pid}_seg-1.nii.gz")
        ct, _ = load_img(ct_path, canonical=False)
        seg, _ = load_img(seg_path, canonical=False)

        if self.patient_cache_size > 0:
            self._patient_cache[pid] = (ct, seg)
            self._patient_cache_order.append(pid)
            while len(self._patient_cache_order) > self.patient_cache_size:
                evict_pid = self._patient_cache_order.pop(0)
                self._patient_cache.pop(evict_pid, None)

        return ct, seg

    def _build_patch(self, pid, vname):
        ct, seg = self._get_patient_arrays(pid)
        return self._build_patch_from_arrays(ct, seg, vname)

    def _normalize_foreground(self, ct_patch, mask):
        out = np.zeros_like(ct_patch, dtype=np.float32)
        if not np.any(mask):
            return out

        vals = ct_patch[mask]
        if self.norm_mode == "robust_minmax":
            lo = float(np.percentile(vals, 1.0))
            hi = float(np.percentile(vals, 99.0))
            if hi <= lo:
                lo = float(vals.min())
                hi = float(vals.max())
            if hi > lo:
                out = (ct_patch - lo) / (hi - lo)
                out = np.clip(out, 0.0, 1.0).astype(np.float32)
        elif self.norm_mode == "zscore_sigmoid":
            mu = float(np.mean(vals))
            sigma = float(np.std(vals))
            sigma = max(sigma, 1e-6)
            z = (ct_patch - mu) / (sigma * self.zscore_scale)
            z = np.clip(z, -12.0, 12.0)
            out = (1.0 / (1.0 + np.exp(-z))).astype(np.float32)
        else:
            raise ValueError(f"Unknown norm_mode: {self.norm_mode}")

        # Lift vertebra foreground away from 0 while keeping background at 0.
        out = self.foreground_floor + out * (1.0 - self.foreground_floor)
        out[~mask] = 0.0
        return out.astype(np.float32)

    def _build_patch_from_arrays(self, ct, seg, vname):
        vid = VERTEBRA_LABELS[vname]
        ct_patch = extract_centered_label_cube(ct, seg, vid, size=self.patch_size).astype(np.float32)
        seg_bin = np.where(seg == vid, 1.0, 0.0)
        seg_patch = extract_centered_label_cube(seg_bin, seg, vid, size=self.patch_size).astype(np.float32)
        mask = seg_patch > 0.5

        # Clip to CT window then normalize using vertebra-only statistics.
        ct_patch = np.clip(ct_patch, -200.0, 1000.0)
        out = self._normalize_foreground(ct_patch, mask)
        return out

    def precompute_cache(self, force=False, verbose=True):
        if not self.use_patch_cache:
            if verbose:
                print("Patch cache disabled; skipping precompute.")
            return

        total = len(self.df)
        built = 0
        skipped = 0
        seen = 0

        # Memory-safe precompute: keep only one patient's CT/seg in memory at a time.
        for pid, group in self.df.groupby("patient_id", sort=False):
            pid = str(pid)
            patient_dir = os.path.join(self.root, pid)
            ct_path = os.path.join(patient_dir, f"{pid}_ct.nii.gz")
            seg_path = os.path.join(patient_dir, f"{pid}_seg-1.nii.gz")
            ct, _ = load_img(ct_path, canonical=False)
            seg, _ = load_img(seg_path, canonical=False)

            for row in group.itertuples(index=False):
                vname = row.vertebra
                out_path = self._cache_path(pid, vname)
                if out_path.exists() and not force:
                    skipped += 1
                else:
                    patch_np = self._build_patch_from_arrays(ct, seg, vname)
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
    build_dataset_csv(
        metadata_path="data/patient_metadata.csv",
        output_path="data/vertebra_dataset.csv",
    )
    dataset = VertebraDataset(
        csv_path="data/vertebra_dataset.csv",
        root_dir="data/Spine-Mets-CT-SEG-Nifti",
    )
    print(f"Dataset size: {len(dataset)}")
    patch, label = dataset[0]
    print(f"Patch shape: {patch.shape}, Label: {label}")
    
    
    
    
    
