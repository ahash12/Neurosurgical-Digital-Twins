import os
import argparse
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from scipy import ndimage  # type: ignore[import-untyped]

from pipeline.scripts.stage2_collapse import analyze_height_geometry
from utils import INV_LABELS, load_img, vox2world


DEFAULT_LESION_LOW_HU = -200.0
DEFAULT_LESION_HIGH_HU = 450.0
SIDE_COUNT_THRESHOLD = 50
DEFAULT_THRESHOLD_MODE = "auto_patient"
AUTO_EROSION_ITERS = 2
AUTO_LOW_PERCENTILE = 1.0
AUTO_HIGH_PERCENTILE = 99.0
AUTO_CLIP_MIN_HU = -500.0
AUTO_CLIP_MAX_HU = 1500.0
AUTO_LOW_CLAMP = (-300.0, -100.0)
AUTO_HIGH_CLAMP = (150.0, 650.0)
AUTO_MIN_SPREAD_HU = 250.0
AUTO_BODY_EROSION_ITERS = 4
AUTO_BODY_MAX_PER_LEVEL = 4000
BONE_FLOOR_HU = -300.0
MRI_REFERENCE_EROSION_ITERS = 2
MRI_MIN_REFERENCE_VOXELS = 100
MRI_LOW_SIGNAL_PERCENTILE = 30.0
MRI_POST_CONTRAST_HIGH_PERCENTILE = 80.0
MRI_VIBE_HIGH_PERCENTILE = 75.0


def prepare_ct_seg_arrays(ct, seg):
    common_shape = tuple(min(c, s) for c, s in zip(ct.shape, seg.shape))
    ct = ct[: common_shape[0], : common_shape[1], : common_shape[2]]
    seg = seg[: common_shape[0], : common_shape[1], : common_shape[2]]
    return ct, seg


def build_label_ijk_map(seg):
    fg = np.argwhere(seg > 0)
    if fg.size == 0:
        return {}
    labels = seg[fg[:, 0], fg[:, 1], fg[:, 2]].astype(np.int16)
    out = {}
    for lab in range(1, 18):
        m = labels == lab
        if np.any(m):
            out[lab] = fg[m]
    return out


def bone_like_mask(ct_vals, bone_floor_hu=BONE_FLOOR_HU):
    return ct_vals > bone_floor_hu


def compute_patient_hu_thresholds(
    ct,
    seg,
    erosion_iters=AUTO_EROSION_ITERS,
    low_percentile=AUTO_LOW_PERCENTILE,
    high_percentile=AUTO_HIGH_PERCENTILE,
    clip_min_hu=AUTO_CLIP_MIN_HU,
    clip_max_hu=AUTO_CLIP_MAX_HU,
    low_clamp=AUTO_LOW_CLAMP,
    high_clamp=AUTO_HIGH_CLAMP,
    min_spread_hu=AUTO_MIN_SPREAD_HU,
    body_erosion_iters=AUTO_BODY_EROSION_ITERS,
    body_max_per_level=AUTO_BODY_MAX_PER_LEVEL,
):
    level_vals = []
    for lab in range(1, 18):
        mask = seg == lab
        if not np.any(mask):
            continue

        body = (
            ndimage.binary_erosion(mask, iterations=int(body_erosion_iters))
            if body_erosion_iters > 0
            else mask
        )
        if body.sum() < 500:
            body = (
                ndimage.binary_erosion(
                    mask, iterations=max(int(body_erosion_iters) // 2, 1)
                )
                if body_erosion_iters > 1
                else mask
            )
        if body.sum() < 200:
            body = (
                ndimage.binary_erosion(mask, iterations=int(erosion_iters))
                if erosion_iters > 0
                else mask
            )
        if body.sum() < 100:
            body = mask

        vals = ct[body]
        vals = vals[(vals > clip_min_hu) & (vals < clip_max_hu)]
        if vals.size == 0:
            continue

        vals = vals[bone_like_mask(vals)]
        if vals.size == 0:
            continue

        if vals.size > body_max_per_level:
            idx = np.linspace(0, vals.size - 1, body_max_per_level, dtype=int)
            vals = np.sort(vals)[idx]
        level_vals.append(vals)

    if level_vals:
        vals = np.concatenate(level_vals, axis=0)
    else:
        mask = seg > 0
        if erosion_iters > 0:
            mask = ndimage.binary_erosion(mask, iterations=int(erosion_iters))
        vals = ct[mask]
        vals = vals[(vals > clip_min_hu) & (vals < clip_max_hu)]
        vals = vals[bone_like_mask(vals)]

    if vals.size < 1000:
        mask = seg > 0
        vals = ct[mask]
        vals = vals[(vals > clip_min_hu) & (vals < clip_max_hu)]
        vals = vals[bone_like_mask(vals)]

    if vals.size == 0:
        return (
            DEFAULT_LESION_LOW_HU,
            DEFAULT_LESION_HIGH_HU,
            {
                "mode": "fixed_fallback",
                "n": 0,
                "q01": np.nan,
                "q50": np.nan,
                "q95": np.nan,
                "raw_low": np.nan,
                "raw_high": np.nan,
                "fallback_used": True,
            },
        )
    raw_low = float(np.percentile(vals, low_percentile))
    raw_high = float(np.percentile(vals, high_percentile))
    low = float(np.clip(raw_low, low_clamp[0], low_clamp[1]))
    high = float(np.clip(raw_high, high_clamp[0], high_clamp[1]))
    fallback_used = False
    if high - low < min_spread_hu:
        low = DEFAULT_LESION_LOW_HU
        high = DEFAULT_LESION_HIGH_HU
        fallback_used = True
    meta = {
        "mode": "auto_patient",
        "n": int(vals.size),
        "q01": float(np.percentile(vals, 1)),
        "q50": float(np.percentile(vals, 50)),
        "q95": float(np.percentile(vals, 95)),
        "raw_low": raw_low,
        "raw_high": raw_high,
        "fallback_used": fallback_used,
    }
    return low, high, meta


def resolve_thresholds(
    ct,
    seg,
    threshold_mode=DEFAULT_THRESHOLD_MODE,
    lesion_low_hu=None,
    lesion_high_hu=None,
):
    if threshold_mode == "fixed":
        low = DEFAULT_LESION_LOW_HU if lesion_low_hu is None else float(lesion_low_hu)
        high = (
            DEFAULT_LESION_HIGH_HU if lesion_high_hu is None else float(lesion_high_hu)
        )
        return (
            low,
            high,
            {
                "mode": "fixed",
                "n": 0,
                "q01": np.nan,
                "q50": np.nan,
                "q95": np.nan,
                "raw_low": np.nan,
                "raw_high": np.nan,
                "fallback_used": False,
            },
        )
    low, high, meta = compute_patient_hu_thresholds(ct, seg)
    if lesion_low_hu is not None:
        low = float(lesion_low_hu)
    if lesion_high_hu is not None:
        high = float(lesion_high_hu)
    return low, high, meta


def mri_sequence_family(sequence):
    text = str(sequence).lower()
    if not text.strip() or text.strip() in {"unknown", "unspecified"}:
        return "unknown"
    if any(token in text for token in ("adc", "diffusion", "dwi", "localizer", "scout", "mip")):
        return "unsupported"
    if "vibe" in text:
        if "post" in text or "gad" in text:
            return "vibe_post_t1"
        return "t1" if "t1" in text else "unknown"
    if ("fs" in text or "fat" in text) and ("post" in text or "gad" in text):
        return "fs_post_t1"
    if "post" in text or "gad" in text:
        return "post_t1"
    if "stir" in text:
        return "stir"
    if "t2" in text:
        return "t2"
    if "t1" in text:
        return "t1"
    return "unknown"


def mri_percentile_thresholds(sequence_family):
    if sequence_family == "t1":
        return MRI_LOW_SIGNAL_PERCENTILE, 100.0
    if sequence_family == "vibe_post_t1":
        return -1.0, MRI_VIBE_HIGH_PERCENTILE
    if sequence_family in {"post_t1", "fs_post_t1", "stir", "t2"}:
        return -1.0, MRI_POST_CONTRAST_HIGH_PERCENTILE
    return MRI_LOW_SIGNAL_PERCENTILE, 100.0


def mri_reference_mask(seg):
    reference = np.zeros_like(seg, dtype=bool)
    for label in range(1, 18):
        mask = seg == label
        if not np.any(mask):
            continue
        eroded = ndimage.binary_erosion(mask, iterations=MRI_REFERENCE_EROSION_ITERS)
        reference |= (
            eroded if np.count_nonzero(eroded) >= MRI_MIN_REFERENCE_VOXELS else mask
        )
    return reference


def relative_percentile_image(image, seg):
    reference_mask = mri_reference_mask(seg)
    reference_values = np.asarray(image[reference_mask], dtype=np.float32)
    reference_values = reference_values[np.isfinite(reference_values)]
    reference_values = reference_values[reference_values > 0]
    if reference_values.size < MRI_MIN_REFERENCE_VOXELS:
        reference_values = np.asarray(image[seg > 0], dtype=np.float32)
        reference_values = reference_values[np.isfinite(reference_values)]
        reference_values = reference_values[reference_values > 0]
    if reference_values.size == 0:
        return np.zeros_like(image, dtype=np.float32)

    sorted_values = np.sort(reference_values)
    ranks = np.asarray(
        np.searchsorted(sorted_values, image.astype(np.float32), side="right")
    )
    percentiles = (ranks.astype(np.float32) / float(sorted_values.size)) * 100.0
    percentiles[seg <= 0] = 50.0
    return np.clip(percentiles, 0.0, 100.0).astype(np.float32)


def resolve_mri_thresholds(image, seg, sequence):
    sequence_family = mri_sequence_family(sequence)
    if sequence_family in {"unknown", "unsupported"}:
        raise ValueError(
            "MRI Stage 3 requires a reviewed T1, T2, STIR, or post-contrast "
            "sequence; unknown or unsupported MRI cannot be scored."
        )
    percentiles = relative_percentile_image(image, seg)
    low, high = mri_percentile_thresholds(sequence_family)
    meta = {
        "mode": "mri_patient_relative_percentile",
        "sequence_family": sequence_family,
        "feature_image": percentiles,
        "use_bone_mask": False,
        "n": int(np.count_nonzero(seg > 0)),
        "q01": 1.0,
        "q50": 50.0,
        "q95": 95.0,
        "raw_low": low,
        "raw_high": high,
        "fallback_used": False,
    }
    return low, high, meta


# Future paired-sequence hook:
# If pre/post T1 scans can be registered for one patient, compute
# enhancement_ratio = post_contrast_signal / max(pre_contrast_signal, eps)
# inside the vertebral-body marrow reference and use that feature image for
# post-contrast MRI Stage 3 instead of single-scan percentile rank.


def analyze_posterolateral_from_ijk(
    ct,
    aff,
    ijk,
    lesion_low_hu,
    lesion_high_hu,
    threshold_mode=DEFAULT_THRESHOLD_MODE,
    threshold_meta=None,
):
    if ijk.size == 0:
        return None

    in_bounds = (
        (ijk[:, 0] >= 0)
        & (ijk[:, 0] < ct.shape[0])
        & (ijk[:, 1] >= 0)
        & (ijk[:, 1] < ct.shape[1])
        & (ijk[:, 2] >= 0)
        & (ijk[:, 2] < ct.shape[2])
    )
    ijk = ijk[in_bounds]
    if ijk.size == 0:
        return None

    xyz = vox2world(aff, ijk)
    x = xyz[:, 0]
    y = xyz[:, 1]
    centroid_x = float(np.median(x))
    centroid_y = float(np.median(y))
    centroid = np.array(
        [centroid_x, centroid_y, float(np.median(xyz[:, 2]))], dtype=np.float64
    )

    stage2_geom = analyze_height_geometry(ijk, aff)
    if stage2_geom is not None:
        center_x = float(stage2_geom["canal"]["center_world"][0])
        center_y = float(stage2_geom["canal"]["center_world"][1])
        reference_center = np.array(
            stage2_geom["canal"]["center_world"], dtype=np.float64
        )
    else:
        center_x = centroid_x
        center_y = centroid_y
        reference_center = centroid.copy()

    threshold_meta = threshold_meta or {}
    ct_vals = ct[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
    use_bone_mask = bool(threshold_meta.get("use_bone_mask", True))
    bone_mask = (
        bone_like_mask(ct_vals) if use_bone_mask else np.ones(ct_vals.shape, dtype=bool)
    )
    scored_ct_vals = ct_vals[bone_mask]
    scored_xyz = xyz[bone_mask]
    abnormal_mask = np.logical_or(
        scored_ct_vals < lesion_low_hu, scored_ct_vals > lesion_high_hu
    )
    abnormal_xyz = scored_xyz[abnormal_mask]

    posterior_xyz = (
        abnormal_xyz[abnormal_xyz[:, 1] < center_y]
        if len(abnormal_xyz)
        else np.empty((0, 3), dtype=np.float64)
    )
    left_xyz = (
        posterior_xyz[posterior_xyz[:, 0] < center_x]
        if len(posterior_xyz)
        else np.empty((0, 3), dtype=np.float64)
    )
    right_xyz = (
        posterior_xyz[posterior_xyz[:, 0] > center_x]
        if len(posterior_xyz)
        else np.empty((0, 3), dtype=np.float64)
    )

    left_count = int(len(left_xyz))
    right_count = int(len(right_xyz))
    total = left_count + right_count

    if total == 0:
        score = 0
    else:
        left_flag = left_count > SIDE_COUNT_THRESHOLD
        right_flag = right_count > SIDE_COUNT_THRESHOLD
        if left_flag and right_flag:
            left_frac = left_count / total
            right_frac = right_count / total
            if left_frac >= 0.70 or right_frac >= 0.70:
                score = 1
            else:
                score = 3
        elif left_flag or right_flag:
            score = 1
        else:
            score = 0

    return {
        "score": score,
        "centroid": centroid,
        "reference_center": reference_center,
        "all_xyz": xyz,
        "scored_xyz": scored_xyz,
        "abnormal_xyz": abnormal_xyz,
        "posterior_xyz": posterior_xyz,
        "left_xyz": left_xyz,
        "right_xyz": right_xyz,
        "center_x": center_x,
        "center_y": center_y,
        "centroid_x": centroid_x,
        "centroid_y": centroid_y,
        "used_canal_center": bool(stage2_geom is not None),
        "left_count": left_count,
        "right_count": right_count,
        "total_posterior_abnormal": total,
        "left_flag": left_count > SIDE_COUNT_THRESHOLD,
        "right_flag": right_count > SIDE_COUNT_THRESHOLD,
        "ct_min": float(np.min(ct_vals)),
        "ct_max": float(np.max(ct_vals)),
        "ct_q05": float(np.percentile(ct_vals, 5)),
        "ct_q50": float(np.percentile(ct_vals, 50)),
        "ct_q95": float(np.percentile(ct_vals, 95)),
        "bone_voxel_count": int(np.sum(bone_mask)),
        "bone_voxel_fraction": float(np.mean(bone_mask)),
        "lesion_low_hu": float(lesion_low_hu),
        "lesion_high_hu": float(lesion_high_hu),
        "threshold_mode": threshold_mode,
        "threshold_patient_n": int(threshold_meta.get("n", 0)),
        "threshold_patient_q01": float(threshold_meta.get("q01", np.nan)),
        "threshold_patient_q50": float(threshold_meta.get("q50", np.nan)),
        "threshold_patient_q95": float(threshold_meta.get("q95", np.nan)),
        "threshold_raw_low": float(threshold_meta.get("raw_low", np.nan)),
        "threshold_raw_high": float(threshold_meta.get("raw_high", np.nan)),
        "threshold_fallback_used": bool(threshold_meta.get("fallback_used", False)),
    }


def posterolateral_score_from_ijk(
    ct,
    aff,
    ijk,
    lesion_low_hu,
    lesion_high_hu,
    threshold_mode=DEFAULT_THRESHOLD_MODE,
    threshold_meta=None,
):
    result = analyze_posterolateral_from_ijk(
        ct,
        aff,
        ijk,
        lesion_low_hu=lesion_low_hu,
        lesion_high_hu=lesion_high_hu,
        threshold_mode=threshold_mode,
        threshold_meta=threshold_meta,
    )
    if result is None:
        return 0
    return int(result["score"])


def run(
    root,
    threshold_mode=DEFAULT_THRESHOLD_MODE,
    lesion_low_hu=None,
    lesion_high_hu=None,
    output_csv="output/stage3_posterolateral.csv",
):
    rows = []
    for pid in os.listdir(root):
        pdir = os.path.join(root, pid)
        if not os.path.isdir(pdir):
            continue
        seg_path = os.path.join(pdir, f"{pid}_seg-1.nii.gz")
        ct_path = os.path.join(pdir, f"{pid}_ct.nii.gz")

        if not os.path.exists(seg_path) or not os.path.exists(ct_path):
            continue

        seg, aff = load_img(seg_path, canonical=False)
        ct, _ = load_img(ct_path, canonical=False)
        ct, seg = prepare_ct_seg_arrays(ct, seg)
        low_hu, high_hu, threshold_meta = resolve_thresholds(
            ct,
            seg,
            threshold_mode=threshold_mode,
            lesion_low_hu=lesion_low_hu,
            lesion_high_hu=lesion_high_hu,
        )
        label_ijk = build_label_ijk_map(seg)

        for lab in range(1, 18):
            ijk = label_ijk.get(lab, np.empty((0, 3), dtype=np.int64))
            score = posterolateral_score_from_ijk(
                ct,
                aff,
                ijk,
                lesion_low_hu=low_hu,
                lesion_high_hu=high_hu,
                threshold_mode=threshold_mode,
                threshold_meta=threshold_meta,
            )
            rows.append(
                {
                    "patient_id": pid,
                    "vertebra": INV_LABELS[lab],
                    "posterolateral_score": score,
                }
            )

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"saved {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument(
        "--threshold-mode",
        choices=["auto_patient", "fixed"],
        default=DEFAULT_THRESHOLD_MODE,
    )
    parser.add_argument("--lesion-low-hu", type=float, default=None)
    parser.add_argument("--lesion-high-hu", type=float, default=None)
    parser.add_argument(
        "--output-csv", type=str, default="output/stage3_posterolateral.csv"
    )
    args = parser.parse_args()
    run(
        args.root,
        threshold_mode=args.threshold_mode,
        lesion_low_hu=args.lesion_low_hu,
        lesion_high_hu=args.lesion_high_hu,
        output_csv=args.output_csv,
    )
