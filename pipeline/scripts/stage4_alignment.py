import os
import argparse
import numpy as np
import pandas as pd
from utils import INV_LABELS, load_seg, vox2world


def centroid_world_from_ijk(ijk, aff):
    if ijk.size == 0:
        return None
    xyz = vox2world(aff, ijk)
    c = np.median(xyz, axis=0)
    return c


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


def fit_line(z, v):
    z = np.asarray(z, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    A = np.vstack([z, np.ones_like(z)]).T
    coef, _, _, _ = np.linalg.lstsq(A, v, rcond=None)
    a, b = float(coef[0]), float(coef[1])
    pred = a * z + b
    resid = v - pred
    return a, b, pred, resid


def fit_quad(z, v):
    z = np.asarray(z, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    A = np.vstack([z ** 2, z, np.ones_like(z)]).T
    coef, _, _, _ = np.linalg.lstsq(A, v, rcond=None)
    a2, a1, a0 = float(coef[0]), float(coef[1]), float(coef[2])
    pred = a2 * z ** 2 + a1 * z + a0
    resid = v - pred
    return (a2, a1, a0), pred, resid


def fit_cubic(z, v):
    z = np.asarray(z, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    degree = min(3, max(len(z) - 1, 1))
    coef = np.polyfit(z, v, degree)
    pred = np.polyval(coef, z)
    resid = v - pred
    return tuple(float(c) for c in coef), pred, resid


def local_subluxation_flags(levels, x, y, z, thresh_mm):
    """
    levels is list of integer labels 1..17, sorted by z descending or ascending
    We compute expected (x,y) at each internal point by linear interpolation between neighbors in z.
    Then we flag if displacement is large and locally outlying.
    """
    levels = np.asarray(levels, dtype=np.int32)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)

    n = len(levels)
    flags = []

    if n < 3:
        return flags

    for i in range(1, n - 1):
        z0, z1, z2 = z[i - 1], z[i], z[i + 1]
        if z2 == z0:
            continue

        t = (z1 - z0) / (z2 - z0)

        x_exp = x[i - 1] + t * (x[i + 1] - x[i - 1])
        y_exp = y[i - 1] + t * (y[i + 1] - y[i - 1])

        dx = x[i] - x_exp
        dy = y[i] - y_exp
        disp = float(np.sqrt(dx * dx + dy * dy))

        if disp >= thresh_mm:
            flags.append({
                "label_id": int(levels[i]),
                "vertebra": INV_LABELS[int(levels[i])],
                "disp_mm": disp,
                "dx_mm": float(dx),
                "dy_mm": float(dy),
            })

    return flags


def alignment_from_points(pts, translation_thresh_mm, scoliosis_max_dev_mm, kyphosis_max_dev_mm):
    """Score alignment from vertebral centroids [(label, x, y, z), ...] in world mm."""
    if len(pts) < 5:
        return None

    pts = sorted(pts, key=lambda t: t[3])
    labs = [p[0] for p in pts]
    x = np.array([p[1] for p in pts], dtype=np.float64)
    y = np.array([p[2] for p in pts], dtype=np.float64)
    z = np.array([p[3] for p in pts], dtype=np.float64)

    # coronal analysis: x versus z
    _, _, _, resid_x_line = fit_line(z, x)
    coronal_max_dev = float(np.max(np.abs(resid_x_line)))

    # sagittal analysis: y versus z with cubic smooth curve when enough vertebrae exist.
    _, _, resid_y_cubic = fit_cubic(z, y)
    sagittal_max_dev = float(np.max(np.abs(resid_y_cubic)))

    # local subluxation or translation: large local displacement compared to neighbors
    flags = local_subluxation_flags(labs, x, y, z, translation_thresh_mm)
    has_sublux = len(flags) > 0

    # global curve abnormality
    has_curve = (coronal_max_dev >= scoliosis_max_dev_mm) or (sagittal_max_dev >= kyphosis_max_dev_mm)

    # score rule
    if has_sublux:
        score = 4
    elif has_curve:
        score = 2
    else:
        score = 0

    summary = {
        "coronal_max_dev_mm": coronal_max_dev,
        "sagittal_max_dev_mm": sagittal_max_dev,
        "has_subluxation": int(has_sublux),
        "has_curve_deviation": int(has_curve),
        "alignment_score": int(score),
        "n_vertebrae_found": int(len(pts)),
    }

    return summary, flags


def centroid_points_from_seg(seg, aff):
    label_ijk = build_label_ijk_map(seg)
    pts = []
    for lab in range(1, 18):
        ijk = label_ijk.get(lab, np.empty((0, 3), dtype=np.int64))
        c = centroid_world_from_ijk(ijk, aff)
        if c is None:
            continue
        # canonical axis assumption after as_closest_canonical:
        # x left right, y posterior anterior, z inferior superior or superior inferior depending on affine sign
        pts.append((lab, c[0], c[1], c[2]))
    return pts


def compute_patient_alignment(seg_path, translation_thresh_mm, scoliosis_max_dev_mm, kyphosis_max_dev_mm):
    seg, aff = load_seg(seg_path)
    pts = centroid_points_from_seg(seg, aff)
    return alignment_from_points(pts, translation_thresh_mm, scoliosis_max_dev_mm, kyphosis_max_dev_mm)


def analyze_patient_alignment_geometry(seg_path, translation_thresh_mm=6.0, scoliosis_max_dev_mm=10.0, kyphosis_max_dev_mm=10.0):
    seg, aff = load_seg(seg_path)
    label_ijk = build_label_ijk_map(seg)

    pts = []
    for lab in range(1, 18):
        ijk = label_ijk.get(lab, np.empty((0, 3), dtype=np.int64))
        c = centroid_world_from_ijk(ijk, aff)
        if c is not None:
            pts.append((lab, c[0], c[1], c[2]))

    if len(pts) < 5:
        return None

    pts = sorted(pts, key=lambda t: t[3])
    labs = np.asarray([p[0] for p in pts], dtype=np.int32)
    x = np.asarray([p[1] for p in pts], dtype=np.float64)
    y = np.asarray([p[2] for p in pts], dtype=np.float64)
    z = np.asarray([p[3] for p in pts], dtype=np.float64)

    line_a, line_b, x_fit, resid_x = fit_line(z, x)
    cubic_coef, y_fit, resid_y = fit_cubic(z, y)
    fitted_xyz = np.column_stack((x_fit, y_fit, z))
    center_xyz = np.column_stack((x, y, z))
    deviation = np.sqrt((x - x_fit) ** 2 + (y - y_fit) ** 2)

    z_dense = np.linspace(float(z.min()), float(z.max()), 160)
    x_dense = line_a * z_dense + line_b
    y_dense = np.polyval(np.asarray(cubic_coef, dtype=np.float64), z_dense)
    fitted_dense_xyz = np.column_stack((x_dense, y_dense, z_dense))

    flags = local_subluxation_flags(labs, x, y, z, translation_thresh_mm)
    coronal_max_dev = float(np.max(np.abs(resid_x)))
    sagittal_max_dev = float(np.max(np.abs(resid_y)))
    has_sublux = len(flags) > 0
    has_curve = (coronal_max_dev >= scoliosis_max_dev_mm) or (sagittal_max_dev >= kyphosis_max_dev_mm)
    if has_sublux:
        score = 4
    elif has_curve:
        score = 2
    else:
        score = 0

    summary = {
        "coronal_max_dev_mm": coronal_max_dev,
        "sagittal_max_dev_mm": sagittal_max_dev,
        "has_subluxation": int(has_sublux),
        "has_curve_deviation": int(has_curve),
        "alignment_score": int(score),
        "n_vertebrae_found": int(len(pts)),
        "line_x_coef": (line_a, line_b),
        "cubic_y_coef": cubic_coef,
    }

    return {
        "summary": summary,
        "flags": flags,
        "labels": labs,
        "vertebrae": [INV_LABELS[int(lab)] for lab in labs],
        "centers": center_xyz,
        "fitted": fitted_xyz,
        "fitted_dense": fitted_dense_xyz,
        "residual_x": resid_x,
        "residual_y": resid_y,
        "deviation_mm": deviation,
        "seg": seg,
        "affine": aff,
    }


def run(root, translation_thresh_mm, scoliosis_max_dev_mm, kyphosis_max_dev_mm):
    summaries = []
    all_flags = []

    for pid in os.listdir(root):
        pdir = os.path.join(root, pid)
        if not os.path.isdir(pdir):
            continue

        seg_path = os.path.join(pdir, f"{pid}_seg-1.nii.gz")
        if not os.path.exists(seg_path):
            continue

        res = compute_patient_alignment(
            seg_path=seg_path,
            translation_thresh_mm=translation_thresh_mm,
            scoliosis_max_dev_mm=scoliosis_max_dev_mm,
            kyphosis_max_dev_mm=kyphosis_max_dev_mm
        )

        if res is None:
            continue

        summary, flags = res
        summary["patient_id"] = str(pid)
        summaries.append(summary)

        for f in flags:
            f["patient_id"] = str(pid)
            all_flags.append(f)

    df_sum = pd.DataFrame(summaries)
    df_sum.to_csv("output/stage4_alignment.csv", index=False)

    df_flags = pd.DataFrame(all_flags)
    df_flags.to_csv("output/stage4_subluxation_flags.csv", index=False)

    print("Saved stage4_alignment.csv")
    print("Saved stage4_subluxation_flags.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--root", required=True, type=str)

    parser.add_argument("--translation_thresh_mm", type=float, default=6.0)
    parser.add_argument("--scoliosis_max_dev_mm", type=float, default=10.0)
    parser.add_argument("--kyphosis_max_dev_mm", type=float, default=10.0)

    args = parser.parse_args()

    run(
        root=args.root,
        translation_thresh_mm=args.translation_thresh_mm,
        scoliosis_max_dev_mm=args.scoliosis_max_dev_mm,
        kyphosis_max_dev_mm=args.kyphosis_max_dev_mm
    )
