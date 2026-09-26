import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nibabel.processing import resample_from_to
from vedo import Plotter, Sphere, Text2D, Volume, show

from read_data_nifti import _load_ct_and_seg_aligned
from utils import (
    DEFAULT_DATA_ROOT,
    STAGE0_VERTEBRA_LABELS,
    VERTEBRA_LABELS,
    load_seg,
    vox2world,
)


DEFAULT_TS_ROOT = "output/stage0"
TS_NAME_MAP = {
    name: [f"vertebrae_{name}.nii.gz", f"{name}.nii.gz", f"vertebra_{name}.nii.gz"]
    for name in STAGE0_VERTEBRA_LABELS
}


def parse_args():
    parser = argparse.ArgumentParser(description="Validate TotalSegmentator vertebra masks against ground truth.")
    parser.add_argument("--patient-id", type=str, default="", help="Patient id for single-patient visualization.")
    parser.add_argument("--gt-seg", type=str, default="", help="Ground-truth segmentation path for single-patient validation.")
    parser.add_argument("--ts-dir", type=str, default="", help="TotalSegmentator output directory for one patient.")
    parser.add_argument("--ts-root", type=str, default=DEFAULT_TS_ROOT, help="TotalSegmentator output root for all patients.")
    parser.add_argument("--root-dir", type=str, default=DEFAULT_DATA_ROOT, help="Ground-truth dataset root.")
    parser.add_argument("--max-points", type=int, default=12000, help="Max points per rendered subset.")
    parser.add_argument("--point-size", type=float, default=4.0, help="Rendered point size.")
    parser.add_argument("--hide-gt", action="store_true", help="Hide the ground-truth mask in patient-wise visualization.")
    parser.add_argument("--show-ct", action="store_true", help="Show CT bone-context overlay in patient-wise visualization.")
    parser.add_argument("--ct-thresh", type=float, default=150.0, help="HU threshold for CT context points.")
    parser.add_argument("--ct-max-points", type=int, default=30000, help="Max CT context points to render.")
    parser.add_argument("--ct-alpha", type=float, default=0.18, help="Opacity for CT context points.")
    parser.add_argument("--ct-wl", type=float, default=300.0, help="CT window level for context coloring.")
    parser.add_argument("--ct-ww", type=float, default=1500.0, help="CT window width for context coloring.")
    parser.add_argument("--show", action="store_true", help="Open patient-wise vedo visualization.")
    parser.add_argument("--save-csv", type=str, default="", help="Optional CSV path for all-patient metrics.")
    parser.add_argument("--check-output", action="store_true", help="Check generated vertebra masks without requiring ground truth.")
    parser.add_argument("--show-output", action="store_true", help="Show generated vertebra masks over the source CT or MRI volume.")
    return parser.parse_args()


def inspect_stage0_output(patient_id, ts_dir, seg_path=None):
    ts_dir = Path(ts_dir)
    if not ts_dir.exists():
        raise FileNotFoundError(f"TotalSegmentator output dir not found: {ts_dir}")

    combined = None
    if seg_path:
        seg_path = Path(seg_path)
        if not seg_path.exists():
            raise FileNotFoundError(f"Combined segmentation not found: {seg_path}")
        combined, _ = load_seg(str(seg_path))

    rows = []
    for vertebra, label in STAGE0_VERTEBRA_LABELS.items():
        mask, _, path = load_ts_mask(ts_dir, vertebra)
        mask_voxels = 0 if mask is None else int(np.count_nonzero(mask))
        combined_voxels = 0 if combined is None else int(np.count_nonzero(combined == label))
        if mask_voxels == 0 and combined_voxels == 0:
            continue
        rows.append({
            "patient_id": str(patient_id),
            "vertebra": vertebra,
            "mask_voxels": mask_voxels,
            "combined_voxels": combined_voxels,
            "mask_file": "" if path is None else str(path),
        })
    return pd.DataFrame(rows)


def resolve_patient_image(patient_id, root_dir):
    patient_dir = Path(root_dir) / patient_id
    candidates = [
        (patient_dir / f"{patient_id}_ct.nii.gz", "CT"),
        (patient_dir / f"{patient_id}_mr.nii.gz", "MR"),
    ]
    matches = [(path, modality) for path, modality in candidates if path.exists()]
    if not matches:
        raise FileNotFoundError(f"CT or MRI source image not found in: {patient_dir}")
    if len(matches) > 1:
        raise ValueError(f"Multiple source images found in: {patient_dir}")
    return matches[0]


def attach_review_navigation(plotter: Plotter) -> dict[str, str]:
    navigation = {"action": "quit"}

    def on_key(event: object) -> None:
        key = str(getattr(event, "keypress", "")).lower()
        actions = {
            "n": "next_patient", "p": "previous_patient",
            "s": "next_series", "a": "previous_series",
            "q": "quit", "escape": "quit",
        }
        if key in actions:
            navigation["action"] = actions[key]
            plotter.break_interaction()

    plotter.remove_callback("KeyPress")
    plotter.add_callback("KeyPress", on_key, enable_picking=False)
    return navigation


def show_stage0_output(
    patient_id, image_path, modality, seg_path, review_title="", navigation=False
):
    image = nib.as_closest_canonical(nib.load(str(image_path)))
    segmentation = nib.as_closest_canonical(nib.load(str(seg_path)))
    if segmentation.shape != image.shape or not np.allclose(segmentation.affine, image.affine, atol=1e-5):
        segmentation = resample_from_to(segmentation, image, order=0)

    image_xyz = np.asanyarray(image.dataobj).astype(np.float32)
    segmentation_xyz = np.asanyarray(segmentation.dataobj).astype(np.int16)
    image_zyx = np.transpose(image_xyz, (2, 1, 0))
    segmentation_zyx = np.transpose(segmentation_xyz, (2, 1, 0))
    spacing_xyz = spacing_xyz_from_aff(image.affine)
    spacing_zyx = (spacing_xyz[2], spacing_xyz[1], spacing_xyz[0])

    if modality == "CT":
        display = np.clip(image_zyx, -450.0, 1200.0)
        volume_actor = Volume(display).spacing(spacing_zyx).cmap("bone").alpha([0.0, 0.0, 0.04, 0.14])
    else:
        nonzero = image_zyx[image_zyx > 0]
        lower = float(np.percentile(nonzero, 1.0)) if nonzero.size else 0.0
        upper = float(np.percentile(nonzero, 99.5)) if nonzero.size else 1.0
        scale = max(upper - lower, 1.0)
        display = np.clip((image_zyx - lower) / scale, 0.0, 1.0)
        display[image_zyx <= 0] = 0.0
        volume_actor = Volume(display).spacing(spacing_zyx).cmap("gray").alpha([0.0, 0.0, 0.03, 0.12])

    palette = [
        "tomato", "deepskyblue", "gold", "orchid", "limegreen", "cyan",
        "orange", "hotpink", "springgreen", "dodgerblue", "khaki", "salmon",
        "turquoise", "magenta", "wheat", "steelblue", "plum",
    ]
    actors = [volume_actor]
    detected = []
    for index, (vertebra, label) in enumerate(STAGE0_VERTEBRA_LABELS.items()):
        mask = segmentation_zyx == label
        if not np.any(mask):
            continue
        surface = mask_to_surface(
            mask,
            palette[index % len(palette)],
            0.85,
            spacing_zyx,
            smooth=modality == "MR",
        )
        if surface is not None:
            actors.append(surface)
        detected.append(vertebra)

    actors.append(
        Text2D(
            f"{patient_id} | {modality} | Stage 0 vertebrae\n"
            + " ".join(detected)
            + (f"\n{review_title}" if review_title else ""),
            pos="top-left",
            s=0.8,
        )
    )
    plotter = Plotter(
        axes=1, bg="black", bg2="gray3", title=f"{patient_id} {modality} vertebrae"
    )
    state = attach_review_navigation(plotter) if navigation else {"action": "quit"}
    try:
        plotter.show(*actors, interactive=True)
    finally:
        plotter.close()
    return state["action"]


def resolve_gt_path(patient_id, gt_seg, root_dir):
    if gt_seg:
        return Path(gt_seg)
    if patient_id:
        return Path(root_dir) / patient_id / f"{patient_id}_seg-1.nii.gz"
    raise ValueError("Provide either --gt-seg or --patient-id.")


def resolve_ct_path(patient_id, gt_path, root_dir):
    if patient_id:
        path = Path(root_dir) / patient_id / f"{patient_id}_ct.nii.gz"
    else:
        pid = gt_path.stem.replace("_seg-1", "")
        path = gt_path.parent / f"{pid}_ct.nii.gz"
    return path


def load_ct_for_display(ct_path):
    ct_zyx, _, spacing_xyz = _load_ct_and_seg_aligned(str(ct_path), str(ct_path))
    return ct_zyx.astype(np.float32), spacing_xyz


def sample_points(points, max_points):
    if len(points) <= max_points:
        return points
    step = max(int(np.ceil(len(points) / max_points)), 1)
    return points[::step]


def load_ts_mask(ts_dir, vertebra):
    ts_dir = Path(ts_dir)
    for name in TS_NAME_MAP[vertebra]:
        path = ts_dir / name
        if path.exists():
            img = nib.as_closest_canonical(nib.load(str(path)))
            arr = np.asanyarray(img.dataobj)
            return arr > 0.5, img.affine, path
    return None, None, None


def resample_mask_to_gt_grid(mask, mask_aff, gt_shape, gt_aff):
    src = nib.Nifti1Image(mask.astype(np.uint8), affine=mask_aff)
    dst = (gt_shape, gt_aff)
    out = resample_from_to(src, dst, order=0)
    arr = np.asanyarray(out.dataobj)
    return arr > 0.5


def load_mask_zyx_resampled_to_target(mask_path, target_img):
    src = nib.as_closest_canonical(nib.load(str(mask_path)))
    if src.shape != target_img.shape or not np.allclose(src.affine, target_img.affine, atol=1e-5):
        src = resample_from_to(src, target_img, order=0)
    arr_xyz = np.asanyarray(src.dataobj)
    return np.transpose(arr_xyz > 0.5, (2, 1, 0))


def spacing_xyz_from_aff(aff):
    rot = aff[:3, :3]
    sx = float(np.linalg.norm(rot[:, 0]))
    sy = float(np.linalg.norm(rot[:, 1]))
    sz = float(np.linalg.norm(rot[:, 2]))
    return (sx, sy, sz)


def mask_to_surface(mask_zyx, color, alpha, spacing_zyx, smooth=False):
    if mask_zyx is None or not np.any(mask_zyx):
        return None
    surface = Volume(mask_zyx.astype(np.uint8)).spacing(spacing_zyx).isosurface(0.5)
    if smooth:
        surface = surface.smooth(niter=10, pass_band=0.15)
    return surface.c(color).alpha(alpha)


def to_zyx(arr_xyz):
    return np.transpose(arr_xyz, (2, 1, 0))


def coords_from_ijk_zyx(ijk, spacing_xyz):
    ijk = np.asarray(ijk, dtype=np.float64)
    sx, sy, sz = spacing_xyz
    x = ijk[:, 2] * sx
    y = ijk[:, 1] * sy
    z = ijk[:, 0] * sz
    return np.column_stack((x, y, z))


def coords_for_volume_zyx(ijk, spacing_xyz):
    ijk = np.asarray(ijk, dtype=np.float64)
    sx, sy, sz = spacing_xyz
    # Match vedo Volume(arr_zyx).spacing((sz, sy, sx)):
    # display-x <- arr axis 0 (z index), display-y <- axis 1, display-z <- axis 2
    x = ijk[:, 0] * sz
    y = ijk[:, 1] * sy
    z = ijk[:, 2] * sx
    return np.column_stack((x, y, z))


def centroid_world(mask, aff):
    ijk = np.argwhere(mask)
    if ijk.size == 0:
        return None
    xyz = vox2world(aff, ijk)
    return np.median(xyz, axis=0)


def dice_score(a, b):
    a_sum = int(np.count_nonzero(a))
    b_sum = int(np.count_nonzero(b))
    if a_sum == 0 and b_sum == 0:
        return 1.0
    if a_sum == 0 or b_sum == 0:
        return 0.0
    inter = int(np.count_nonzero(a & b))
    return (2.0 * inter) / (a_sum + b_sum)


def iou_score(a, b):
    union = int(np.count_nonzero(a | b))
    if union == 0:
        return 1.0
    inter = int(np.count_nonzero(a & b))
    return inter / union


def evaluate_patient(patient_id, gt_path, ts_dir):
    if not gt_path.exists():
        raise FileNotFoundError(f"Ground-truth seg not found: {gt_path}")
    if not ts_dir.exists():
        raise FileNotFoundError(f"TotalSegmentator output dir not found: {ts_dir}")

    gt_seg, gt_aff = load_seg(str(gt_path))
    gt_shape = gt_seg.shape

    rows = []
    viz = []

    for vertebra, lab in VERTEBRA_LABELS.items():
        gt_mask = gt_seg == lab
        ts_mask, ts_aff, ts_path = load_ts_mask(ts_dir, vertebra)

        gt_count = int(np.count_nonzero(gt_mask))
        ts_count = 0 if ts_mask is None else int(np.count_nonzero(ts_mask))

        # Skip levels absent in both GT and TotalSegmentator for this patient.
        if gt_count == 0 and ts_count == 0:
            continue

        row = {
            "patient_id": str(patient_id),
            "vertebra": vertebra,
            "gt_voxels": gt_count,
            "ts_voxels": ts_count,
            "dice": 0.0 if gt_count > 0 else 1.0,
            "iou": 0.0 if gt_count > 0 else 1.0,
            "centroid_dist_mm": np.nan,
            "ts_file": "",
        }

        if ts_mask is not None:
            if ts_mask.shape != gt_shape or not np.allclose(ts_aff, gt_aff, atol=1e-5):
                ts_mask = resample_mask_to_gt_grid(ts_mask, ts_aff, gt_shape, gt_aff)
                ts_aff = gt_aff

            row["dice"] = dice_score(gt_mask, ts_mask)
            row["iou"] = iou_score(gt_mask, ts_mask)
            gt_cent = centroid_world(gt_mask, gt_aff)
            ts_cent = centroid_world(ts_mask, ts_aff)
            row["centroid_dist_mm"] = float(np.linalg.norm(gt_cent - ts_cent)) if gt_cent is not None and ts_cent is not None else np.nan
            row["ts_file"] = str(ts_path)
        else:
            gt_cent = centroid_world(gt_mask, gt_aff) if gt_count > 0 else None
            ts_cent = None

        rows.append(row)
        viz.append((vertebra, gt_mask, gt_aff, ts_mask, ts_aff if ts_mask is not None else None, gt_cent, ts_cent, row))

    return pd.DataFrame(rows), viz


def show_patient(
    viz,
    max_points,
    point_size,
    patient_id,
    gt_path,
    ct_path=None,
    ct_thresh=150.0,
    ct_max_points=30000,
    ct_alpha=0.18,
    ct_wl=300.0,
    ct_ww=1500.0,
    hide_gt=False,
):
    actors = []
    title_lines = []
    gt_path = Path(gt_path)
    gt_img = nib.as_closest_canonical(nib.load(str(gt_path)))
    ct_img = None

    if ct_path is not None and Path(ct_path).exists():
        ct_path = Path(ct_path)
        ct_img = nib.as_closest_canonical(nib.load(str(ct_path)))
        target_img = ct_img
        gt_for_display = gt_img
        if gt_for_display.shape != target_img.shape or not np.allclose(gt_for_display.affine, target_img.affine, atol=1e-5):
            gt_for_display = resample_from_to(gt_for_display, target_img, order=0)

        ct_xyz = np.asanyarray(ct_img.dataobj).astype(np.float32)
        ct_zyx = np.transpose(ct_xyz, (2, 1, 0))
        gt_seg_zyx = np.transpose(np.asanyarray(gt_for_display.dataobj).astype(np.int16), (2, 1, 0))
        spacing_xyz = spacing_xyz_from_aff(target_img.affine)
        spacing_zyx = (float(spacing_xyz[2]), float(spacing_xyz[1]), float(spacing_xyz[0]))
        wl = float(ct_wl)
        ww = max(float(ct_ww), 1.0)
        vmin = wl - ww / 2.0
        vmax = wl + ww / 2.0
        ct_vis = np.clip(ct_zyx, vmin, vmax)
        ct_actor = (
            Volume(ct_vis)
            .spacing(spacing_zyx)
            .cmap("bone")
            .alpha([0.0, 0.0, float(ct_alpha) * 0.35, float(ct_alpha)])
        )
        actors.append(ct_actor)
    else:
        target_img = gt_img
        gt_seg_zyx = np.transpose(np.asanyarray(target_img.dataobj).astype(np.int16), (2, 1, 0))
        spacing_xyz = spacing_xyz_from_aff(target_img.affine)
        spacing_zyx = (float(spacing_xyz[2]), float(spacing_xyz[1]), float(spacing_xyz[0]))

    for vertebra, gt_mask, gt_aff, ts_mask, ts_aff, gt_cent, ts_cent, row in viz:
        lab = VERTEBRA_LABELS[vertebra]
        gt_disp = (gt_seg_zyx == lab)
        ts_disp = None
        if row["ts_file"]:
            ts_disp = load_mask_zyx_resampled_to_target(row["ts_file"], target_img)

        if not hide_gt and gt_disp is not None and np.count_nonzero(gt_disp) > 0:
            gt_surf = mask_to_surface(gt_disp, "lime", 0.35, spacing_zyx)
            if gt_surf is not None:
                actors.append(gt_surf)
            gt_ijk = np.argwhere(gt_disp)
            gt_xyz = coords_for_volume_zyx(gt_ijk, spacing_xyz)
            actors.append(Sphere(np.median(gt_xyz, axis=0), r=2.5).c("lime"))

        if ts_disp is not None and np.count_nonzero(ts_disp) > 0:
            ts_surf = mask_to_surface(ts_disp, "tomato", 0.35, spacing_zyx)
            if ts_surf is not None:
                actors.append(ts_surf)
            ts_ijk = np.argwhere(ts_disp)
            ts_xyz = coords_for_volume_zyx(ts_ijk, spacing_xyz)
            actors.append(Sphere(np.median(ts_xyz, axis=0), r=2.5).c("tomato"))

        title_lines.append(
            f"{vertebra}: dice={row['dice']:.3f}, iou={row['iou']:.3f}, ctr={row['centroid_dist_mm']:.1f} mm"
        )

    actors.append(
        Text2D(
            f"Patient {patient_id}\n"
            + ("" if hide_gt else "GT=green | ")
            + "TS=red | CT=bone\n"
            + "\n".join(title_lines[:16]),
            pos="top-left",
            s=0.7,
        )
    )
    show(*actors, axes=1, bg="black", bg2="gray3", title="Stage 0 Validation", interactive=True)


def run_batch(root_dir, ts_root):
    rows = []
    root_dir = Path(root_dir)
    ts_root = Path(ts_root)

    for patient_dir in sorted(root_dir.iterdir()):
        if not patient_dir.is_dir():
            continue
        pid = patient_dir.name
        gt_path = patient_dir / f"{pid}_seg-1.nii.gz"
        ts_dir = ts_root / pid
        if not gt_path.exists() or not ts_dir.exists():
            continue
        try:
            df, _ = evaluate_patient(pid, gt_path, ts_dir)
        except Exception as exc:
            rows.append({
                "patient_id": pid,
                "vertebra": "",
                "gt_voxels": np.nan,
                "ts_voxels": np.nan,
                "dice": np.nan,
                "iou": np.nan,
                "centroid_dist_mm": np.nan,
                "ts_file": "",
                "error": str(exc),
            })
            continue
        df["error"] = ""
        rows.append(df)

    if not rows:
        return pd.DataFrame()

    parts = [r if isinstance(r, pd.DataFrame) else pd.DataFrame([r]) for r in rows]
    return pd.concat(parts, ignore_index=True)


def main():
    args = parse_args()

    if args.show_output:
        if not args.patient_id:
            raise ValueError("--show-output requires --patient-id.")
        image_path, modality = resolve_patient_image(args.patient_id, args.root_dir)
        seg_path = Path(args.root_dir) / args.patient_id / f"{args.patient_id}_seg-1.nii.gz"
        if not seg_path.exists():
            raise FileNotFoundError(f"Combined segmentation not found: {seg_path}")
        show_stage0_output(args.patient_id, image_path, modality, seg_path)
        return

    if args.check_output:
        if not args.patient_id:
            raise ValueError("--check-output requires --patient-id.")
        ts_dir = Path(args.ts_dir) if args.ts_dir else Path(args.ts_root) / args.patient_id
        seg_path = Path(args.root_dir) / args.patient_id / f"{args.patient_id}_seg-1.nii.gz"
        df = inspect_stage0_output(args.patient_id, ts_dir, seg_path=seg_path)
        if df.empty:
            raise RuntimeError(f"No vertebrae found for {args.patient_id}.")
        print(df[["vertebra", "mask_voxels", "combined_voxels"]].to_string(index=False))
        print(f"Detected vertebrae: {len(df)}")
        return

    if args.show:
        if not args.patient_id and not args.gt_seg:
            raise ValueError("Patient-wise visualization needs --patient-id or --gt-seg.")
        if not args.ts_dir:
            if not args.patient_id:
                raise ValueError("Use --ts-dir for patient-wise visualization when --patient-id is not provided.")
            ts_dir = Path(args.ts_root) / args.patient_id
        else:
            ts_dir = Path(args.ts_dir)
        gt_path = resolve_gt_path(args.patient_id, args.gt_seg, args.root_dir)
        df, viz = evaluate_patient(args.patient_id or gt_path.stem.replace("_seg-1", ""), gt_path, ts_dir)
        print(df[["vertebra", "gt_voxels", "ts_voxels", "dice", "iou", "centroid_dist_mm"]].to_string(index=False))
        print()
        print(f"Mean dice: {df['dice'].mean():.3f}")
        print(f"Mean IoU: {df['iou'].mean():.3f}")
        ct_path = resolve_ct_path(args.patient_id, gt_path, args.root_dir) if args.show_ct else None
        show_patient(
            viz,
            args.max_points,
            args.point_size,
            args.patient_id or gt_path.parent.name,
            gt_path=gt_path,
            ct_path=ct_path,
            ct_thresh=args.ct_thresh,
            ct_max_points=args.ct_max_points,
            ct_alpha=args.ct_alpha,
            ct_wl=args.ct_wl,
            ct_ww=args.ct_ww,
            hide_gt=args.hide_gt,
        )
        return

    if args.save_csv:
        df = run_batch(args.root_dir, args.ts_root)
        if df.empty:
            print("No patients found for batch validation.")
            return
        out = Path(args.save_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"Saved {out}")
        valid = df[df["vertebra"] != ""]
        if not valid.empty:
            print(f"Mean dice: {valid['dice'].mean():.3f}")
            print(f"Mean IoU: {valid['iou'].mean():.3f}")
        return

    raise ValueError("Use --show for patient-wise validation, or --save-csv for all-patient CSV validation.")


if __name__ == "__main__":
    main()
