import argparse

import numpy as np
from vedo import Text2D, Volume, show

from dataset_class import VertebraDataset
from utils import DEFAULT_DATA_ROOT, VERTEBRA_LABELS, extract_centered_label_cube, load_img

def parse_args():
    parser = argparse.ArgumentParser(description="Generate one vertebra patch and display it with vedo.")
    parser.add_argument(
        "--csv-path",
        type=str,
        default="data/vertebra_dataset.csv",
        help="Dataset CSV path.",
    )
    parser.add_argument(
        "--root-dir",
        type=str,
        default=DEFAULT_DATA_ROOT,
        help="Root directory containing patient CT/SEG NIfTI files.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="data/vertebra_patch_cache",
        help="Directory to read/write cached .npy patches.",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Row index in CSV to generate/view.",
    )
    parser.add_argument(
        "--patient-id",
        type=str,
        default="",
        help="Optional patient id override (must be paired with --vertebra).",
    )
    parser.add_argument(
        "--vertebra",
        type=str,
        default="",
        help="Optional vertebra override (e.g., T8, L2).",
    )
    parser.add_argument("--wl", type=float, default=300.0, help="CT window level in HU (default: bone preset).")
    parser.add_argument("--ww", type=float, default=1500.0, help="CT window width in HU (default: bone preset).")
    parser.add_argument("--patch-size", type=int, default=96, help="Cubic patch size (N => NxNxN).")
    parser.add_argument("--seg-alpha", type=float, default=0.45, help="Segmentation overlay opacity.")
    parser.add_argument(
        "--norm-mode",
        type=str,
        default="zscore_sigmoid",
        choices=["zscore_sigmoid", "robust_minmax"],
        help="Foreground intensity normalization mode.",
    )
    parser.add_argument(
        "--zscore-scale",
        type=float,
        default=1.5,
        help="Z-score denominator scaling for sigmoid mode (higher = flatter contrast).",
    )
    parser.add_argument(
        "--foreground-floor",
        type=float,
        default=0.25,
        help="Minimum normalized value assigned to vertebra foreground voxels.",
    )
    return parser.parse_args()


def resolve_index(dataset, args):
    if args.patient_id or args.vertebra:
        if not args.patient_id or not args.vertebra:
            raise ValueError("Use both --patient-id and --vertebra together.")
        matches = dataset.df[
            (dataset.df["patient_id"].astype(str) == str(args.patient_id))
            & (dataset.df["vertebra"].astype(str) == str(args.vertebra))
        ]
        if matches.empty:
            raise ValueError(f"No row found for patient_id={args.patient_id}, vertebra={args.vertebra}")
        return int(matches.index[0])

    if args.index < 0 or args.index >= len(dataset):
        raise IndexError(f"--index out of range (0 to {len(dataset) - 1}): {args.index}")
    return int(args.index)


def main():
    args = parse_args()
    dataset = VertebraDataset(
        csv_path=args.csv_path,
        root_dir=args.root_dir,
        use_patch_cache=True,
        cache_dir=args.cache_dir,
        patient_cache_size=1,
        patch_size=(args.patch_size, args.patch_size, args.patch_size),
        norm_mode=args.norm_mode,
        zscore_scale=args.zscore_scale,
        foreground_floor=args.foreground_floor,
    )
    idx = resolve_index(dataset, args)
    row = dataset.df.iloc[idx]
    pid = str(row["patient_id"])
    vname = str(row["vertebra"])
    label = int(row["label"])
    vid = VERTEBRA_LABELS[vname]

    patch, _ = dataset[idx]
    file_path = dataset._cache_path(pid, vname)
    arr = patch.detach().cpu().numpy().astype(np.float32)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape {arr.shape}")

    # Viewer windowing is applied on already-normalized patch intensities.
    wl_norm = args.wl / 1000.0
    ww_norm = max(args.ww / 1000.0, 1e-6)
    low = wl_norm - (ww_norm / 2.0)
    high = wl_norm + (ww_norm / 2.0)

    vmin = float(np.min(arr))
    vmax = float(np.max(arr))
    volume = (
        Volume(arr)
        .mode(1)
        .cmap("bone", vmin=low, vmax=high)
        .alpha([0.0, 0.03, 0.08, 0.16, 0.30, 0.55, 0.85, 1.0])
    )

    ct_path = f"{args.root_dir}/{pid}/{pid}_ct.nii.gz"
    seg_path = f"{args.root_dir}/{pid}/{pid}_seg-1.nii.gz"
    ct_raw, _ = load_img(ct_path, canonical=False)
    seg_raw, _ = load_img(seg_path, canonical=False)
    seg_bin = np.where(seg_raw == vid, 1.0, 0.0)
    seg_patch = extract_centered_label_cube(seg_bin, seg_raw, vid, size=arr.shape)[0]
    seg_actor = Volume(seg_patch).isosurface(0.5).c("red").alpha(args.seg_alpha)

    title = Text2D(
        (
            f"{file_path.name} | pid={pid} {vname} label={label} | "
            f"shape={arr.shape} | min={vmin:.3f} max={vmax:.3f} | "
            f"WL={args.wl:.0f} WW={args.ww:.0f} mode={args.norm_mode} "
            f"zs={args.zscore_scale:.2f} floor={args.foreground_floor:.2f}"
        ),
        pos="top-left",
        s=0.9,
    )
    print(f"Generated/loaded patch: {file_path}")
    show(volume, title, axes=1, bg="black", bg2="gray3", title="Cached Patch Viewer", interactive=True)


if __name__ == "__main__":
    main()
