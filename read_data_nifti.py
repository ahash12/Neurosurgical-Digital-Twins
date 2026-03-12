import os
import numpy as np
import matplotlib.pyplot as plt
from vedo import Volume, show


DATA_ROOT = "data/Spine-Mets-CT-SEG-Nifti"
PATIENT_ID = "10250"
VERTEBRA_LABEL = None  # Set e.g. 13 for L1; None -> first available label in volume.
CT_ALPHA = [0, 0.02, 0.08, 0.22]
SEG_SURFACE_ALPHA = 0.7


def _load_nifti_with_spacing(path):
    """
    Return volume as numpy array in (z, y, x) order and spacing as (sx, sy, sz).
    """
    try:
        import nibabel as nib

        img = nib.load(path)
        arr_xyz = np.asanyarray(img.dataobj)
        arr_zyx = np.transpose(arr_xyz, (2, 1, 0))
        sx, sy, sz = img.header.get_zooms()[:3]
        return arr_zyx, (float(sx), float(sy), float(sz))
    except Exception:
        try:
            import SimpleITK as sitk

            img = sitk.ReadImage(path)
            arr_zyx = sitk.GetArrayFromImage(img)
            sx, sy, sz = img.GetSpacing()
            return arr_zyx, (float(sx), float(sy), float(sz))
        except Exception as exc:
            raise RuntimeError(
                "Failed to load NIfTI. Install one loader: `pip install nibabel` "
                "or `pip install SimpleITK`."
            ) from exc


def _load_ct_and_seg_aligned(ct_path, seg_path):
    """
    Load CT + SEG as (z, y, x). If SEG grid differs, resample SEG onto CT grid
    using nearest-neighbor interpolation to preserve class labels.
    """
    try:
        import SimpleITK as sitk

        ct_img = sitk.ReadImage(ct_path)
        seg_img = sitk.ReadImage(seg_path)

        if (
            ct_img.GetSize() != seg_img.GetSize()
            or not np.allclose(ct_img.GetSpacing(), seg_img.GetSpacing(), atol=1e-5)
            or not np.allclose(ct_img.GetOrigin(), seg_img.GetOrigin(), atol=1e-5)
            or not np.allclose(ct_img.GetDirection(), seg_img.GetDirection(), atol=1e-5)
        ):
            print("Resampling SEG to CT grid (SimpleITK nearest-neighbor)...")
            seg_img = sitk.Resample(
                seg_img,
                ct_img,
                sitk.Transform(),
                sitk.sitkNearestNeighbor,
                0,
                seg_img.GetPixelID(),
            )

        ct = sitk.GetArrayFromImage(ct_img)
        seg = sitk.GetArrayFromImage(seg_img)
        sx, sy, sz = ct_img.GetSpacing()
        return ct, seg, (float(sx), float(sy), float(sz))
    except Exception:
        import nibabel as nib
        from nibabel.processing import resample_from_to

        ct_img = nib.load(ct_path)
        seg_img = nib.load(seg_path)

        if ct_img.shape != seg_img.shape or not np.allclose(ct_img.affine, seg_img.affine, atol=1e-5):
            print("Resampling SEG to CT grid (nibabel nearest-neighbor)...")
            seg_img = resample_from_to(seg_img, ct_img, order=0)

        ct_xyz = np.asanyarray(ct_img.dataobj)
        seg_xyz = np.asanyarray(seg_img.dataobj)
        ct = np.transpose(ct_xyz, (2, 1, 0))
        seg = np.transpose(seg_xyz, (2, 1, 0))
        sx, sy, sz = ct_img.header.get_zooms()[:3]
        return ct, seg, (float(sx), float(sy), float(sz))


def window_ct(ct, center=400, width=800):
    lower = center - width // 2
    upper = center + width // 2
    ct_windowed = np.clip(ct, lower, upper)
    return (ct_windowed - lower) / (upper - lower)


def save_overlay_slices(ct, seg, patient_id, spacing_xyz, out_dir="slice_outputs_nifti"):
    os.makedirs(out_dir, exist_ok=True)

    sx, sy, sz = spacing_xyz  # x, y, z spacing
    z_dim, y_dim, x_dim = ct.shape

    ct_disp = window_ct(ct, center=400, width=800)

    axial_indices = np.linspace(max(1, z_dim // 6), max(1, z_dim - z_dim // 6), 10, dtype=int)
    sagittal_indices = np.linspace(max(1, x_dim // 6), max(1, x_dim - x_dim // 6), 10, dtype=int)

    axial_extent = [0, x_dim * sx, y_dim * sy, 0]
    sagittal_extent = [0, y_dim * sy, z_dim * sz, 0]

    for idx in axial_indices:
        ct_slice = ct_disp[idx]
        seg_slice = seg[idx]

        plt.figure(figsize=(6, 6))
        plt.imshow(ct_slice, cmap="gray", extent=axial_extent, origin="upper")

        labels = np.unique(seg_slice)
        labels = labels[labels > 0]
        for label in labels:
            mask = (seg_slice == label).astype(np.uint8)
            if np.any(mask):
                plt.contour(
                    mask,
                    levels=[0.5],
                    linewidths=1,
                    extent=axial_extent,
                    origin="upper",
                )

        plt.axis("off")
        plt.title(f"Axial z={idx}")
        plt.savefig(
            os.path.join(out_dir, f"{patient_id}_axial_{idx}.png"),
            bbox_inches="tight",
            dpi=150,
        )
        plt.close()

    for idx in sagittal_indices:
        ct_slice = ct_disp[:, :, idx]
        seg_slice = seg[:, :, idx]

        plt.figure(figsize=(6, 6))
        plt.imshow(ct_slice, cmap="gray", extent=sagittal_extent, origin="upper")

        labels = np.unique(seg_slice)
        labels = labels[labels > 0]
        for label in labels:
            mask = (seg_slice == label).astype(np.uint8)
            if np.any(mask):
                plt.contour(
                    mask,
                    levels=[0.5],
                    linewidths=1,
                    extent=sagittal_extent,
                    origin="upper",
                )

        plt.axis("off")
        plt.title(f"Sagittal x={idx}")
        plt.savefig(
            os.path.join(out_dir, f"{patient_id}_sagittal_{idx}.png"),
            bbox_inches="tight",
            dpi=150,
        )
        plt.close()


def find_case_files(case_dir, patient_id):
    ct_path = os.path.join(case_dir, f"{patient_id}_ct.nii.gz")
    seg_path = os.path.join(case_dir, f"{patient_id}_seg-1.nii.gz")
    if not os.path.exists(ct_path):
        raise FileNotFoundError(f"Missing CT file: {ct_path}")
    if not os.path.exists(seg_path):
        raise FileNotFoundError(f"Missing segmentation file: {seg_path}")
    return ct_path, seg_path


def main():
    case_dir = os.path.join(DATA_ROOT, PATIENT_ID)
    ct_path, seg_path = find_case_files(case_dir, PATIENT_ID)

    ct, seg, spacing_xyz = _load_ct_and_seg_aligned(ct_path, seg_path)
    if ct.shape != seg.shape:
        raise ValueError(f"Failed to align CT/SEG shapes: {ct.shape} vs {seg.shape}")

    seg = seg.astype(np.int16)
    labels = np.unique(seg)
    labels = labels[labels > 0]

    print("CT shape:", ct.shape)
    print("SEG shape:", seg.shape)
    print("SEG labels present:", labels.tolist())

    if len(labels) == 0:
        raise ValueError("No positive segmentation labels found.")

    vertebra_label = int(VERTEBRA_LABEL) if VERTEBRA_LABEL is not None else int(labels[0])
    if vertebra_label not in labels:
        raise ValueError(f"Requested vertebra label {vertebra_label} is not present in this case.")

    vertebra_mask = (seg == vertebra_label).astype(np.uint8)
    print("Example vertebra label:", vertebra_label)

    sx, sy, sz = spacing_xyz
    spacing_zyx = (sz, sy, sx)

    ct = np.clip(ct, -200, 1000)

    ct_vol = Volume(ct).spacing(spacing_zyx).cmap("bone").alpha(CT_ALPHA)

    palette = [
        "tomato", "deepskyblue", "gold", "orchid", "limegreen", "cyan",
        "orange", "hotpink", "springgreen", "dodgerblue", "khaki", "salmon",
        "turquoise", "magenta", "wheat", "steelblue", "plum",
    ]
    seg_actors = []
    for i, label in enumerate(labels):
        label_mask = (seg == label).astype(np.uint8)
        if np.count_nonzero(label_mask) == 0:
            continue
        actor = (
            Volume(label_mask)
            .spacing(spacing_zyx)
            .isosurface(0.5)
            .color(palette[i % len(palette)])
            .alpha(SEG_SURFACE_ALPHA)
        )
        seg_actors.append(actor)

    vertebra_actor = (
        Volume(vertebra_mask)
        .spacing(spacing_zyx)
        .isosurface(0.5)
        .color("lime")
        .alpha(1.0)
    )

    save_overlay_slices(ct, seg, PATIENT_ID, spacing_xyz)
    show(
        ct_vol,
        *seg_actors,
        vertebra_actor,
        axes=1,
        bg="black",
        title=f"Patient {PATIENT_ID} - NIfTI (example vertebra label {vertebra_label})",
    )


if __name__ == "__main__":
    main()
