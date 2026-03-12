import os
import numpy as np
import pydicom
import SimpleITK as sitk
from vedo import Volume, show
import matplotlib.pyplot as plt

DATA_ROOT = "data/manifest-1724965242274/Spine-Mets-CT-SEG"
PATIENT_ID = "10250"

def save_overlay_slices(ct, seg, patient_id, spacing_xyz, out_dir="slice_outputs"):
    os.makedirs(out_dir, exist_ok=True)

    sx, sy, sz = spacing_xyz  # x, y, z spacing
    z_dim, y_dim, x_dim = ct.shape

    ct_disp = window_ct(ct, center=400, width=800)

    axial_indices = np.linspace(200, z_dim - 400, 10, dtype=int)
    sagittal_indices = np.linspace(200, x_dim - 200, 10, dtype=int)

    # physical extents for proper scaling
    axial_extent = [0, x_dim * sx, y_dim * sy, 0]
    sagittal_extent = [0, y_dim * sy, z_dim * sz, 0]

    # ----- AXIAL -----
    for idx in axial_indices:
        ct_slice = ct_disp[idx]
        seg_slice = seg[idx]

        plt.figure(figsize=(6, 6))
        plt.imshow(ct_slice, cmap="gray",
                   extent=axial_extent, origin="upper")

        mask = seg_slice > 0
        if np.any(mask):
            plt.contour(mask.astype(np.uint8),
                        levels=[0.5],
                        colors="red",
                        linewidths=1,
                        extent=axial_extent,
                        origin="upper")

        plt.axis("off")
        plt.title(f"Axial z={idx}")
        plt.savefig(os.path.join(out_dir,
                    f"{patient_id}_axial_{idx}.png"),
                    bbox_inches="tight", dpi=150)
        plt.close()

    # ----- SAGITTAL -----
    for idx in sagittal_indices:
        ct_slice = ct_disp[:, :, idx]
        seg_slice = seg[:, :, idx]

        plt.figure(figsize=(6, 6))
        plt.imshow(ct_slice, cmap="gray",
                   extent=sagittal_extent, origin="upper")

        mask = seg_slice > 0
        if np.any(mask):
            plt.contour(mask.astype(np.uint8),
                        levels=[0.5],
                        colors="red",
                        linewidths=1,
                        extent=sagittal_extent,
                        origin="upper")

        plt.axis("off")
        plt.title(f"Sagittal x={idx}")
        plt.savefig(os.path.join(out_dir,
                    f"{patient_id}_sagittal_{idx}.png"),
                    bbox_inches="tight", dpi=150)
        plt.close()


def find_ct_and_seg(patient_dir):
    ct_folder = None
    seg_path = None

    for study in os.listdir(patient_dir):
        study_path = os.path.join(patient_dir, study)
        if not os.path.isdir(study_path):
            continue

        for sub in os.listdir(study_path):
            sub_path = os.path.join(study_path, sub)
            if not os.path.isdir(sub_path):
                continue

            name = sub.upper()

            if "SKINTOSKIN" in name:
                ct_folder = sub_path

            if "SEGMENTATION" in name:
                for f in os.listdir(sub_path):
                    if f.lower().endswith(".dcm"):
                        seg_path = os.path.join(sub_path, f)

    return ct_folder, seg_path


def load_ct_slices(ct_folder):
    files = [os.path.join(ct_folder, f)
             for f in os.listdir(ct_folder)
             if f.endswith(".dcm")]

    slices = [pydicom.dcmread(f) for f in files]

    slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))

    volume = []

    for s in slices:
        img = s.pixel_array.astype(np.float32)

        slope = float(getattr(s, "RescaleSlope", 1.0))
        intercept = float(getattr(s, "RescaleIntercept", 0.0))

        img = img * slope + intercept  # convert to HU
        volume.append(img)

    volume = np.stack(volume)

    uids = [s.SOPInstanceUID for s in slices]

    spacing = (
        float(slices[0].PixelSpacing[0]),
        float(slices[0].PixelSpacing[1]),
        abs(slices[1].ImagePositionPatient[2] -
            slices[0].ImagePositionPatient[2])
    )

    print("HU range:", volume.min(), volume.max())

    return volume, uids, spacing


def window_ct(ct, center=300, width=1500):
    lower = center - width // 2
    upper = center + width // 2

    ct_windowed = np.clip(ct, lower, upper)
    ct_windowed = (ct_windowed - lower) / (upper - lower)
    return ct_windowed



def build_seg_volume(seg_path, ct_uids, ct_shape):
    seg_ds = pydicom.dcmread(seg_path)
    seg_frames = seg_ds.pixel_array

    seg_volume = np.zeros(ct_shape, dtype=np.uint8)

    for i, frame in enumerate(seg_ds.PerFrameFunctionalGroupsSequence):
        ref_uid = (
            frame.DerivationImageSequence[0]
            .SourceImageSequence[0]
            .ReferencedSOPInstanceUID
        )

        if ref_uid not in ct_uids:
            continue

        z = ct_uids.index(ref_uid)
        seg_volume[z] = np.maximum(seg_volume[z], seg_frames[i])

    return seg_volume


def main():
    patient_dir = os.path.join(DATA_ROOT, PATIENT_ID)
    ct_folder, seg_path = find_ct_and_seg(patient_dir)

    ct, ct_uids, spacing_xyz = load_ct_slices(ct_folder)
    seg = build_seg_volume(seg_path, ct_uids, ct.shape)
    seg = np.flip(seg, axis=1)
    # seg = np.flip(seg, axis=2)

    print("CT shape:", ct.shape)
    print("SEG shape:", seg.shape)
    print("SEG labels:", np.unique(seg))

    # convert spacing to numpy order (z,y,x)
    sx, sy, sz = spacing_xyz
    spacing_zyx = (sz, sy, sx)

    ct = np.clip(ct, -200, 1000)

    ct_vol = Volume(ct).spacing(spacing_zyx).cmap("bone")#.alpha([0, 0.05, 0.2, 0.8])
    seg_vol = Volume(seg).spacing(spacing_zyx).cmap("jet").alpha([0, 0, 0.7, 1])
    save_overlay_slices(ct, seg, PATIENT_ID, spacing_xyz)
    show(ct_vol, seg_vol, axes=1, bg="black", title=f"Patient {PATIENT_ID}")


if __name__ == "__main__":
    main()
