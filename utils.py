import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F

VERTEBRA_LABELS = {
    "T1": 1, "T2": 2, "T3": 3, "T4": 4, "T5": 5, "T6": 6, "T7": 7, "T8": 8, "T9": 9, "T10": 10,
    "T11": 11, "T12": 12,
    "L1": 13, "L2": 14, "L3": 15, "L4": 16, "L5": 17,
}
INV_LABELS = {v: k for k, v in VERTEBRA_LABELS.items()}
VERTEBRAE = list(VERTEBRA_LABELS.keys())

DEFAULT_DATA_ROOT = "data/Spine-Mets-CT-SEG-Nifti"

def load_img(path, canonical=True):
    img = nib.load(path)
    if canonical:
        img = nib.as_closest_canonical(img)
    data = img.get_fdata()
    aff = img.affine
    return data, aff

def load_seg(seg_path):
    seg, aff = load_img(seg_path, canonical=True)
    return seg.astype(np.int16), aff

def vox2world(aff, ijk):
    ijk_h = np.concatenate(
        [ijk, np.ones((ijk.shape[0], 1), dtype=np.float64)],
        axis=1,
    )
    xyz_h = (aff @ ijk_h.T).T
    return xyz_h[:, :3]

def crop_from_mask(vol, mask, label, margin=5):
    coords = np.where(mask == label)
    if len(coords[0]) == 0:
        return None

    z0, z1 = coords[0].min(), coords[0].max()
    y0, y1 = coords[1].min(), coords[1].max()
    x0, x1 = coords[2].min(), coords[2].max()

    z0 = max(0, z0 - margin)
    y0 = max(0, y0 - margin)
    x0 = max(0, x0 - margin)

    z1 = min(vol.shape[0], z1 + margin)
    y1 = min(vol.shape[1], y1 + margin)
    x1 = min(vol.shape[2], x1 + margin)

    return vol[z0:z1, y0:y1, x0:x1]


def extract_centered_label_cube(vol, mask, label, size=(64, 64, 64)):
    """Keep only the requested label voxels and center them in a fixed-size zero-padded cube."""
    if vol.shape != mask.shape:
        # Some studies have slight CT/SEG size mismatches; use the overlapping region.
        common_shape = tuple(min(v, m) for v, m in zip(vol.shape, mask.shape))
        vol = vol[:common_shape[0], :common_shape[1], :common_shape[2]]
        mask = mask[:common_shape[0], :common_shape[1], :common_shape[2]]

    target_mask = mask == label
    if not np.any(target_mask):
        return np.zeros((1, size[0], size[1], size[2]), dtype=np.float32)

    isolated = np.where(target_mask, vol, 0.0)
    coords = np.argwhere(target_mask)

    mins = coords.min(axis=0)
    maxs = coords.max(axis=0) + 1
    src = isolated[mins[0]:maxs[0], mins[1]:maxs[1], mins[2]:maxs[2]].astype(np.float32)

    # If the vertebra crop is larger than target cube, shrink it first to avoid truncation.
    src_shape = src.shape
    if any(src_shape[d] > size[d] for d in range(3)):
        scale = min(size[d] / src_shape[d] for d in range(3))
        new_shape = tuple(max(1, int(round(src_shape[d] * scale))) for d in range(3))
        src_t = torch.from_numpy(src).float()[None, None]
        src = F.interpolate(
            src_t,
            size=new_shape,
            mode="trilinear",
            align_corners=False,
        )[0, 0].cpu().numpy()

    out = np.zeros(size, dtype=np.float32)
    src_shape = src.shape

    src_slices = []
    dst_slices = []
    for dim, dim_size in enumerate(size):
        src_dim = src_shape[dim]
        if src_dim <= dim_size:
            dst_start = (dim_size - src_dim) // 2
            dst_end = dst_start + src_dim
            src_start = 0
            src_end = src_dim
        else:
            src_start = (src_dim - dim_size) // 2
            src_end = src_start + dim_size
            dst_start = 0
            dst_end = dim_size

        src_slices.append(slice(src_start, src_end))
        dst_slices.append(slice(dst_start, dst_end))

    out[dst_slices[0], dst_slices[1], dst_slices[2]] = src[src_slices[0], src_slices[1], src_slices[2]]
    return out[None, ...]

def resize_patch(patch, size=(64, 64, 64)):
    t = torch.tensor(patch).float()[None, None]
    t = F.interpolate(
        t,
        size=size,
        mode="trilinear",
        align_corners=False,
    )
    return t[0]
