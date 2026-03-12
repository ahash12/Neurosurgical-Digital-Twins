import os
import argparse
import numpy as np
import pandas as pd
from utils import INV_LABELS, load_img, vox2world


def lesion_mask(ct):
    return np.logical_or(
        ct < 100,
        ct > 600
    )


def align_ct_seg(ct, seg):
    if ct.shape == seg.shape:
        return ct, seg
    common_shape = tuple(min(c, s) for c, s in zip(ct.shape, seg.shape))
    ct = ct[:common_shape[0], :common_shape[1], :common_shape[2]]
    seg = seg[:common_shape[0], :common_shape[1], :common_shape[2]]
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


def posterolateral_score_from_ijk(ct, aff, ijk):
    if ijk.size == 0:
        return 0

    xyz = vox2world(aff, ijk)
    x = xyz[:,0]
    y = xyz[:,1]
    cx = np.median(x)
    cy = np.median(y)
    ct_vals = ct[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
    abnormal = np.logical_or(
        ct_vals < 100,
        ct_vals > 600
    )

    if abnormal.sum()==0:
        return 0
    x_abn = x[abnormal]
    y_abn = y[abnormal]

    # posterior only
    posterior = y_abn < cy

    if posterior.sum()==0:
        return 0

    x_abn = x_abn[posterior]
    left_count = int((x_abn < cx).sum())
    right_count = int((x_abn > cx).sum())
    total = left_count + right_count
    if total == 0:
        return 0

    left_flag = left_count > 50
    right_flag = right_count > 50

    if left_flag and right_flag:
        left_frac = left_count / total
        right_frac = right_count / total
        # If one side strongly dominates, treat as unilateral.
        if left_frac >= 0.70 or right_frac >= 0.70:
            return 1
        return 3
    if left_flag or right_flag:
        return 1
    return 0

def run(root):
    rows=[]
    for pid in os.listdir(root):
        pdir = os.path.join(root,pid)
        if not os.path.isdir(pdir):
            continue
        seg_path = os.path.join(pdir,f"{pid}_seg-1.nii.gz")
        ct_path  = os.path.join(pdir,f"{pid}_ct.nii.gz")
        
        if not os.path.exists(seg_path):
            continue

        seg,aff = load_img(seg_path)
        ct,_    = load_img(ct_path)
        ct, seg = align_ct_seg(ct, seg)
        label_ijk = build_label_ijk_map(seg)

        for lab in range(1,18):
            ijk = label_ijk.get(lab, np.empty((0, 3), dtype=np.int64))
            score = posterolateral_score_from_ijk(ct, aff, ijk)
            rows.append({
                "patient_id":pid,
                "vertebra":INV_LABELS[lab],
                "posterolateral_score":score
            })

    df = pd.DataFrame(rows)
    df.to_csv(
        "stage3_posterolateral.csv",
        index=False
    )
    print("saved stage3_posterolateral.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root")
    args = parser.parse_args()
    run(args.root)
