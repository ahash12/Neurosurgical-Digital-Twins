import os
import argparse
import numpy as np
import pandas as pd
from utils import VERTEBRA_LABELS, INV_LABELS, load_seg, vox2world

LIT_FILE = "data/vertebra_height.csv"

def measure_height_from_ijk(ijk, aff):
    if ijk.size == 0:
        return None
    xyz = vox2world(aff, ijk)
    x, y, z = xyz[:,0], xyz[:,1], xyz[:,2]
    cx = np.median(x)
    keep = np.abs(x - cx) < 15

    x = x[keep]
    y = y[keep]
    z = z[keep]

    if len(x) < 200:
        return None

    y_ant = np.percentile(y, 90)
    y_post = np.percentile(y, 10)

    def height_at_y(y0):
        slab = np.abs(y - y0) < 3
        if slab.sum() < 20:
            slab = np.abs(y - y0) < 6
        if slab.sum() < 10:
            return None
        zz = z[slab]
        return zz.max() - zz.min()

    h_ant = height_at_y(y_ant)
    h_post = height_at_y(y_post)

    if h_ant is None or h_post is None:
        return None

    return h_ant, h_post


def measure_height(seg, aff, label):
    ijk = np.argwhere(seg == label)
    return measure_height_from_ijk(ijk, aff)


def build_label_ijk_map(seg):
    """Build per-label voxel coordinates with a single pass over segmentation."""
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

def collapse_score(r):
    if r < 0.5:
        return 3
    if r < 0.8:
        return 2
    return 0

def load_lit():
    if not os.path.exists(LIT_FILE):
        return {}
    df = pd.read_csv(LIT_FILE)
    return {
        r["vertebra"]: (
            r["mean_anterior_mm"],
            r["mean_posterior_mm"]
        )
        for _, r in df.iterrows()
    }

def run_batch(root):
    lit = load_lit()
    rows = []
    
    for pid in os.listdir(root):
        pdir = os.path.join(root, pid)
        if not os.path.isdir(pdir):
            continue
        seg_path = os.path.join(
            pdir,
            f"{pid}_seg-1.nii.gz"
        )
        if not os.path.exists(seg_path):
            continue

        seg, aff = load_seg(seg_path)
        label_ijk = build_label_ijk_map(seg)
        for lab in range(1,18):
            ijk = label_ijk.get(lab, np.empty((0, 3), dtype=np.int64))
            res = measure_height_from_ijk(ijk, aff)
            if res is None:
                continue
            h_ant, h_post = res
            rows.append({
                "patient_id": pid,
                "vertebra": INV_LABELS[lab],
                "ant": h_ant,
                "post": h_post
            })

    df = pd.DataFrame(rows)
    mean = df.groupby("vertebra")[["ant","post"]].mean()
    mean_map = {
        k:(v["ant"], v["post"])
        for k,v in mean.to_dict("index").items()
    }

    out = []

    for _, r in df.iterrows():
        v = r["vertebra"]
        ant = r["ant"]
        post = r["post"]

        c_ant, c_post = mean_map[v]

        if v in lit:
            l_ant, l_post = lit[v]
        else:
            l_ant, l_post = c_ant, c_post

        normal_ant = max(c_ant, l_ant)
        normal_post = max(c_post, l_post)

        ratio_ant = ant / normal_ant
        ratio_post = post / normal_post

        worst = min(ratio_ant, ratio_post)

        score = collapse_score(worst)

        out.append({
            "patient_id": r["patient_id"],
            "vertebra": v,

            "ant": ant,
            "post": post,

            "ratio": worst,
            "collapse_score": score
        })

    out = pd.DataFrame(out)
    out.to_csv("stage2_collapse.csv", index=False)
    print("saved stage2_collapse.csv")


def run_single(seg_path, vertebra):
    seg, aff = load_seg(seg_path)
    label = VERTEBRA_LABELS[vertebra]
    res = measure_height(seg, aff, label)
    print(res)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--root", type=str)
    parser.add_argument("--seg", type=str)
    parser.add_argument("--vertebra", type=str)

    args = parser.parse_args()

    if args.root:
        run_batch(args.root)

    elif args.seg:
        run_single(args.seg, args.vertebra)
