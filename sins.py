import os
import torch
import numpy as np

from monai.networks.nets import DenseNet121
from utils import (
    VERTEBRA_LABELS,
    INV_LABELS,
    DEFAULT_DATA_ROOT,
    load_img,
    crop_from_mask,
    resize_patch,
)

from stage2_collapse import measure_height, collapse_score
from stage3_postlateral import posterolateral_score
from stage4_alignment import compute_patient_alignment

ROOT = DEFAULT_DATA_ROOT
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_stage1(model_path=None):
    if model_path is None:
        model_path = "stage1_lesion_model.pt"
    model = DenseNet121(
        spatial_dims=3,
        in_channels=1,
        out_channels=4
    )
    model.load_state_dict(
        torch.load(model_path, map_location=DEVICE)
    )

    model.to(DEVICE)
    model.eval()

    return model

def location_score(v):
    # junctional
    if v in ["T1","T2","T11","T12","L1","L5"]:
        return 3
    # mobile
    if v in ["L2","L3","L4"]:
        return 2
    # semirigid
    if v in ["T3","T4","T5","T6","T7","T8","T9","T10"]:
        return 1
    return 0

def bone_score(label):
    if label == 0:
        return 0
    if label == 1:
        return 0
    if label == 2:
        return 1
    if label == 3:
        return 2
    return 0


def run(patient_id):
    model = load_stage1()
    pdir = os.path.join(ROOT, patient_id)
    ct_path = os.path.join(pdir, f"{patient_id}_ct.nii.gz")
    seg_path = os.path.join(pdir, f"{patient_id}_seg-1.nii.gz")

    ct, aff = load_img(ct_path, canonical=True)
    seg, _ = load_img(seg_path, canonical=True)

    # stage4 once
    align_summary, _ = compute_patient_alignment(
        seg_path,
        6,
        10,
        10
    )
    align_score = align_summary["alignment_score"]

    results = []
    for lab in range(1,18):
        vname = INV_LABELS[lab]
        patch = crop_from_mask(ct, seg, lab)

        if patch is None:
            continue

        patch = np.clip(patch, -200, 1000)
        patch = resize_patch(patch)
        patch = patch.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            out = model(patch)
            cls = out.argmax().item()

        if cls == 0:
            continue

        # Collapse score
        h = measure_height(seg, aff, lab)

        if h is None:
            collapse = 0
        else:
            ant, post = h
            ratio = min(ant, post) / max(ant, post)
            collapse = collapse_score(ratio)

        # Posterolateral score
        post_score = posterolateral_score(
            ct,
            seg,
            aff,
            lab
        )

        # Location Score
        loc = location_score(vname)
        
        # Bone Score
        bone = bone_score(cls)

        sins = (loc + bone + collapse + post_score + align_score)
        sins_category = "Stable"
        if sins >= 10:
            sins_category = "Unstable"
        elif sins >= 4:
            sins_category = "Potentially Unstable"
        
        results.append({
            "vertebra": vname,
            "SINS": sins,
            "SINS_category": sins_category,
            "loc": loc,
            "bone": bone,
            "collapse": collapse,
            "posterolateral": post_score,
            "align": align_score
        })

    results = sorted(
        results,
        key=lambda x: x["SINS"],
        reverse=True
    )

    print("Top 3")
    for r in results[:3]:
        print(r.iloc[:3, :])

if __name__ == "__main__":
    pid = input("Patient ID: ")
    run(pid)
