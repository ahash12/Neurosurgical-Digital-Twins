import csv
import json
import os
import random
import sys
from collections import Counter
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import cast

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import nibabel as nib
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from pipeline.core.stage1_multimodal import (
    ARCHITECTURE,
    CLASS_NAMES,
    MODALITIES,
    CTOrMRIResNet10,
    MultimodalPatchConfig,
    MultimodalVertebraDataset,
    VertebraSample,
    load_ct_samples,
    load_mri_samples,
    validate_samples,
)
from pipeline.scripts.stage3_postlateral import mri_sequence_family
from utils import VERTEBRA_LABELS


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = (
    Path(
        os.environ.get(
            "RADIEL_DATA_ROOT", PROJECT_ROOT.parent.parent / "datasets" / "radiel"
        )
    )
    .expanduser()
    .resolve()
)
MRI_ROOT = PROJECT_ROOT / "output" / "trial_b" / "nifti"
MRI_LABELS_PATH = PROJECT_ROOT / "output" / "trial_b" / "stage1_binary_labels.csv"
REVIEW_MANIFEST_PATH = PROJECT_ROOT / "output" / "trial_b" / "review_manifest.csv"
CT_SOURCES = (
    (
        "original",
        DATA_ROOT / "vertebra_dataset.csv",
        DATA_ROOT / "Spine-Mets-CT-SEG-Nifti",
    ),
    (
        "archive_gt",
        DATA_ROOT / "archive_stage1_labels.csv",
        DATA_ROOT / "archive_nifti",
    ),
)
EXCLUDED_ORIGINAL_PATIENTS = {"13627", "13977"}
PATCH_CONFIG = MultimodalPatchConfig()
CACHE_DIR = PROJECT_ROOT / "output" / "stage1" / "multimodal_patch_cache"
OUTPUT_ROOT = PROJECT_ROOT / "output" / "stage1" / "ct_or_mri_binary"
SEED = 42
SPLIT_FOLDS = 5
SPLIT_SEARCH_ATTEMPTS = 100
EPOCHS = 40
PATIENCE = 8
BATCH_SIZE = 8
NUM_WORKERS = 0
PRETRAINED = True
BACKBONE_LR = 1e-5
CLASSIFIER_LR = 1e-4
WEIGHT_DECAY = 1e-4
DECISION_THRESHOLD = 0.5
AUG_FLIP_PROBABILITY = 0.5
AUG_INTENSITY_SCALE = 0.08
AUG_INTENSITY_SHIFT = 0.03
AUG_NOISE_STD = 0.01


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_label_template() -> None:
    if MRI_LABELS_PATH.exists():
        raise FileExistsError(f"Refusing to overwrite MRI GT: {MRI_LABELS_PATH}")
    rows = []
    with REVIEW_MANIFEST_PATH.open(newline="", encoding="utf-8") as stream:
        for record in csv.DictReader(stream):
            scan_id = record["scan_id"]
            seg_path = MRI_ROOT / scan_id / f"{scan_id}_seg-1.nii.gz"
            if not seg_path.is_file():
                continue
            segmentation = cast(nib.Nifti1Image, nib.load(seg_path))
            labels = set(np.unique(np.asanyarray(segmentation.dataobj)))
            for vertebra, label in VERTEBRA_LABELS.items():
                if label in labels:
                    rows.append(
                        {
                            "patient_id": record["patient_id"],
                            "scan_id": scan_id,
                            "vertebra": vertebra,
                            "label": "",
                            "sequence": record["sequence"],
                            "spine_coverage_reviewed": record[
                                "spine_coverage_reviewed"
                            ],
                            "group_id": "",
                        }
                    )
    if not rows:
        raise ValueError("No segmented T1-L5 vertebrae available for a label template.")
    write_csv(MRI_LABELS_PATH, rows)
    print(f"Wrote {len(rows)} unlabeled candidates: {MRI_LABELS_PATH}")
    print("Label 0=none, 1=cancer. Blank rows are excluded, never assumed normal.")


def build_samples() -> list[VertebraSample]:
    # Require MRI GT before any CT loading, cache computation or model downloads.
    samples = load_mri_samples(MRI_LABELS_PATH, MRI_ROOT)
    with REVIEW_MANIFEST_PATH.open(newline="", encoding="utf-8") as stream:
        identities = {
            row["scan_id"]: row["patient_id"] for row in csv.DictReader(stream)
        }
    for sample in samples:
        if identities.get(sample.scan_id) != sample.patient_id:
            raise ValueError(
                f"MRI GT patient/scan identity differs from Trial B inventory: {sample.scan_id}"
            )
    for source, csv_path, root in CT_SOURCES:
        excluded = EXCLUDED_ORIGINAL_PATIENTS if source == "original" else set()
        samples.extend(load_ct_samples(csv_path, root, source, excluded))
    validate_samples(samples)
    return samples


def split_samples(samples: list[VertebraSample]) -> dict[str, list[int]]:
    groups = np.asarray([sample.group_id for sample in samples])
    strata = np.asarray([f"{sample.modality}:{sample.label}" for sample in samples])
    for modality in MODALITIES:
        for label in (0, 1):
            patients = {
                s.group_id
                for s in samples
                if s.modality == modality and s.label == label
            }
            if len(patients) < 3:
                raise ValueError(
                    f"Need at least 3 patient groups with {modality} label {label} for train/val/test; found {len(patients)}. This is a split minimum, not evidence of adequate sample size."
                )
    if len(set(groups)) < SPLIT_FOLDS:
        raise ValueError("Not enough patient groups for configured split folds.")
    for attempt in range(SPLIT_SEARCH_ATTEMPTS):
        splitter = StratifiedGroupKFold(
            SPLIT_FOLDS, shuffle=True, random_state=SEED + attempt
        )
        folds = [
            held_out.tolist()
            for _, held_out in splitter.split(np.zeros(len(samples)), strata, groups)
        ]
        splits = {
            "test": folds[0],
            "val": folds[1],
            "train": [i for fold in folds[2:] for i in fold],
        }
        required = {
            f"{modality}:{label}" for modality in MODALITIES for label in (0, 1)
        }
        if not all(set(strata[indices]) == required for indices in splits.values()):
            continue
        training_families = {
            mri_sequence_family(samples[i].sequence)
            for i in splits["train"]
            if samples[i].modality == "MR"
        }
        all_families = {
            mri_sequence_family(s.sequence) for s in samples if s.modality == "MR"
        }
        if training_families != all_families:
            continue
        group_sets = [set(groups[indices]) for indices in splits.values()]
        assert all(
            not group_sets[i] & group_sets[j] for i in range(3) for j in range(i)
        )
        return splits
    raise ValueError(
        "Could not create patient-disjoint splits containing both classes in CT and MRI, with all MRI sequence families represented in training. Add labels/data or revise split settings; never split scans from the same patient."
    )


def sampling_weights(samples: list[VertebraSample]) -> torch.Tensor:
    groups = {m: {s.group_id for s in samples if s.modality == m} for m in MODALITIES}
    scans: dict[tuple[str, str], set[str]] = {}
    counts = Counter((s.modality, s.group_id, s.scan_id) for s in samples)
    for sample in samples:
        scans.setdefault((sample.modality, sample.group_id), set()).add(sample.scan_id)
    return torch.tensor(
        [
            1.0
            / (
                len(groups[s.modality])
                * len(scans[(s.modality, s.group_id)])
                * counts[(s.modality, s.group_id, s.scan_id)]
            )
            for s in samples
        ],
        dtype=torch.double,
    )


def augment(inputs: torch.Tensor) -> torch.Tensor:
    inputs = inputs.clone()
    for sample in inputs:
        for axis in (1, 2, 3):
            if torch.rand(()) < AUG_FLIP_PROBABILITY:
                sample.copy_(torch.flip(sample, dims=(axis,)))
        scale = 1 + (torch.rand((), device=sample.device) * 2 - 1) * AUG_INTENSITY_SCALE
        shift = (torch.rand((), device=sample.device) * 2 - 1) * AUG_INTENSITY_SHIFT
        sample[0] = (
            sample[0] * scale + shift + torch.randn_like(sample[0]) * AUG_NOISE_STD
        ).clamp(0, 1)
    return inputs


def evaluate_metrics(rows: list[dict]) -> dict:
    metrics = {}
    for name in ("all", *MODALITIES):
        selected = (
            rows if name == "all" else [row for row in rows if row["modality"] == name]
        )
        labels = [r["label"] for r in selected]
        probabilities = [r["cancer_probability"] for r in selected]
        predictions = [int(p >= DECISION_THRESHOLD) for p in probabilities]
        matrix = confusion_matrix(labels, predictions, labels=[0, 1])
        tn, fp, fn, tp = matrix.ravel()
        metrics[name] = {
            "count": len(labels),
            "accuracy": float(accuracy_score(labels, predictions)),
            "macro_f1": float(
                f1_score(
                    labels, predictions, labels=[0, 1], average="macro", zero_division=0
                )
            ),
            "kappa": float(cohen_kappa_score(labels, predictions)),
            "auroc": float(roc_auc_score(labels, probabilities)),
            "average_precision": float(average_precision_score(labels, probabilities)),
            "sensitivity": float(tp / (tp + fn)),
            "specificity": float(tn / (tn + fp)),
            "confusion_matrix": matrix.tolist(),
        }
    return metrics


def run_epoch(
    model: CTOrMRIResNet10,
    loader: DataLoader,
    dataset: MultimodalVertebraDataset,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, list[dict]]:
    model.train(optimizer is not None)
    total_loss, count = 0.0, 0
    rows = []
    for inputs, targets, domains, indices in loader:
        inputs, targets, domains = (
            inputs.to(device),
            targets.to(device),
            domains.to(device),
        )
        if optimizer is not None:
            inputs = augment(inputs)
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(optimizer is not None):
            logits = model(inputs, domains)
            loss = criterion(logits, targets)
            if optimizer is not None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
        total_loss += float(loss.item()) * len(targets)
        count += len(targets)
        probabilities = logits.detach().softmax(dim=1)[:, 1].cpu().tolist()
        for index, probability in zip(indices.tolist(), probabilities):
            sample = dataset.samples[index]
            rows.append(
                {
                    "source": sample.source,
                    "group_id": sample.group_id,
                    "patient_id": sample.patient_id,
                    "scan_id": sample.scan_id,
                    "vertebra": sample.vertebra,
                    "modality": sample.modality,
                    "sequence": sample.sequence,
                    "label": sample.label,
                    "cancer_probability": probability,
                    "prediction": int(probability >= DECISION_THRESHOLD),
                }
            )
    return total_loss / count, rows


def main(check_data: bool = False, write_template: bool = False) -> None:
    if write_template:
        write_label_template()
        return
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    samples = build_samples()
    splits = split_samples(samples)
    for name, indices in splits.items():
        print(
            f"{name}: {len(indices)} rows, {len({samples[i].group_id for i in indices})} patient groups; {dict(Counter((samples[i].modality, samples[i].label) for i in indices))}"
        )
    if check_data:
        print("Data and split checks passed. No cache computation or training started.")
        return
    output_dir = OUTPUT_ROOT / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir.mkdir(parents=True, exist_ok=False)
    sequence_families = sorted(
        {
            mri_sequence_family(samples[i].sequence)
            for i in splits["train"]
            if samples[i].modality == "MR"
        }
    )
    params = {
        "architecture": ARCHITECTURE,
        "modalities": list(MODALITIES),
        "class_names": CLASS_NAMES,
        "patch_config": asdict(PATCH_CONFIG),
        "normalization": "modality_groupnorm",
        "sequence_families": sequence_families,
        "supported_vertebrae": list(VERTEBRA_LABELS),
        "decision_threshold": DECISION_THRESHOLD,
        "seed": SEED,
        "pretrained": PRETRAINED,
        "epochs": EPOCHS,
        "patience": PATIENCE,
        "batch_size": BATCH_SIZE,
        "backbone_lr": BACKBONE_LR,
        "classifier_lr": CLASSIFIER_LR,
        "weight_decay": WEIGHT_DECAY,
        "sampler": "equal_modality_patient_scan",
        "loss": "sqrt_inverse_sampled_class_frequency_ce",
        "split_folds": SPLIT_FOLDS,
        "mri_labels": str(MRI_LABELS_PATH),
        "ct_sources": [
            (name, str(csv_path), str(root)) for name, csv_path, root in CT_SOURCES
        ],
        "augmentation": {
            "flip_probability": AUG_FLIP_PROBABILITY,
            "scale": AUG_INTENSITY_SCALE,
            "shift": AUG_INTENSITY_SHIFT,
            "noise_std": AUG_NOISE_STD,
        },
    }
    (output_dir / "param.json").write_text(
        json.dumps(params, indent=2), encoding="utf-8"
    )
    split_rows = []
    for split, indices in splits.items():
        for index in indices:
            row = asdict(samples[index])
            row.update(
                image_path=str(row["image_path"]),
                seg_path=str(row["seg_path"]),
                split=split,
            )
            split_rows.append(row)
    write_csv(output_dir / "splits.csv", split_rows)
    dataset = MultimodalVertebraDataset(samples, PATCH_CONFIG, CACHE_DIR)
    dataset.precompute_cache()
    training_samples = [samples[i] for i in splits["train"]]
    weights = sampling_weights(training_samples)
    loaders = {}
    for name, indices in splits.items():
        sampler = (
            WeightedRandomSampler(weights.tolist(), len(indices), replacement=True)
            if name == "train"
            else None
        )
        loaders[name] = DataLoader(
            Subset(dataset, indices),
            batch_size=BATCH_SIZE,
            sampler=sampler,
            num_workers=NUM_WORKERS,
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CTOrMRIResNet10(pretrained=PRETRAINED).to(device)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": BACKBONE_LR},
            {"params": model.classifier.parameters(), "lr": CLASSIFIER_LR},
        ],
        weight_decay=WEIGHT_DECAY,
    )
    sampled_mass = torch.stack(
        [
            weights[torch.tensor([s.label == label for s in training_samples])].sum()
            for label in (0, 1)
        ]
    )
    class_weights = (sampled_mass.sum() / (2 * sampled_mass)).sqrt().float().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    validation_criterion = nn.CrossEntropyLoss()
    history, best_score, stale = [], -1.0, 0
    print(f"Device: {device} | output: {output_dir}")
    for epoch in range(1, EPOCHS + 1):
        train_loss, _ = run_epoch(
            model, loaders["train"], dataset, criterion, device, optimizer
        )
        val_loss, val_rows = run_epoch(
            model, loaders["val"], dataset, validation_criterion, device
        )
        metrics = evaluate_metrics(val_rows)
        score = (metrics["CT"]["macro_f1"] + metrics["MR"]["macro_f1"]) / 2
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_ct_f1": metrics["CT"]["macro_f1"],
                "val_mr_f1": metrics["MR"]["macro_f1"],
                "selection_score": score,
            }
        )
        write_csv(output_dir / "history.csv", history)
        print(
            f"Epoch {epoch:02d} | train_loss={train_loss:.4f} val_loss={val_loss:.4f} | CT_f1={metrics['CT']['macro_f1']:.4f} MR_f1={metrics['MR']['macro_f1']:.4f}"
        )
        torch.save(model.state_dict(), output_dir / "last.pth")
        if score > best_score:
            best_score, stale = score, 0
            torch.save(model.state_dict(), output_dir / "best.pth")
            write_csv(output_dir / "best_val_predictions.csv", val_rows)
            (output_dir / "best_val_metrics.json").write_text(
                json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8"
            )
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    model.load_state_dict(
        torch.load(output_dir / "best.pth", map_location=device, weights_only=True)
    )
    test_loss, test_rows = run_epoch(
        model, loaders["test"], dataset, validation_criterion, device
    )
    write_csv(output_dir / "test_predictions.csv", test_rows)
    test_metrics = evaluate_metrics(test_rows)
    test_metrics["loss"] = test_loss
    (output_dir / "test_metrics.json").write_text(
        json.dumps(test_metrics, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(f"Test metrics: {json.dumps(test_metrics)}")
