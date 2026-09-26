import csv
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import torch.nn as nn
from monai.networks.nets import DenseNet121, resnet10, resnet18
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    cohen_kappa_score,
    confusion_matrix,
    fbeta_score,
    f1_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset, Subset

from dataset_class import (
    CombinedVertebraDataset,
    PATIENT_RELATIVE_CACHE_VERSION,
    PHYSICAL_CONTEXT_CACHE_VERSION,
    Stage1CTWindowAugmentation,
    Stage1PatchAugmentation,
    TransformedDataset,
    VertebraDataset,
    build_archive_stage1_labels,
    resolve_dataset_root,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = resolve_dataset_root(
    default=str(PROJECT_ROOT.parent.parent / "datasets" / "radiel")
)
CSV_PATH = DATA_ROOT / "vertebra_dataset.csv"
ROOT_DIR = DATA_ROOT / "Spine-Mets-CT-SEG-Nifti"
INCLUDE_ARCHIVE_GT = True
ARCHIVE_GT_PATH = DATA_ROOT / "Archive" / "SINS Model Ground Truth.xlsx"
ARCHIVE_ROOT_DIR = DATA_ROOT / "archive_nifti"
ARCHIVE_CSV_PATH = DATA_ROOT / "archive_stage1_labels.csv"
ARCHIVE_MAPPING_PATH = None
ARCHIVE_REQUIRE_SEGMENTATION = True
EXCLUDED_PATIENT_IDS = {"13627", "13977"}

N_FOLDS = 5
INNER_FOLDS = 5
SEED = 42
EPOCHS = 30
EARLY_STOPPING_PATIENCE = 8
BATCH_SIZE = 8
BACKBONE_LR = 1e-5
CLASSIFIER_LR = 1e-4
WEIGHT_DECAY = 1e-4
USE_CLASS_WEIGHTS = True
FALLBACK_DECISION_THRESHOLD = 0.5
MIN_VALIDATION_SPECIFICITY = 0.80
THRESHOLD_F_BETA = 2.0
NUM_WORKERS = 0
PIN_MEMORY = False

PATCH_CACHE_DIR = DATA_ROOT / "vertebra_patch_cache"
PATCH_CACHE_WORKERS = 8
PATIENT_CACHE_SIZE = 1
PATCH_SIZE = (96, 96, 64)
PATCH_MODE = "physical_context"
PHYSICAL_FOV_MM = (128.0, 128.0, 96.0)
INPUT_CHANNELS = 3
PATCH_CONFIGS = {
    "physical_context": {
        "cache_version": PHYSICAL_CONTEXT_CACHE_VERSION,
        "output_group": "model_benchmark",
        "channels": [
            "ct_bone_window_-200_1000",
            "ct_marrow_window_-150_400",
            "vertebra_mask",
        ],
    },
    "patient_relative_context": {
        "cache_version": PATIENT_RELATIVE_CACHE_VERSION,
        "output_group": "medicalnet_resnet10_relative_cv",
        "channels": [
            "patient_relative_context",
            "patient_relative_target_vertebra",
            "vertebra_mask",
        ],
    },
}
PATCH_CONFIG = PATCH_CONFIGS[PATCH_MODE]

USE_TRAIN_AUGMENTATION = True
AUG_FLIP_PROBABILITY = 0.5
AUG_HU_SHIFT = 150.0
AUG_HU_SCALE = 0.15
AUG_NOISE_STD = 0.01

RUN_DATE = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
OUTPUT_DIR = Path("output/stage1") / str(PATCH_CONFIG["output_group"]) / RUN_DATE
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASS_NAMES = ["none", "cancer"]
NUM_CLASSES = len(CLASS_NAMES)
MODEL_VARIANTS = (
    {
        "name": "resnet10_medicalnet",
        "architecture": "resnet10",
        "pretrained": True,
    },
    {
        "name": "resnet10_scratch",
        "architecture": "resnet10",
        "pretrained": False,
    },
    {
        "name": "resnet18_medicalnet",
        "architecture": "resnet18",
        "pretrained": True,
    },
    {
        "name": "resnet18_scratch",
        "architecture": "resnet18",
        "pretrained": False,
    },
    {
        "name": "densenet121_scratch",
        "architecture": "densenet121",
        "pretrained": False,
    },
)


class BinaryVertebraDataset(Dataset):
    def __init__(self, base_dataset):
        self.base = base_dataset
        self.df = base_dataset.df.copy()
        self.df["label"] = (self.df["label"].astype(int) > 0).astype(int)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        patch, label = self.base[index]
        binary_label = 0 if int(label) == 0 else 1
        return patch, torch.tensor(binary_label, dtype=torch.long)


class MedicalNetResNet(nn.Module):
    def __init__(
        self,
        architecture,
        input_channels=INPUT_CHANNELS,
        pretrained=True,
    ):
        super().__init__()
        architecture_options = {
            "resnet10": {
                "factory": resnet10,
                "shortcut_type": "B",
                "bias_downsample": False,
            },
            "resnet18": {
                "factory": resnet18,
                "shortcut_type": "A",
                "bias_downsample": True,
            },
        }
        if architecture not in architecture_options:
            raise ValueError(f"Unsupported MedicalNet architecture: {architecture}")
        options = architecture_options[architecture]
        backbone_input_channels = 1 if pretrained else input_channels
        self.backbone = options["factory"](
            pretrained=pretrained,
            progress=True,
            spatial_dims=3,
            n_input_channels=backbone_input_channels,
            feed_forward=False,
            shortcut_type=options["shortcut_type"],
            bias_downsample=options["bias_downsample"],
        )
        if pretrained and input_channels != 1:
            self._adapt_input_channels(input_channels)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(512, NUM_CLASSES),
        )

    def _adapt_input_channels(self, input_channels):
        old_conv = self.backbone.conv1
        new_conv = nn.Conv3d(
            input_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            dilation=old_conv.dilation,
            groups=old_conv.groups,
            bias=False,
        )
        with torch.no_grad():
            new_conv.weight.zero_()
            intensity_channels = min(input_channels, 2)
            for channel in range(intensity_channels):
                new_conv.weight[:, channel] = old_conv.weight[:, 0] / intensity_channels
        self.backbone.conv1 = new_conv

    def forward(self, inputs):
        features = self.backbone(inputs)
        return self.classifier(features)


class DenseNetClassifier(nn.Module):
    def __init__(self, input_channels=INPUT_CHANNELS):
        super().__init__()
        self.network = DenseNet121(
            spatial_dims=3,
            in_channels=input_channels,
            out_channels=NUM_CLASSES,
            dropout_prob=0.3,
        )

    @property
    def backbone(self):
        return self.network.features

    @property
    def classifier(self):
        return self.network.class_layers

    def forward(self, inputs):
        return self.network(inputs)


def build_model(variant):
    architecture = variant["architecture"]
    if architecture in {"resnet10", "resnet18"}:
        return MedicalNetResNet(
            architecture=architecture,
            input_channels=INPUT_CHANNELS,
            pretrained=variant["pretrained"],
        )
    if architecture == "densenet121":
        return DenseNetClassifier(input_channels=INPUT_CHANNELS)
    raise ValueError(f"Unsupported model architecture: {architecture}")


def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(dataset, indices, shuffle=False, transform=None):
    subset = Subset(dataset, indices)
    if transform is not None:
        subset = TransformedDataset(subset, transform=transform)
    return DataLoader(
        subset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )


def make_train_transform():
    if not USE_TRAIN_AUGMENTATION:
        return None
    if PATCH_MODE == "physical_context":
        return Stage1CTWindowAugmentation(
            flip_probability=AUG_FLIP_PROBABILITY,
            hu_shift=AUG_HU_SHIFT,
            hu_scale=AUG_HU_SCALE,
            noise_std=AUG_NOISE_STD,
        )
    return Stage1PatchAugmentation(
        flip_probability=AUG_FLIP_PROBABILITY,
        noise_std=AUG_NOISE_STD,
        intensity_scale=AUG_HU_SCALE,
        intensity_shift=AUG_HU_SHIFT / 1000.0,
    )


def class_weights_for_indices(dataset, indices):
    labels = dataset.df.iloc[indices]["label"].astype(int).to_numpy()
    counts = np.bincount(labels, minlength=NUM_CLASSES)
    weights = np.zeros(NUM_CLASSES, dtype=np.float32)
    for class_id, count in enumerate(counts):
        if count > 0:
            weights[class_id] = len(labels) / (NUM_CLASSES * count)
    return torch.tensor(weights), counts


def metrics_from_predictions(labels, probabilities, predictions, threshold):
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = np.asarray(predictions, dtype=int)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    true_negative, false_positive, false_negative, true_positive = matrix.ravel()
    sensitivity = true_positive / max(true_positive + false_negative, 1)
    specificity = true_negative / max(true_negative + false_positive, 1)
    return {
        "pr_auc": float(average_precision_score(labels, probabilities)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(
            f1_score(labels, predictions, labels=[0, 1], average="macro")
        ),
        "cohen_kappa": float(cohen_kappa_score(labels, predictions)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "threshold": (None if threshold is None else float(threshold)),
        "confusion_matrix": matrix.tolist(),
        "prediction_counts": np.bincount(
            predictions,
            minlength=NUM_CLASSES,
        ).tolist(),
        "predictions": predictions.tolist(),
    }


def binary_metrics(
    labels,
    probabilities,
    threshold=FALLBACK_DECISION_THRESHOLD,
):
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = (probabilities >= threshold).astype(int)
    return metrics_from_predictions(
        labels,
        probabilities,
        predictions,
        threshold,
    )


def select_validation_threshold(labels, probabilities):
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    candidates = np.unique(np.concatenate(([0.0], probabilities, [1.0])))
    best = None
    for threshold in candidates:
        predictions = (probabilities >= threshold).astype(int)
        matrix = confusion_matrix(labels, predictions, labels=[0, 1])
        true_negative, false_positive, _, _ = matrix.ravel()
        specificity = true_negative / max(
            true_negative + false_positive,
            1,
        )
        if specificity < MIN_VALIDATION_SPECIFICITY:
            continue
        score = fbeta_score(
            labels,
            predictions,
            beta=THRESHOLD_F_BETA,
            zero_division=0,
        )
        candidate = (score, specificity, -float(threshold))
        if best is None or candidate > best[0]:
            best = (candidate, float(threshold))
    if best is None:
        return FALLBACK_DECISION_THRESHOLD
    return best[1]


def apply_threshold(metrics, threshold):
    selected = binary_metrics(
        metrics["labels"],
        metrics["probabilities"],
        threshold=threshold,
    )
    selected["loss"] = metrics["loss"]
    selected["labels"] = metrics["labels"]
    selected["probabilities"] = metrics["probabilities"]
    return selected


def run_epoch(model, loader, criterion, optimizer=None):
    is_training = optimizer is not None
    model.train() if is_training else model.eval()
    total_loss = 0.0
    total_examples = 0
    labels = []
    probabilities = []

    for inputs, targets in loader:
        inputs = inputs.to(DEVICE)
        targets = targets.to(DEVICE)
        if is_training:
            optimizer.zero_grad()
        with torch.set_grad_enabled(is_training):
            logits = model(inputs)
            loss = criterion(logits, targets)
            if is_training:
                loss.backward()
                optimizer.step()
        batch_probabilities = torch.softmax(logits, dim=1)[:, 1]
        total_loss += loss.item() * inputs.size(0)
        total_examples += inputs.size(0)
        labels.extend(targets.detach().cpu().tolist())
        probabilities.extend(batch_probabilities.detach().cpu().tolist())

    metrics = binary_metrics(labels, probabilities)
    metrics["loss"] = total_loss / max(total_examples, 1)
    metrics["labels"] = labels
    metrics["probabilities"] = probabilities
    return metrics


def build_source_datasets():
    original_dataset = VertebraDataset(
        csv_path=CSV_PATH,
        root_dir=ROOT_DIR,
        use_patch_cache=True,
        cache_dir=PATCH_CACHE_DIR,
        patient_cache_size=PATIENT_CACHE_SIZE,
        patch_size=PATCH_SIZE,
        source="original",
        patch_mode=PATCH_MODE,
        physical_fov_mm=PHYSICAL_FOV_MM,
    )
    removed = original_dataset.exclude_patients(EXCLUDED_PATIENT_IDS)
    print(
        f"Excluded {removed} rows from known invalid patients: "
        f"{sorted(EXCLUDED_PATIENT_IDS)}"
    )
    datasets = [original_dataset]

    if INCLUDE_ARCHIVE_GT:
        build_archive_stage1_labels(
            gt_path=ARCHIVE_GT_PATH,
            archive_root=ARCHIVE_ROOT_DIR,
            output_path=ARCHIVE_CSV_PATH,
            mapping_path=ARCHIVE_MAPPING_PATH,
            require_segmentation=ARCHIVE_REQUIRE_SEGMENTATION,
        )
        archive_dataset = VertebraDataset(
            csv_path=ARCHIVE_CSV_PATH,
            root_dir=ARCHIVE_ROOT_DIR,
            use_patch_cache=True,
            cache_dir=PATCH_CACHE_DIR / "archive",
            patient_cache_size=PATIENT_CACHE_SIZE,
            patch_size=PATCH_SIZE,
            source="archive_gt",
            patch_mode=PATCH_MODE,
            physical_fov_mm=PHYSICAL_FOV_MM,
        )
        if len(archive_dataset) > 0:
            datasets.append(archive_dataset)
    return datasets


def precompute_patch_cache(source_datasets):
    for source_dataset in source_datasets:
        start_time = datetime.now()
        print(f"Precomputing patch cache -> {source_dataset.cache_dir}")
        source_dataset.precompute_cache(
            force=False,
            verbose=True,
            num_workers=PATCH_CACHE_WORKERS,
        )
        print(f"Precomputation time: {datetime.now() - start_time}")


def split_indices(dataset):
    labels = dataset.df["label"].astype(int).to_numpy()
    groups = dataset.df["patient_id"].astype(str).to_numpy()
    outer_splitter = StratifiedGroupKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=SEED,
    )
    for fold, (development_indices, test_indices) in enumerate(
        outer_splitter.split(np.zeros(len(labels)), labels, groups),
        start=1,
    ):
        development_labels = labels[development_indices]
        development_groups = groups[development_indices]
        inner_splitter = StratifiedGroupKFold(
            n_splits=INNER_FOLDS,
            shuffle=True,
            random_state=SEED + fold,
        )
        train_relative, validation_relative = next(
            inner_splitter.split(
                np.zeros(len(development_indices)),
                development_labels,
                development_groups,
            )
        )
        yield (
            fold,
            development_indices[train_relative].tolist(),
            development_indices[validation_relative].tolist(),
            test_indices.tolist(),
        )


def index_summary(dataset, indices):
    frame = dataset.df.iloc[indices]
    return {
        "samples": len(frame),
        "patients": int(frame["patient_id"].astype(str).nunique()),
        "label_counts": dict(Counter(frame["label"].astype(int).tolist())),
        "source_counts": dict(Counter(frame["source"].astype(str).tolist())),
        "patient_ids": sorted(frame["patient_id"].astype(str).unique().tolist()),
    }


def save_history(history, path):
    with open(path, "w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)


def save_predictions(dataset, indices, metrics, fold, path):
    with open(path, "w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(
            [
                "fold",
                "patient_id",
                "vertebra",
                "source",
                "label",
                "probability_cancer",
                "decision_threshold",
                "prediction",
            ]
        )
        for index, label, probability, prediction in zip(
            indices,
            metrics["labels"],
            metrics["probabilities"],
            metrics["predictions"],
        ):
            row = dataset.df.iloc[index]
            writer.writerow(
                [
                    fold,
                    row["patient_id"],
                    row["vertebra"],
                    row["source"],
                    label,
                    probability,
                    metrics["threshold"],
                    prediction,
                ]
            )


def train_fold(
    dataset,
    variant,
    fold,
    train_indices,
    validation_indices,
    test_indices,
):
    model_dir = OUTPUT_DIR / variant["name"]
    fold_dir = model_dir / f"fold_{fold:02d}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    set_random_seed(SEED + fold)

    train_loader = make_loader(
        dataset,
        train_indices,
        shuffle=True,
        transform=make_train_transform(),
    )
    validation_loader = make_loader(dataset, validation_indices)
    test_loader = make_loader(dataset, test_indices)

    model = build_model(variant).to(DEVICE)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    class_weights, class_counts = class_weights_for_indices(
        dataset,
        train_indices,
    )
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(DEVICE) if USE_CLASS_WEIGHTS else None
    )
    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": BACKBONE_LR},
            {"params": model.classifier.parameters(), "lr": CLASSIFIER_LR},
        ],
        weight_decay=WEIGHT_DECAY,
    )

    history = []
    best_pr_auc = float("-inf")
    best_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    best_path = fold_dir / "best.pth"

    for epoch in range(1, EPOCHS + 1):
        train_metrics = run_epoch(model, train_loader, criterion, optimizer)
        validation_metrics = run_epoch(model, validation_loader, criterion)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_pr_auc": train_metrics["pr_auc"],
                "train_macro_f1": train_metrics["macro_f1"],
                "val_loss": validation_metrics["loss"],
                "val_pr_auc": validation_metrics["pr_auc"],
                "val_macro_f1": validation_metrics["macro_f1"],
                "val_sensitivity": validation_metrics["sensitivity"],
                "val_specificity": validation_metrics["specificity"],
            }
        )
        print(
            f"{variant['name']} fold {fold}/{N_FOLDS} "
            f"epoch {epoch:02d} | "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_pr_auc={train_metrics['pr_auc']:.4f} | "
            f"val_loss={validation_metrics['loss']:.4f} "
            f"val_pr_auc={validation_metrics['pr_auc']:.4f} "
            f"val_sens={validation_metrics['sensitivity']:.4f} "
            f"val_spec={validation_metrics['specificity']:.4f}"
        )

        improved = validation_metrics["pr_auc"] > best_pr_auc or (
            validation_metrics["pr_auc"] == best_pr_auc
            and validation_metrics["loss"] < best_loss
        )
        if improved:
            best_pr_auc = validation_metrics["pr_auc"]
            best_loss = validation_metrics["loss"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "fold": fold,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "best_val_pr_auc": best_pr_auc,
                    "best_val_loss": best_loss,
                },
                best_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print(
                    f"{variant['name']} fold {fold}: early stopping at epoch {epoch}."
                )
                break

    save_history(history, fold_dir / "history.csv")
    checkpoint = torch.load(best_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    validation_metrics = run_epoch(model, validation_loader, criterion)
    selected_threshold = select_validation_threshold(
        validation_metrics["labels"],
        validation_metrics["probabilities"],
    )
    validation_metrics = apply_threshold(
        validation_metrics,
        selected_threshold,
    )
    test_metrics = apply_threshold(
        run_epoch(model, test_loader, criterion),
        selected_threshold,
    )
    save_predictions(
        dataset,
        validation_indices,
        validation_metrics,
        fold,
        fold_dir / "validation_predictions.csv",
    )
    save_predictions(
        dataset,
        test_indices,
        test_metrics,
        fold,
        fold_dir / "test_predictions.csv",
    )

    fold_result = {
        "model": variant["name"],
        "architecture": variant["architecture"],
        "pretrained": variant["pretrained"],
        "parameter_count": parameter_count,
        "fold": fold,
        "best_epoch": best_epoch,
        "selected_threshold": selected_threshold,
        "class_counts": class_counts.tolist(),
        "train": index_summary(dataset, train_indices),
        "validation": index_summary(dataset, validation_indices),
        "test": index_summary(dataset, test_indices),
        "validation_metrics": {
            key: value
            for key, value in validation_metrics.items()
            if key not in {"labels", "probabilities", "predictions"}
        },
        "test_metrics": {
            key: value
            for key, value in test_metrics.items()
            if key not in {"labels", "probabilities", "predictions"}
        },
    }
    with open(fold_dir / "result.json", "w") as output:
        json.dump(fold_result, output, indent=2)
    print(
        f"{variant['name']} fold {fold} test | "
        f"pr_auc={test_metrics['pr_auc']:.4f} "
        f"f1={test_metrics['macro_f1']:.4f} "
        f"sens={test_metrics['sensitivity']:.4f} "
        f"spec={test_metrics['specificity']:.4f} "
        f"threshold={selected_threshold:.4f}"
    )
    return fold_result, test_metrics, test_indices


def save_fold_assignments(dataset, fold_splits):
    path = OUTPUT_DIR / "fold_assignments.csv"
    with open(path, "w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(
            [
                "fold",
                "role",
                "dataset_index",
                "patient_id",
                "vertebra",
                "source",
                "label",
            ]
        )
        for fold, train_indices, validation_indices, test_indices in fold_splits:
            for role, indices in (
                ("train", train_indices),
                ("validation", validation_indices),
                ("test", test_indices),
            ):
                for index in indices:
                    row = dataset.df.iloc[index]
                    writer.writerow(
                        [
                            fold,
                            role,
                            index,
                            row["patient_id"],
                            row["vertebra"],
                            row["source"],
                            row["label"],
                        ]
                    )


def run_model_variant(dataset, variant, fold_splits):
    model_dir = OUTPUT_DIR / variant["name"]
    model_dir.mkdir(parents=True, exist_ok=True)
    print()
    print(
        f"Starting {variant['name']} "
        f"(architecture={variant['architecture']}, "
        f"pretrained={variant['pretrained']})"
    )
    fold_results = []
    pooled_labels = []
    pooled_probabilities = []
    pooled_predictions = []
    pooled_rows = []
    for fold, train_indices, validation_indices, test_indices in fold_splits:
        result, test_metrics, ordered_test_indices = train_fold(
            dataset,
            variant,
            fold,
            train_indices,
            validation_indices,
            test_indices,
        )
        fold_results.append(result)
        pooled_labels.extend(test_metrics["labels"])
        pooled_probabilities.extend(test_metrics["probabilities"])
        pooled_predictions.extend(test_metrics["predictions"])
        for index, label, probability, prediction in zip(
            ordered_test_indices,
            test_metrics["labels"],
            test_metrics["probabilities"],
            test_metrics["predictions"],
        ):
            row = dataset.df.iloc[index]
            pooled_rows.append(
                [
                    fold,
                    row["patient_id"],
                    row["vertebra"],
                    row["source"],
                    label,
                    probability,
                    test_metrics["threshold"],
                    prediction,
                ]
            )

    pooled_metrics = metrics_from_predictions(
        pooled_labels,
        pooled_probabilities,
        pooled_predictions,
        threshold=None,
    )
    with open(
        model_dir / "out_of_fold_predictions.csv",
        "w",
        newline="",
    ) as output:
        writer = csv.writer(output)
        writer.writerow(
            [
                "fold",
                "patient_id",
                "vertebra",
                "source",
                "label",
                "probability_cancer",
                "decision_threshold",
                "prediction",
            ]
        )
        writer.writerows(pooled_rows)

    summary = {
        "model": variant["name"],
        "architecture": variant["architecture"],
        "pretrained": variant["pretrained"],
        "folds": fold_results,
        "pooled_out_of_fold_metrics": pooled_metrics,
        "mean_test_pr_auc": float(
            np.mean([fold["test_metrics"]["pr_auc"] for fold in fold_results])
        ),
        "std_test_pr_auc": float(
            np.std(
                [fold["test_metrics"]["pr_auc"] for fold in fold_results],
                ddof=1,
            )
        ),
    }
    with open(model_dir / "summary.json", "w") as output:
        json.dump(summary, output, indent=2)
    print(
        f"{variant['name']} pooled out-of-fold | "
        f"pr_auc={pooled_metrics['pr_auc']:.4f} "
        f"f1={pooled_metrics['macro_f1']:.4f} "
        f"sens={pooled_metrics['sensitivity']:.4f} "
        f"spec={pooled_metrics['specificity']:.4f} "
        f"kappa={pooled_metrics['cohen_kappa']:.4f}"
    )
    return summary


def save_benchmark_summary(summaries):
    ranked = sorted(
        summaries,
        key=lambda summary: summary["pooled_out_of_fold_metrics"]["pr_auc"],
        reverse=True,
    )
    with open(OUTPUT_DIR / "benchmark_summary.json", "w") as output:
        json.dump({"models_ranked_by_pr_auc": ranked}, output, indent=2)
    with open(
        OUTPUT_DIR / "benchmark_summary.csv",
        "w",
        newline="",
    ) as output:
        writer = csv.writer(output)
        writer.writerow(
            [
                "rank",
                "model",
                "architecture",
                "pretrained",
                "parameter_count",
                "pr_auc",
                "macro_f1",
                "sensitivity",
                "specificity",
                "cohen_kappa",
                "mean_fold_pr_auc",
                "std_fold_pr_auc",
            ]
        )
        for rank, summary in enumerate(ranked, start=1):
            metrics = summary["pooled_out_of_fold_metrics"]
            writer.writerow(
                [
                    rank,
                    summary["model"],
                    summary["architecture"],
                    summary["pretrained"],
                    summary["folds"][0]["parameter_count"],
                    metrics["pr_auc"],
                    metrics["macro_f1"],
                    metrics["sensitivity"],
                    metrics["specificity"],
                    metrics["cohen_kappa"],
                    summary["mean_test_pr_auc"],
                    summary["std_test_pr_auc"],
                ]
            )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    set_random_seed(SEED)
    source_datasets = build_source_datasets()
    precompute_patch_cache(source_datasets)
    base_dataset = (
        source_datasets[0]
        if len(source_datasets) == 1
        else CombinedVertebraDataset(source_datasets)
    )
    dataset = BinaryVertebraDataset(base_dataset)
    fold_splits = list(split_indices(dataset))
    save_fold_assignments(dataset, fold_splits)

    params = {
        "device": DEVICE,
        "models": list(MODEL_VARIANTS),
        "selection_metric": "validation_pr_auc",
        "threshold_selection": {
            "source": "fold_validation",
            "metric": f"f{THRESHOLD_F_BETA:g}",
            "minimum_specificity": MIN_VALIDATION_SPECIFICITY,
            "fallback": FALLBACK_DECISION_THRESHOLD,
        },
        "n_folds": N_FOLDS,
        "inner_folds": INNER_FOLDS,
        "seed": SEED,
        "epochs": EPOCHS,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "batch_size": BATCH_SIZE,
        "backbone_lr": BACKBONE_LR,
        "classifier_lr": CLASSIFIER_LR,
        "weight_decay": WEIGHT_DECAY,
        "use_class_weights": USE_CLASS_WEIGHTS,
        "patch_mode": PATCH_MODE,
        "patch_size": list(PATCH_SIZE),
        "physical_fov_mm": list(PHYSICAL_FOV_MM),
        "input_channels": INPUT_CHANNELS,
        "channel_definitions": PATCH_CONFIG["channels"],
        "cache_version": PATCH_CONFIG["cache_version"],
        "augmentation": {
            "flip_probability": AUG_FLIP_PROBABILITY,
            "hu_shift": AUG_HU_SHIFT,
            "hu_scale": AUG_HU_SCALE,
            "noise_std": AUG_NOISE_STD,
        },
        "excluded_patient_ids": sorted(EXCLUDED_PATIENT_IDS),
        "source_counts": dict(Counter(dataset.df["source"].astype(str))),
        "label_counts": dict(Counter(dataset.df["label"].astype(int))),
        "patient_count": int(dataset.df["patient_id"].astype(str).nunique()),
    }
    with open(OUTPUT_DIR / "param.json", "w") as output:
        json.dump(params, output, indent=2)

    print(f"Device: {DEVICE}")
    print(f"Samples: {len(dataset)}")
    print(f"Patients: {params['patient_count']}")
    print(f"Models: {len(MODEL_VARIANTS)}")
    print(f"Output dir: {OUTPUT_DIR}")
    print("MedicalNet weights are downloaded by MONAI on first use and cached locally.")

    summaries = []
    for variant in MODEL_VARIANTS:
        summaries.append(run_model_variant(dataset, variant, fold_splits))
        save_benchmark_summary(summaries)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print(f"Saved benchmark ranking: {OUTPUT_DIR / 'benchmark_summary.csv'}")
