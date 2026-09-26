import csv
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, cohen_kappa_score
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler

from dataset_class import (
    CombinedVertebraDataset,
    STAGE1_CACHE_VERSION,
    Stage1PatchAugmentation,
    TransformedDataset,
    VertebraDataset,
    build_archive_stage1_labels,
    resolve_dataset_root,
)
from monai.networks.nets import DenseNet121

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

BATCH_SIZE = 16
EPOCHS = 50
LR = 5e-5
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
USE_WEIGHTED_SAMPLER = False
USE_CLASS_WEIGHTS = True
USE_FOCAL_LOSS = True
FOCAL_GAMMA = 2.0
SPLIT_SEARCH_TRIALS = 5000
SEED = 42
NUM_WORKERS = 0
PIN_MEMORY = False
USE_PATCH_CACHE = True
PREBUILD_PATCH_CACHE = True
PATCH_CACHE_DIR = DATA_ROOT / "vertebra_patch_cache"
PATCH_CACHE_WORKERS = 8
PATIENT_CACHE_SIZE = 1
PATCH_SIZE = (96, 96, 64)
NORM_MODE = "ct_hu_window"
ZSCORE_SCALE = 1.5
FOREGROUND_FLOOR = 0.15
USE_TRAIN_AUGMENTATION = True
AUG_FLIP_PROBABILITY = 0.5
AUG_NOISE_STD = 0.02
AUG_INTENSITY_SCALE = 0.08
AUG_INTENSITY_SHIFT = 0.04

RUN_DATE = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
OUTPUT_DIR = Path("output/stage1/2_class") / RUN_DATE
BEST_MODEL_PATH = OUTPUT_DIR / "best.pth"
LAST_MODEL_PATH = OUTPUT_DIR / "last.pth"
HISTORY_PATH = OUTPUT_DIR / "history.csv"
PARAMS_PATH = OUTPUT_DIR / "param.json"
CURVES_PATH = OUTPUT_DIR / "curves.png"
TRAIN_CONFUSION_PATH = OUTPUT_DIR / "train_confusion_matrix.csv"
TRAIN_REPORT_PATH = OUTPUT_DIR / "train_classification_report.txt"
TRAIN_REPORT_JSON_PATH = OUTPUT_DIR / "train_classification_report.json"
TRAIN_PREDICTIONS_PATH = OUTPUT_DIR / "train_predictions.csv"
VAL_CONFUSION_PATH = OUTPUT_DIR / "best_val_confusion_matrix.csv"
VAL_REPORT_PATH = OUTPUT_DIR / "best_val_classification_report.txt"
VAL_REPORT_JSON_PATH = OUTPUT_DIR / "best_val_classification_report.json"
VAL_PREDICTIONS_PATH = OUTPUT_DIR / "best_val_predictions.csv"
TEST_CONFUSION_PATH = OUTPUT_DIR / "test_confusion_matrix.csv"
TEST_REPORT_PATH = OUTPUT_DIR / "test_classification_report.txt"
TEST_REPORT_JSON_PATH = OUTPUT_DIR / "test_classification_report.json"
TEST_PREDICTIONS_PATH = OUTPUT_DIR / "test_predictions.csv"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASS_NAMES = ["none", "cancer"]
NUM_CLASSES = len(CLASS_NAMES)
SPLIT_MODE = "patient_balanced"


class BinaryVertebraDataset(Dataset):
    def __init__(self, base_dataset):
        self.base = base_dataset
        self.df = base_dataset.df.copy()
        self.df["label"] = (self.df["label"].astype(int) > 0).astype(int)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        patch, label = self.base[idx]
        return patch, torch.tensor(0 if int(label) == 0 else 1, dtype=torch.long)


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = nn.functional.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        loss = ((1.0 - pt) ** self.gamma) * ce
        if self.alpha is not None:
            loss = self.alpha.to(logits.device)[targets] * loss
        return loss.mean()


def build_patient_balanced_splits(
    dataset, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO, seed=SEED
):
    patients = []
    for pid, group in dataset.df.groupby("patient_id", sort=False):
        counts = torch.zeros(NUM_CLASSES, dtype=torch.float32)
        for label in group["label"].astype(int).tolist():
            counts[label] += 1.0
        patients.append(
            {
                "patient_id": str(pid),
                "indices": group.index.tolist(),
                "counts": counts,
                "size": len(group),
            }
        )

    n_total = len(patients)
    n_train = max(1, int(round(n_total * train_ratio)))
    n_val = max(1, int(round(n_total * val_ratio)))
    if n_train + n_val >= n_total:
        n_val = max(1, n_total - n_train - 1)

    total_counts = torch.zeros(NUM_CLASSES, dtype=torch.float32)
    for patient in patients:
        total_counts += patient["counts"]

    target_counts = {
        "train": total_counts * (n_train / n_total),
        "val": total_counts * (n_val / n_total),
        "test": total_counts * ((n_total - n_train - n_val) / n_total),
    }

    def candidate_score(groups):
        score = 0.0
        for split_name, group in groups.items():
            counts = torch.zeros(NUM_CLASSES, dtype=torch.float32)
            for patient in group:
                counts += patient["counts"]
            target = target_counts[split_name]
            score += torch.sum(
                torch.abs(counts - target) / torch.clamp(target, min=1.0)
            ).item()
            for class_id in range(1, NUM_CLASSES):
                if total_counts[class_id] > 0 and counts[class_id] == 0:
                    score += 25.0 if split_name == "train" else 5.0
        return score

    generator = torch.Generator().manual_seed(seed)
    best_groups = None
    best_score = float("inf")
    for _ in range(SPLIT_SEARCH_TRIALS):
        order = torch.randperm(n_total, generator=generator).tolist()
        ordered = [patients[i] for i in order]
        groups = {
            "train": ordered[:n_train],
            "val": ordered[n_train : n_train + n_val],
            "test": ordered[n_train + n_val :],
        }
        score = candidate_score(groups)
        if score < best_score:
            best_score = score
            best_groups = groups

    return tuple(
        sorted(index for patient in best_groups[name] for index in patient["indices"])
        for name in ("train", "val", "test")
    )


def make_loader(subset, shuffle=False):
    return DataLoader(
        subset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )


def make_weighted_train_loader(dataset, indices, transform=None):
    labels = dataset.df.iloc[indices]["label"].astype(int).tolist()
    counts = Counter(labels)
    sample_weights = [1.0 / max(counts[label], 1) for label in labels]
    sampler = WeightedRandomSampler(
        torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )
    subset = Subset(dataset, indices)
    if transform is not None:
        subset = TransformedDataset(subset, transform=transform)
    return DataLoader(
        subset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )


def label_counts(df, indices):
    return dict(Counter(df.iloc[indices]["label"].astype(int).tolist()))


def split_source_label_counts(df, indices):
    summary = {}
    split_df = df.iloc[indices]
    for source, source_df in split_df.groupby("source", sort=True):
        summary[str(source)] = dict(Counter(source_df["label"].astype(int).tolist()))
    return summary


def split_patient_ids(df, indices):
    return sorted(df.iloc[indices]["patient_id"].astype(str).unique().tolist())


def compute_class_weights(df, indices):
    labels = df.iloc[indices]["label"].astype(int).tolist()
    counts = torch.zeros(NUM_CLASSES, dtype=torch.float32)
    for label in labels:
        counts[label] += 1.0
    total = counts.sum().item()
    weights = torch.zeros(NUM_CLASSES, dtype=torch.float32)
    for c in range(NUM_CLASSES):
        if counts[c] > 0:
            weights[c] = total / (NUM_CLASSES * counts[c])
    return weights, counts


def build_report_from_confusion(conf):
    eps = 1e-8
    f1_scores = []
    for c in range(conf.size(0)):
        tp = conf[c, c].item()
        fp = conf[:, c].sum().item() - tp
        fn = conf[c, :].sum().item() - tp
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1_scores.append((2 * precision * recall) / (precision + recall + eps))
    return sum(f1_scores) / len(f1_scores)


def run_epoch(model, loader, criterion, optimizer=None, collect_predictions=False):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    conf = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.long)
    y_true, y_pred = [], []

    for x, y in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        if is_train:
            optimizer.zero_grad()
        with torch.set_grad_enabled(is_train):
            out = model(x)
            loss = criterion(out, y)
            if is_train:
                loss.backward()
                optimizer.step()
        preds = out.argmax(dim=1)
        total_loss += loss.item() * x.size(0)
        total_correct += (preds == y).sum().item()
        total_examples += x.size(0)
        y_cpu = y.detach().cpu()
        preds_cpu = preds.detach().cpu()
        for t, p in zip(y_cpu, preds_cpu):
            conf[int(t), int(p)] += 1
        if collect_predictions:
            y_true.extend(int(v) for v in y_cpu.tolist())
            y_pred.extend(int(v) for v in preds_cpu.tolist())

    metrics = {
        "loss": total_loss / max(total_examples, 1),
        "acc": total_correct / max(total_examples, 1),
        "f1": build_report_from_confusion(conf),
        "confusion_matrix": conf,
        "pred_counts": conf.sum(dim=0).tolist(),
    }
    if collect_predictions:
        metrics["y_true"] = y_true
        metrics["y_pred"] = y_pred
    return metrics


def save_confusion_matrix_csv(conf, path):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true/pred", *CLASS_NAMES])
        for i, row in enumerate(conf.tolist()):
            writer.writerow([CLASS_NAMES[i], *row])


def save_classification_report(y_true, y_pred, txt_path, json_path):
    labels = list(range(NUM_CLASSES))
    kappa = cohen_kappa_score(y_true, y_pred, labels=labels)
    report_text = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=CLASS_NAMES,
        zero_division=0,
        digits=4,
    )
    report_json = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=CLASS_NAMES,
        zero_division=0,
        digits=4,
        output_dict=True,
    )
    report_json["cohen_kappa"] = float(kappa)
    with open(txt_path, "w") as f:
        f.write(report_text)
        f.write(f"\nCohen kappa: {kappa:.4f}\n")
    with open(json_path, "w") as f:
        json.dump(report_json, f, indent=2)


def save_split_reports(
    metrics, confusion_path, report_path, report_json_path, split_name
):
    save_confusion_matrix_csv(metrics["confusion_matrix"], confusion_path)
    save_classification_report(
        metrics["y_true"], metrics["y_pred"], report_path, report_json_path
    )
    print(f"Saved {split_name} confusion matrix: {confusion_path}")
    print(f"Saved {split_name} classification report: {report_path}")


def save_prediction_csv(dataset, indices, metrics, path):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "patient_id",
                "vertebra",
                "source",
                "label",
                "label_name",
                "prediction",
                "prediction_name",
            ]
        )
        for index, y_true, y_pred in zip(indices, metrics["y_true"], metrics["y_pred"]):
            row = dataset.df.iloc[index]
            writer.writerow(
                [
                    row["patient_id"],
                    row["vertebra"],
                    row.get("source", ""),
                    y_true,
                    CLASS_NAMES[y_true],
                    y_pred,
                    CLASS_NAMES[y_pred],
                ]
            )
    print(f"Saved predictions: {path}")


def build_base_dataset():
    original_dataset = VertebraDataset(
        csv_path=CSV_PATH,
        root_dir=ROOT_DIR,
        use_patch_cache=USE_PATCH_CACHE,
        cache_dir=PATCH_CACHE_DIR,
        patient_cache_size=PATIENT_CACHE_SIZE,
        patch_size=PATCH_SIZE,
        norm_mode=NORM_MODE,
        zscore_scale=ZSCORE_SCALE,
        foreground_floor=FOREGROUND_FLOOR,
        source="original",
    )
    removed = original_dataset.exclude_patients(EXCLUDED_PATIENT_IDS)
    print(
        f"Excluded {removed} rows from known invalid patients: {sorted(EXCLUDED_PATIENT_IDS)}"
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
            use_patch_cache=USE_PATCH_CACHE,
            cache_dir=Path(PATCH_CACHE_DIR) / "archive",
            patient_cache_size=PATIENT_CACHE_SIZE,
            patch_size=PATCH_SIZE,
            norm_mode=NORM_MODE,
            zscore_scale=ZSCORE_SCALE,
            foreground_floor=FOREGROUND_FLOOR,
            source="archive_gt",
        )
        if len(archive_dataset) > 0:
            datasets.append(archive_dataset)
    if len(datasets) == 1:
        return datasets[0], datasets
    return CombinedVertebraDataset(datasets), datasets


def main() -> None:
    base_dataset, source_datasets = build_base_dataset()
    dataset = BinaryVertebraDataset(base_dataset)

    if PREBUILD_PATCH_CACHE and USE_PATCH_CACHE:
        for source_dataset in source_datasets:
            start_time = datetime.now()
            print(f"Precomputing patch cache -> {source_dataset.cache_dir}")
            source_dataset.precompute_cache(
                force=False,
                verbose=True,
                num_workers=PATCH_CACHE_WORKERS,
            )
            print(f"Precomputation time: {datetime.now() - start_time}")

    train_idx, val_idx, test_idx = build_patient_balanced_splits(dataset)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_set: Subset | TransformedDataset = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)
    test_set = Subset(dataset, test_idx)

    train_transform = (
        Stage1PatchAugmentation(
            flip_probability=AUG_FLIP_PROBABILITY,
            noise_std=AUG_NOISE_STD,
            intensity_scale=AUG_INTENSITY_SCALE,
            intensity_shift=AUG_INTENSITY_SHIFT,
        )
        if USE_TRAIN_AUGMENTATION
        else None
    )
    if train_transform is not None:
        train_set = TransformedDataset(train_set, transform=train_transform)

    if USE_WEIGHTED_SAMPLER:
        train_loader = make_weighted_train_loader(
            dataset, train_idx, transform=train_transform
        )
    else:
        train_loader = make_loader(train_set, shuffle=True)
    eval_train_loader = make_loader(Subset(dataset, train_idx), shuffle=False)
    val_loader = make_loader(val_set, shuffle=False)
    test_loader = make_loader(test_set, shuffle=False)

    print(f"Device: {DEVICE}")
    print(f"Split mode: {SPLIT_MODE}")
    print(f"Weighted sampler: {USE_WEIGHTED_SAMPLER}")
    print(f"Class weights: {USE_CLASS_WEIGHTS}")
    print(f"Focal loss: {USE_FOCAL_LOSS}")
    print(f"Train augmentation: {USE_TRAIN_AUGMENTATION}")
    print(
        f"Count: train - {len(train_set)} val - {len(val_set)} test - {len(test_set)}"
    )
    print(f"Train label counts: {label_counts(dataset.df, train_idx)}")
    print(f"Val label counts:   {label_counts(dataset.df, val_idx)}")
    print(f"Test label counts:  {label_counts(dataset.df, test_idx)}")
    print(f"Source counts: {dict(Counter(dataset.df['source'].astype(str).tolist()))}")
    print(f"Output dir: {OUTPUT_DIR}")

    class_weights, train_counts = compute_class_weights(dataset.df, train_idx)
    print(f"Class weights (train): {class_weights.tolist()}")

    params = {
        "csv_path": str(CSV_PATH),
        "root_dir": str(ROOT_DIR),
        "include_archive_gt": INCLUDE_ARCHIVE_GT,
        "archive_gt_path": str(ARCHIVE_GT_PATH),
        "archive_root_dir": str(ARCHIVE_ROOT_DIR),
        "archive_csv_path": str(ARCHIVE_CSV_PATH),
        "archive_mapping_path": str(ARCHIVE_MAPPING_PATH),
        "archive_require_segmentation": ARCHIVE_REQUIRE_SEGMENTATION,
        "excluded_patient_ids": sorted(EXCLUDED_PATIENT_IDS),
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "lr": LR,
        "split_mode": SPLIT_MODE,
        "use_weighted_sampler": USE_WEIGHTED_SAMPLER,
        "use_class_weights": USE_CLASS_WEIGHTS,
        "use_focal_loss": USE_FOCAL_LOSS,
        "focal_gamma": FOCAL_GAMMA,
        "use_train_augmentation": USE_TRAIN_AUGMENTATION,
        "patch_size": list(PATCH_SIZE),
        "norm_mode": NORM_MODE,
        "zscore_scale": ZSCORE_SCALE,
        "foreground_floor": FOREGROUND_FLOOR,
        "cache_version": STAGE1_CACHE_VERSION,
        "train_label_counts": label_counts(dataset.df, train_idx),
        "val_label_counts": label_counts(dataset.df, val_idx),
        "test_label_counts": label_counts(dataset.df, test_idx),
        "train_source_label_counts": split_source_label_counts(dataset.df, train_idx),
        "val_source_label_counts": split_source_label_counts(dataset.df, val_idx),
        "test_source_label_counts": split_source_label_counts(dataset.df, test_idx),
        "train_patient_ids": split_patient_ids(dataset.df, train_idx),
        "val_patient_ids": split_patient_ids(dataset.df, val_idx),
        "test_patient_ids": split_patient_ids(dataset.df, test_idx),
        "source_counts": dict(Counter(dataset.df["source"].astype(str).tolist())),
        "class_weights": [float(x) for x in class_weights.tolist()],
        "class_count_vector": [int(x) for x in train_counts.tolist()],
    }
    with open(PARAMS_PATH, "w") as f:
        json.dump(params, f, indent=2)

    model = DenseNet121(spatial_dims=3, in_channels=1, out_channels=NUM_CLASSES).to(
        DEVICE
    )
    criterion: nn.Module
    if USE_FOCAL_LOSS:
        alpha = class_weights.to(DEVICE) if USE_CLASS_WEIGHTS else None
        criterion = FocalLoss(alpha=alpha, gamma=FOCAL_GAMMA)
    else:
        weight = class_weights.to(DEVICE) if USE_CLASS_WEIGHTS else None
        criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    history = []
    best_val_loss = float("inf")
    best_val_f1 = float("-inf")
    best_epoch = 0

    for epoch in range(1, EPOCHS + 1):
        train_metrics = run_epoch(model, train_loader, criterion, optimizer=optimizer)
        val_metrics = run_epoch(model, val_loader, criterion, optimizer=None)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_acc": train_metrics["acc"],
                "train_f1": train_metrics["f1"],
                "val_loss": val_metrics["loss"],
                "val_acc": val_metrics["acc"],
                "val_f1": val_metrics["f1"],
            }
        )
        print(
            f"Epoch {epoch:02d} | train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['acc']:.4f} "
            f"train_f1={train_metrics['f1']:.4f} | val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['acc']:.4f} val_f1={val_metrics['f1']:.4f} | "
            f"val_pred={val_metrics['pred_counts']}"
        )
        if val_metrics["f1"] > best_val_f1 or (
            val_metrics["f1"] == best_val_f1 and val_metrics["loss"] < best_val_loss
        ):
            best_val_f1 = val_metrics["f1"]
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                    "best_val_f1": best_val_f1,
                },
                BEST_MODEL_PATH,
            )
            print(
                f"Saved new best model at epoch {epoch} "
                f"(val_f1={best_val_f1:.4f}, val_loss={best_val_loss:.4f}) -> {BEST_MODEL_PATH}"
            )

    with open(HISTORY_PATH, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "train_loss",
                "train_acc",
                "train_f1",
                "val_loss",
                "val_acc",
                "val_f1",
            ],
        )
        writer.writeheader()
        writer.writerows(history)
    print(f"Saved history: {HISTORY_PATH}")

    torch.save(
        {
            "epoch": EPOCHS,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "best_val_f1": best_val_f1,
        },
        LAST_MODEL_PATH,
    )
    print(f"Saved last model: {LAST_MODEL_PATH}")

    epochs = [r["epoch"] for r in history]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(epochs, [r["train_loss"] for r in history], label="train")
    axes[0].plot(epochs, [r["val_loss"] for r in history], label="val")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[1].plot(epochs, [r["train_acc"] for r in history], label="train")
    axes[1].plot(epochs, [r["val_acc"] for r in history], label="val")
    axes[1].set_title("Accuracy")
    axes[1].legend()
    axes[2].plot(epochs, [r["train_f1"] for r in history], label="train")
    axes[2].plot(epochs, [r["val_f1"] for r in history], label="val")
    axes[2].set_title("Macro F1")
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(CURVES_PATH, dpi=140)
    plt.close(fig)
    print(f"Saved curves: {CURVES_PATH}")

    if BEST_MODEL_PATH.exists():
        ckpt = torch.load(BEST_MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])

    train_metrics = run_epoch(
        model, eval_train_loader, criterion, optimizer=None, collect_predictions=True
    )
    val_metrics = run_epoch(
        model, val_loader, criterion, optimizer=None, collect_predictions=True
    )
    test_metrics = run_epoch(
        model, test_loader, criterion, optimizer=None, collect_predictions=True
    )

    save_split_reports(
        train_metrics,
        TRAIN_CONFUSION_PATH,
        TRAIN_REPORT_PATH,
        TRAIN_REPORT_JSON_PATH,
        "train",
    )
    save_split_reports(
        val_metrics,
        VAL_CONFUSION_PATH,
        VAL_REPORT_PATH,
        VAL_REPORT_JSON_PATH,
        "best-val",
    )
    save_split_reports(
        test_metrics,
        TEST_CONFUSION_PATH,
        TEST_REPORT_PATH,
        TEST_REPORT_JSON_PATH,
        "test",
    )
    save_prediction_csv(dataset, train_idx, train_metrics, TRAIN_PREDICTIONS_PATH)
    save_prediction_csv(dataset, val_idx, val_metrics, VAL_PREDICTIONS_PATH)
    save_prediction_csv(dataset, test_idx, test_metrics, TEST_PREDICTIONS_PATH)
    print(
        f"Test metrics | loss={test_metrics['loss']:.4f} acc={test_metrics['acc']:.4f} f1={test_metrics['f1']:.4f}"
    )

    params["best_val_loss"] = best_val_loss
    params["best_val_f1"] = best_val_f1
    params["best_epoch"] = best_epoch
    params["train_loss"] = train_metrics["loss"]
    params["train_acc"] = train_metrics["acc"]
    params["train_f1"] = train_metrics["f1"]
    params["val_loss_at_best"] = val_metrics["loss"]
    params["val_acc_at_best"] = val_metrics["acc"]
    params["val_f1_at_best"] = val_metrics["f1"]
    params["test_loss"] = test_metrics["loss"]
    params["test_acc"] = test_metrics["acc"]
    params["test_f1"] = test_metrics["f1"]
    with open(PARAMS_PATH, "w") as f:
        json.dump(params, f, indent=2)
