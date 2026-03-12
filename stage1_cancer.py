import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from pathlib import Path
from collections import Counter
import csv
import json
from datetime import datetime
import matplotlib.pyplot as plt

from dataset_class import VertebraDataset
from monai.networks.nets import DenseNet121
from utils import DEFAULT_DATA_ROOT

# Lesion is Cancer
CSV_PATH = "data/vertebra_dataset.csv"
ROOT_DIR = DEFAULT_DATA_ROOT

BATCH_SIZE = 16
EPOCHS = 20
LR = 1e-4
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
SEED = 42
NUM_WORKERS = 0
PIN_MEMORY = False
USE_PATCH_CACHE = True
PREBUILD_PATCH_CACHE = True
PATCH_CACHE_DIR = "data/vertebra_patch_cache"
PATIENT_CACHE_SIZE = 1
PATCH_SIZE = (96, 96, 96)
NORM_MODE = "zscore_sigmoid"
ZSCORE_SCALE = 1.5
FOREGROUND_FLOOR = 0.15
RUN_DATE = datetime.now().strftime("%Y-%m-%d")
OUTPUT_DIR = Path("output/stage1") / RUN_DATE
BEST_MODEL_PATH = OUTPUT_DIR / "best.pth"
LAST_MODEL_PATH = OUTPUT_DIR / "last.pth"
HISTORY_PATH = OUTPUT_DIR / "history.csv"
PARAMS_PATH = OUTPUT_DIR / "param.json"
CURVES_PATH = OUTPUT_DIR / "curves.png"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def build_patient_splits(dataset, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO, seed=SEED):
    patients = dataset.df["patient_id"].astype(str).unique().tolist()
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(patients), generator=g).tolist()
    patients = [patients[i] for i in perm]

    n_total = len(patients)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val

    train_patients = set(patients[:n_train])
    val_patients = set(patients[n_train:n_train + n_val])
    test_patients = set(patients[n_train + n_val:])

    train_idx, val_idx, test_idx = [], [], []
    for idx, pid in enumerate(dataset.df["patient_id"].astype(str).tolist()):
        if pid in train_patients:
            train_idx.append(idx)
        elif pid in val_patients:
            val_idx.append(idx)
        elif pid in test_patients:
            test_idx.append(idx)

    return train_idx, val_idx, test_idx


def make_loader(subset, shuffle=False):
    return DataLoader(
        subset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )


def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    conf = torch.zeros((4, 4), dtype=torch.long)

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

        total_loss += loss.item() * x.size(0)
        preds = out.argmax(dim=1)
        total_correct += (preds == y).sum().item()
        total_examples += x.size(0)
        for t, p in zip(y.detach().cpu(), preds.detach().cpu()):
            conf[int(t), int(p)] += 1

    avg_loss = total_loss / max(total_examples, 1)
    avg_acc = total_correct / max(total_examples, 1)
    f1_scores = []
    eps = 1e-8
    for c in range(conf.size(0)):
        tp = conf[c, c].item()
        fp = conf[:, c].sum().item() - tp
        fn = conf[c, :].sum().item() - tp
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = (2 * precision * recall) / (precision + recall + eps)
        f1_scores.append(f1)
    avg_f1 = sum(f1_scores) / len(f1_scores)
    return avg_loss, avg_acc, avg_f1


def label_counts(df, indices):
    labels = df.iloc[indices]["label"].tolist()
    return dict(Counter(labels))


def compute_class_weights(df, indices, num_classes=4):
    labels = df.iloc[indices]["label"].tolist()
    counts = torch.zeros(num_classes, dtype=torch.float32)
    for label in labels:
        if 0 <= int(label) < num_classes:
            counts[int(label)] += 1.0

    total = counts.sum().item()
    weights = torch.zeros(num_classes, dtype=torch.float32)
    for c in range(num_classes):
        if counts[c] > 0:
            weights[c] = total / (num_classes * counts[c])
        else:
            weights[c] = 0.0
    return weights, counts


dataset = VertebraDataset(
    csv_path=CSV_PATH,
    root_dir=ROOT_DIR,
    use_patch_cache=USE_PATCH_CACHE,
    cache_dir=PATCH_CACHE_DIR,
    patient_cache_size=PATIENT_CACHE_SIZE,
    patch_size=PATCH_SIZE,
    norm_mode=NORM_MODE,
    zscore_scale=ZSCORE_SCALE,
    foreground_floor=FOREGROUND_FLOOR,
)
if PREBUILD_PATCH_CACHE and USE_PATCH_CACHE:
    start_time = datetime.now()
    print(f"Precomputing patch cache -> {PATCH_CACHE_DIR}")
    dataset.precompute_cache(force=False, verbose=True)
    end_time = datetime.now()
    print(f"Precomputation time: {end_time - start_time}")

train_idx, val_idx, test_idx = build_patient_splits(dataset)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

train_set = Subset(dataset, train_idx)
val_set = Subset(dataset, val_idx)
test_set = Subset(dataset, test_idx)

train_loader = make_loader(train_set, shuffle=True)
val_loader = make_loader(val_set, shuffle=False)
test_loader = make_loader(test_set, shuffle=False)

print(f"Device: {DEVICE}")
print(f"Count: train - {len(train_set)} val - {len(val_set)} test - {len(test_set)}")
print(f"Train label counts: {label_counts(dataset.df, train_idx)}")
print(f"Val label counts:   {label_counts(dataset.df, val_idx)}")
print(f"Test label counts:  {label_counts(dataset.df, test_idx)}")
print(f"Output dir: {OUTPUT_DIR}")

class_weights, train_counts = compute_class_weights(dataset.df, train_idx, num_classes=4)
print(f"Class weights (train): {class_weights.tolist()}")

params = {
    "csv_path": CSV_PATH,
    "root_dir": ROOT_DIR,
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "lr": LR,
    "train_ratio": TRAIN_RATIO,
    "val_ratio": VAL_RATIO,
    "seed": SEED,
    "device": DEVICE,
    "num_workers": NUM_WORKERS,
    "pin_memory": PIN_MEMORY,
    "use_patch_cache": USE_PATCH_CACHE,
    "prebuild_patch_cache": PREBUILD_PATCH_CACHE,
    "patch_cache_dir": PATCH_CACHE_DIR,
    "patient_cache_size": PATIENT_CACHE_SIZE,
    "patch_size": list(PATCH_SIZE),
    "norm_mode": NORM_MODE,
    "zscore_scale": ZSCORE_SCALE,
    "foreground_floor": FOREGROUND_FLOOR,
    "num_classes": 4,
    "train_samples": len(train_set),
    "val_samples": len(val_set),
    "test_samples": len(test_set),
    "train_label_counts": label_counts(dataset.df, train_idx),
    "val_label_counts": label_counts(dataset.df, val_idx),
    "test_label_counts": label_counts(dataset.df, test_idx),
    "class_weights": [float(x) for x in class_weights.tolist()],
    "class_count_vector": [int(x) for x in train_counts.tolist()],
}
with open(PARAMS_PATH, "w") as f:
    json.dump(params, f, indent=2)

model = DenseNet121(spatial_dims=3, in_channels=1, out_channels=4).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

history = []
best_val_loss = float("inf")

try:
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc, train_f1 = run_epoch(model, train_loader, criterion, optimizer=optimizer)
        val_loss, val_acc, val_f1 = run_epoch(model, val_loader, criterion, optimizer=None)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_f1": train_f1,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_f1": val_f1,
        }
        history.append(row)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} train_f1={train_f1:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            try:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_val_loss": best_val_loss,
                    },
                    BEST_MODEL_PATH,
                )
                print(f"Saved new best model at epoch {epoch} -> {str(BEST_MODEL_PATH)}")
            except Exception as save_err:
                print(f"[WARN] Could not save best model checkpoint: {save_err}")

except Exception as train_err:
    print(f"[ERROR] Training interrupted: {train_err}")
    try:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            LAST_MODEL_PATH,
        )
        print(f"Saved emergency last checkpoint: {str(LAST_MODEL_PATH)}")
    except Exception as emergency_save_err:
        print(f"[ERROR] Could not save emergency checkpoint: {emergency_save_err}")
    raise

# Save training history for tracking.
with open(HISTORY_PATH, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["epoch", "train_loss", "train_acc", "train_f1", "val_loss", "val_acc", "val_f1"],
    )
    writer.writeheader()
    writer.writerows(history)
print(f"Saved history: {str(HISTORY_PATH)}")

try:
    torch.save(
        {
            "epoch": EPOCHS,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
        },
        LAST_MODEL_PATH,
    )
    print(f"Saved last model: {str(LAST_MODEL_PATH)}")
except Exception as last_save_err:
    print(f"[WARN] Could not save last checkpoint: {last_save_err}")

try:
    epochs = [r["epoch"] for r in history]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(epochs, [r["train_loss"] for r in history], label="train")
    axes[0].plot(epochs, [r["val_loss"] for r in history], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, [r["train_acc"] for r in history], label="train")
    axes[1].plot(epochs, [r["val_acc"] for r in history], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    axes[2].plot(epochs, [r["train_f1"] for r in history], label="train")
    axes[2].plot(epochs, [r["val_f1"] for r in history], label="val")
    axes[2].set_title("Macro F1")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(CURVES_PATH, dpi=140)
    plt.close(fig)
    print(f"Saved curves: {str(CURVES_PATH)}")
except Exception as plot_err:
    print(f"[WARN] Could not save curves plot: {plot_err}")

# Evaluate test split using best checkpoint if available.
if BEST_MODEL_PATH.exists():
    ckpt = torch.load(BEST_MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])

test_loss, test_acc, test_f1 = run_epoch(model, test_loader, criterion, optimizer=None)
print(f"Test metrics | loss={test_loss:.4f} acc={test_acc:.4f} f1={test_f1:.4f}")

params["best_val_loss"] = best_val_loss
params["test_loss"] = test_loss
params["test_acc"] = test_acc
params["test_f1"] = test_f1
with open(PARAMS_PATH, "w") as f:
    json.dump(params, f, indent=2)
