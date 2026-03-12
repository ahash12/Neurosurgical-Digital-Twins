import argparse
import csv
import itertools
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset


@dataclass
class LoaderConfig:
    num_workers: int
    pin_memory: bool
    persistent_workers: bool
    prefetch_factor: int


class FakeVolumeDataset(Dataset):
    def __init__(self, size: int, shape: tuple[int, ...], num_classes: int, cpu_work: int = 0):
        self.size = size
        self.shape = shape
        self.num_classes = num_classes
        self.cpu_work = cpu_work

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int):
        value = (idx % 255) / 255.0
        x = torch.full(self.shape, value, dtype=torch.float32)
        # Optional lightweight synthetic preprocessing to better emulate dataset transforms.
        for _ in range(self.cpu_work):
            x = x * 1.0001 + 0.0001
        y = idx % self.num_classes
        return x, y


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark DataLoader combinations on a fake dataset.")
    parser.add_argument("--dataset-size", type=int, default=4096, help="Number of examples in fake dataset.")
    parser.add_argument(
        "--shape",
        nargs="+",
        type=int,
        default=[1, 96, 96, 96],
        help="Tensor shape per sample. Example: --shape 1 96 96 96",
    )
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4, help="Single batch size (ignored if --batch-size-grid is set).")
    parser.add_argument(
        "--batch-size-grid",
        nargs="+",
        type=int,
        default=[4, 8, 16, 32, 64],
        help="Batch sizes to test in one run.",
    )
    parser.add_argument("--warmup-batches", type=int, default=10)
    parser.add_argument("--measure-batches", type=int, default=120)
    parser.add_argument(
        "--num-workers-grid",
        nargs="+",
        type=int,
        default=[0, 2, 4, 8],
        help="Grid of num_workers values to test.",
    )
    parser.add_argument(
        "--prefetch-grid",
        nargs="+",
        type=int,
        default=[2, 4],
        help="Grid of prefetch_factor values (used when num_workers > 0).",
    )
    parser.add_argument("--cpu-work", type=int, default=0, help="Synthetic transform loop count in __getitem__.")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
        help="Device to move each batch to during benchmarking.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-csv", type=str, default="", help="Optional path to save CSV results.")
    return parser.parse_args()


def build_grid(num_workers_grid: list[int], prefetch_grid: list[int]) -> list[LoaderConfig]:
    configs: list[LoaderConfig] = []
    for num_workers in num_workers_grid:
        pin_values = [False, True]
        persistent_values = [False, True] if num_workers > 0 else [False]
        prefetch_values = prefetch_grid if num_workers > 0 else [2]

        for pin_memory, persistent_workers, prefetch_factor in itertools.product(
            pin_values, persistent_values, prefetch_values
        ):
            configs.append(
                LoaderConfig(
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    persistent_workers=persistent_workers,
                    prefetch_factor=prefetch_factor,
                )
            )
    return configs


def run_one_config(
    dataset: Dataset,
    cfg: LoaderConfig,
    batch_size: int,
    warmup_batches: int,
    measure_batches: int,
    device: str,
) -> dict:
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": cfg.num_workers,
        "pin_memory": cfg.pin_memory,
    }
    if cfg.num_workers > 0:
        loader_kwargs["persistent_workers"] = cfg.persistent_workers
        loader_kwargs["prefetch_factor"] = cfg.prefetch_factor

    loader = DataLoader(**loader_kwargs)
    total_samples = 0
    measured_batches = 0
    non_blocking = bool(cfg.pin_memory and device == "cuda")

    iterator = iter(loader)

    for _ in range(warmup_batches):
        try:
            x, y = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            x, y = next(iterator)
        if device == "cuda":
            x = x.to("cuda", non_blocking=non_blocking)
            y = y.to("cuda", non_blocking=non_blocking)
            torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(measure_batches):
        try:
            x, y = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            x, y = next(iterator)

        if device == "cuda":
            x = x.to("cuda", non_blocking=non_blocking)
            y = y.to("cuda", non_blocking=non_blocking)
            torch.cuda.synchronize()

        total_samples += x.size(0)
        measured_batches += 1

    elapsed = max(time.perf_counter() - start, 1e-12)
    batches_per_sec = measured_batches / elapsed
    samples_per_sec = total_samples / elapsed
    ms_per_batch = (elapsed / measured_batches) * 1000.0

    return {
        "batch_size": batch_size,
        "num_workers": cfg.num_workers,
        "pin_memory": cfg.pin_memory,
        "persistent_workers": cfg.persistent_workers,
        "prefetch_factor": cfg.prefetch_factor,
        "batches_measured": measured_batches,
        "samples_measured": total_samples,
        "seconds": elapsed,
        "batches_per_sec": batches_per_sec,
        "samples_per_sec": samples_per_sec,
        "ms_per_batch": ms_per_batch,
    }


def print_results(results: list[dict]):
    print(
        f"{'rank':<5} {'batch':<6} {'workers':<8} {'pin':<5} {'persist':<8} {'prefetch':<9} "
        f"{'ms/batch':<10} {'samples/s':<11} {'batches/s':<10}"
    )
    print("-" * 84)
    for i, row in enumerate(results, start=1):
        print(
            f"{i:<5} {row['batch_size']:<6} {row['num_workers']:<8} {str(row['pin_memory']):<5} "
            f"{str(row['persistent_workers']):<8} {row['prefetch_factor']:<9} "
            f"{row['ms_per_batch']:<10.3f} {row['samples_per_sec']:<11.2f} {row['batches_per_sec']:<10.2f}"
        )


def maybe_write_csv(results: list[dict], out_csv: str):
    if not out_csv:
        return
    path = Path(out_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "batch_size",
        "num_workers",
        "pin_memory",
        "persistent_workers",
        "prefetch_factor",
        "batches_measured",
        "samples_measured",
        "seconds",
        "ms_per_batch",
        "batches_per_sec",
        "samples_per_sec",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved CSV: {path}")


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    dataset = FakeVolumeDataset(
        size=args.dataset_size,
        shape=tuple(args.shape),
        num_classes=args.num_classes,
        cpu_work=args.cpu_work,
    )

    grid = build_grid(args.num_workers_grid, args.prefetch_grid)
    print(f"Dataset size: {args.dataset_size}")
    print(f"Sample shape: {tuple(args.shape)}")
    print(f"Batch sizes: {args.batch_size_grid}")
    print(f"Device: {args.device}")
    print(f"CPU work loops: {args.cpu_work}")
    print(f"Configs to test: {len(grid)}\n")

    results = []
    for batch_size in args.batch_size_grid:
        for cfg in grid:
            row = run_one_config(
                dataset=dataset,
                cfg=cfg,
                batch_size=batch_size,
                warmup_batches=args.warmup_batches,
                measure_batches=args.measure_batches,
                device=args.device,
            )
            results.append(row)

    results.sort(key=lambda r: r["samples_per_sec"], reverse=True)
    print_results(results)
    maybe_write_csv(results, args.out_csv)

    best = results[0]
    print(
        f"\nBest config -> batch_size={best['batch_size']}, workers={best['num_workers']}, pin_memory={best['pin_memory']}, "
        f"persistent_workers={best['persistent_workers']}, prefetch_factor={best['prefetch_factor']}, "
        f"samples/s={best['samples_per_sec']:.2f}"
    )


if __name__ == "__main__":
    main()
