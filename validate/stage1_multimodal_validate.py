import csv
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import nibabel as nib
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from pipeline.core.context import PatientContext
from pipeline.core.stage1_multimodal import (
    ARCHITECTURE,
    CLASS_NAMES,
    CTOrMRIResNet10,
    ModalityGroupNorm,
    MultimodalPatchConfig,
    MultimodalVertebraDataset,
    build_multimodal_patch,
    load_ct_samples,
    load_mri_samples,
    validate_samples,
)
from pipeline.scripts.stage1a_multimodal import (
    evaluate_metrics,
    run_epoch,
    sampling_weights,
    split_samples,
)
from pipeline.stages.stage1 import Stage1CTOrMRIBinary


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_checks() -> None:
    torch.set_num_threads(2)
    torch.manual_seed(42)
    config = MultimodalPatchConfig(
        patch_size=(32, 32, 32), physical_fov_mm=(32.0, 32.0, 32.0)
    )
    with TemporaryDirectory(
        prefix="stage1_multimodal_", dir=Path(__file__).resolve().parents[1]
    ) as temporary:
        root = Path(temporary)
        seg = np.zeros((32, 32, 32), dtype=np.int16)
        seg[8:16, 8:16, 8:16] = 1
        seg[17:24, 17:24, 17:24] = 2
        ramp = np.indices(seg.shape).sum(axis=0).astype(np.float32)
        ct, mr = ramp * 20 - 300, ramp * 5 + 100
        affine = np.diag([1.0, 2.0, 3.0, 1.0])
        for scan, image, modality in (("CT-1", ct, "ct"), ("MR-1", mr, "mr")):
            folder = root / scan
            folder.mkdir()
            nib.save(
                nib.Nifti1Image(image, affine), folder / f"{scan}_{modality}.nii.gz"
            )
            nib.save(nib.Nifti1Image(seg, affine), folder / f"{scan}_seg-1.nii.gz")
        ct_csv, mr_csv = root / "ct.csv", root / "mr.csv"
        write_rows(
            ct_csv,
            [
                {"patient_id": "CT-1", "vertebra": "T1", "label": 0},
                {"patient_id": "CT-1", "vertebra": "T2", "label": 3},
            ],
        )
        mri_rows = [
            {
                "patient_id": "MRI-PATIENT",
                "scan_id": "MR-1",
                "vertebra": vertebra,
                "label": label,
                "sequence": "T1",
                "spine_coverage_reviewed": "True",
            }
            for vertebra, label in (("T1", 0), ("T2", 1))
        ]
        write_rows(mr_csv, mri_rows)
        samples = load_ct_samples(
            ct_csv, root, "synthetic_ct", set()
        ) + load_mri_samples(mr_csv, root)
        validate_samples(samples)
        assert [s.label for s in samples] == [0, 1, 0, 1]
        for invalid in ("", "2"):
            write_rows(mr_csv, [{**row, "label": invalid} for row in mri_rows])
            try:
                load_mri_samples(mr_csv, root)
            except ValueError:
                pass
            else:
                raise AssertionError("Missing or subtype MRI labels must be rejected")
        write_rows(mr_csv, mri_rows)
        dataset = MultimodalVertebraDataset(samples, config, root / "cache")
        dataset.precompute_cache()
        cache_path = dataset.cache_path(samples[0])
        before = cache_path.stat().st_mtime_ns
        dataset.precompute_cache()
        assert cache_path.stat().st_mtime_ns == before
        assert dataset.cache_path(samples[0]) != dataset.cache_path(samples[2])
        assert dataset[0][0].shape == (2, 32, 32, 32)
        assert torch.isfinite(dataset[0][0]).all()
        for sample, image in ((samples[0], ct), (samples[2], mr)):
            direct = build_multimodal_patch(
                image, seg, (1.0, 2.0, 3.0), "T1", sample.modality, config
            )
            cached = dataset[0 if sample.modality == "CT" else 2][0]
            assert torch.allclose(direct, cached)
        grouped = []
        for patient in range(20):
            for sample in samples:
                grouped.append(
                    replace(
                        sample,
                        patient_id=f"P{patient}",
                        group_id=f"P{patient}",
                        scan_id=f"{sample.modality}-P{patient}",
                    )
                )
        splits = split_samples(grouped)
        group_sets = [
            set(grouped[i].group_id for i in indices) for indices in splits.values()
        ]
        assert all(
            not group_sets[i] & group_sets[j] for i in range(3) for j in range(i)
        )
        weights = sampling_weights(grouped)
        assert torch.isclose(
            weights[torch.tensor([s.modality == "CT" for s in grouped])].sum(),
            weights[torch.tensor([s.modality == "MR" for s in grouped])].sum(),
        )
        model = CTOrMRIResNet10(pretrained=False)
        model.eval()
        inputs, _, domains, _ = next(iter(DataLoader(dataset, batch_size=4)))
        with torch.no_grad():
            mixed = model(inputs, domains)
            separate = torch.cat(
                [model(inputs[i : i + 1], domains[i : i + 1]) for i in range(4)]
            )
        assert torch.allclose(mixed, separate, atol=1e-5)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        loss, rows = run_epoch(
            model,
            DataLoader(dataset, batch_size=4),
            dataset,
            nn.CrossEntropyLoss(),
            torch.device("cpu"),
            optimizer,
        )
        assert np.isfinite(loss)
        for layer in model.modules():
            if isinstance(layer, ModalityGroupNorm):
                assert all(norm.weight.grad is not None for norm in layer.layers)
        assert set(evaluate_metrics(rows)) == {"all", "CT", "MR"}
        metadata = {
            "architecture": ARCHITECTURE,
            "modalities": ["CT", "MR"],
            "class_names": CLASS_NAMES,
            "normalization": "modality_groupnorm",
            "patch_config": asdict(config),
            "sequence_families": ["t1"],
            "supported_vertebrae": ["T1", "T2"],
            "decision_threshold": 0.5,
        }
        (root / "param.json").write_text(json.dumps(metadata), encoding="utf-8")
        checkpoint = root / "best.pth"
        torch.save(model.state_dict(), checkpoint)
        predictor = Stage1CTOrMRIBinary(checkpoint, device="cpu")
        for sample in (samples[0], samples[2]):
            context = PatientContext.load(sample.scan_id, root_dir=root)
            context.sequence = sample.sequence
            prediction = predictor.predict_vertebra(context, "T1")
            assert (
                prediction.class_name in CLASS_NAMES
                and len(prediction.probabilities) == 2
            )
        context.sequence = "T2"
        try:
            predictor.predict_vertebra(context, "T1")
        except ValueError:
            pass
        else:
            raise AssertionError("Unsupported MRI sequence must be rejected")
    print(
        "PASS: GT guards, shared physical crops, cache reuse, grouped splits, modality balancing, CT-only/MRI-only forward and backward, checkpoint reload and sequence guards."
    )


if __name__ == "__main__":
    run_checks()
