from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from monai.networks.nets import resnet10

from cams import (
    build_stage1_patch_and_mask,
    build_stage1_patch_mask_and_coords,
    infer_num_classes_from_checkpoint,
    load_stage1_model,
)
from dataset_class import build_stage1_patch_from_arrays
from pipeline.core.results import Stage1Explanation, Stage1Prediction
from pipeline.core.stage1_multimodal import (
    ARCHITECTURE,
    CLASS_NAMES,
    CTOrMRIResNet10,
    MultimodalPatchConfig,
    build_multimodal_patch,
)
from pipeline.explainability.cam import CAMExplainer
from pipeline.stages.base import PipelineStage, register_stage
from utils import VERTEBRAE


@dataclass
class Stage1PatchConfig:
    patch_size: tuple[int, int, int] = (96, 96, 64)
    norm_mode: str = "zscore_sigmoid"
    zscore_scale: float = 1.5
    foreground_floor: float = 0.15


class Stage1Classifier(PipelineStage):
    name = "stage1"
    num_classes = 4
    class_names = ["none", "blastic", "lytic", "mixed"]
    supported_modalities = {"CT"}

    def __init__(
        self,
        model_path,
        device=None,
        cam: CAMExplainer | None = None,
        patch_config=None,
        allow_unsupported_modality=False,
    ):
        self.model_path = model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cam = cam
        self.patch_config = patch_config or Stage1PatchConfig()
        self.allow_unsupported_modality = bool(allow_unsupported_modality)
        self.model = load_stage1_model(
            model_path,
            device=self.device,
            num_classes=self.num_classes,
        )

    @classmethod
    def from_checkpoint(cls, model_path, device=None, cam=None, patch_config=None):
        num_classes = infer_num_classes_from_checkpoint(model_path)
        if cls is Stage1Classifier:
            variant = stage1_variant_for_classes(num_classes)
            return variant(
                model_path, device=device, cam=cam, patch_config=patch_config
            )
        return cls(model_path, device=device, cam=cam, patch_config=patch_config)

    def build_patch(self, context, vertebra, with_coords=False):
        kwargs = {
            "patch_size": self.patch_config.patch_size,
            "norm_mode": self.patch_config.norm_mode,
            "zscore_scale": self.patch_config.zscore_scale,
            "foreground_floor": self.patch_config.foreground_floor,
        }
        if with_coords:
            return build_stage1_patch_mask_and_coords(
                context.ct, context.seg, vertebra, **kwargs
            )
        return build_stage1_patch_and_mask(context.ct, context.seg, vertebra, **kwargs)

    def predict_patch(self, vertebra, patch):
        x = (
            patch.unsqueeze(0).to(self.device)
            if patch.ndim == 4
            else patch.to(self.device)
        )
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)[0].detach().cpu()
        class_id = int(probs.argmax().item())
        class_name = (
            self.class_names[class_id]
            if class_id < len(self.class_names)
            else str(class_id)
        )
        return Stage1Prediction(
            vertebra=str(vertebra).upper(),
            class_id=class_id,
            class_name=class_name,
            probabilities=[float(v) for v in probs.tolist()],
        )

    def predict_vertebra(self, context, vertebra):
        if getattr(
            context, "modality", "CT"
        ) not in self.supported_modalities and not getattr(
            self, "allow_unsupported_modality", False
        ):
            raise ValueError(
                "The current Stage 1 model is CT-specific; "
                f"received {context.modality}."
            )
        patch, _ = self.build_patch(context, vertebra, with_coords=False)
        return self.predict_patch(vertebra, patch)

    def explain_vertebra(self, context, vertebra, target_class=None):
        if self.cam is None:
            raise RuntimeError("CAM is not enabled for stage1.")
        patch, mask, coords = self.build_patch(context, vertebra, with_coords=True)
        prediction = self.predict_patch(vertebra, patch)
        cam_result = self.cam.compute(self.model, patch, target_class=target_class)
        return Stage1Explanation(
            prediction=prediction,
            cam=cam_result.cam,
            mask=mask,
            coords=coords,
        )

    def run(self, context):
        predictions = []
        for vertebra in VERTEBRAE:
            if not (context.seg == self._label_for_vertebra(vertebra)).any():
                continue
            predictions.append(self.predict_vertebra(context, vertebra))
        return predictions

    @staticmethod
    def _label_for_vertebra(vertebra):
        from utils import VERTEBRA_LABELS

        return VERTEBRA_LABELS[str(vertebra).upper()]


@register_stage("stage1_4class")
class Stage1FourClass(Stage1Classifier):
    num_classes = 4
    class_names = ["none", "blastic", "lytic", "mixed"]


@register_stage("stage1_binary")
class Stage1Binary(Stage1Classifier):
    num_classes = 2
    class_names = ["none", "cancer"]


@register_stage("stage1_cancer_type")
class Stage1CancerType(Stage1Classifier):
    num_classes = 3
    class_names = ["blastic", "lytic", "mixed"]


class Stage1MRI(Stage1Classifier):
    supported_modalities = {"MR"}

    def __init__(self, model_path, device=None):
        model_path = Path(model_path)
        metadata_path = model_path.parent / "param.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("modality") != "MR":
            raise ValueError("MRI Stage 1 requires a checkpoint trained on MR data.")
        if metadata.get("class_names") != self.class_names:
            raise ValueError("MRI checkpoint must use none/blastic/lytic/mixed labels.")
        if metadata.get("norm_mode") != "robust_minmax":
            raise ValueError("MRI checkpoint must declare robust_minmax normalization.")
        super().__init__(
            model_path,
            device=device,
            patch_config=Stage1PatchConfig(
                patch_size=tuple(metadata["patch_size"]),
                norm_mode="robust_minmax",
                foreground_floor=float(metadata["foreground_floor"]),
            ),
        )
        self.sequence_families = set(metadata["sequence_families"])

    def predict_vertebra(self, context, vertebra):
        from pipeline.scripts.stage3_postlateral import mri_sequence_family

        family = mri_sequence_family(context.sequence)
        if family not in self.sequence_families:
            raise ValueError(f"MRI checkpoint does not support sequence family: {family}")
        return super().predict_vertebra(context, vertebra)


@register_stage("stage1_ct_or_mri_binary")
class Stage1CTOrMRIBinary(PipelineStage):
    supported_modalities = {"CT", "MR"}
    class_names = CLASS_NAMES

    def __init__(self, model_path: str | Path, device: str | None = None) -> None:
        model_path = Path(model_path)
        metadata = json.loads(
            (model_path.parent / "param.json").read_text(encoding="utf-8")
        )
        if metadata.get("architecture") != ARCHITECTURE:
            raise ValueError(
                "Checkpoint is not the shared CT-or-MRI binary architecture."
            )
        if (
            metadata.get("modalities") != ["CT", "MR"]
            or metadata.get("class_names") != CLASS_NAMES
        ):
            raise ValueError(
                "Checkpoint must declare CT/MR training and none/cancer labels."
            )
        if metadata.get("normalization") != "modality_groupnorm":
            raise ValueError("Checkpoint must declare modality-specific GroupNorm.")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.patch_config = MultimodalPatchConfig(**metadata["patch_config"])
        self.sequence_families = set(metadata["sequence_families"])
        self.supported_vertebrae = set(metadata["supported_vertebrae"])
        if not self.sequence_families:
            raise ValueError(
                "Checkpoint needs MRI sequence families seen during training."
            )
        self.threshold = float(metadata["decision_threshold"])
        if not 0 < self.threshold < 1:
            raise ValueError("Binary decision threshold must be between zero and one.")
        self.model = CTOrMRIResNet10(pretrained=False).to(self.device)
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device, weights_only=True)
        )
        self.model.eval()

    def predict_vertebra(self, context, vertebra: str) -> Stage1Prediction:
        from pipeline.scripts.stage3_postlateral import mri_sequence_family

        if context.modality not in self.supported_modalities:
            raise ValueError(f"Unsupported modality: {context.modality}")
        vertebra = vertebra.upper()
        if vertebra not in self.supported_vertebrae:
            raise ValueError(f"Checkpoint does not support vertebra: {vertebra}")
        if context.modality == "MR":
            family = mri_sequence_family(context.sequence)
            if family not in self.sequence_families:
                raise ValueError(
                    f"MRI sequence was not represented in training: {family}"
                )
        scales = np.linalg.norm(context.affine[:3, :3], axis=0)
        spacing = (float(scales[0]), float(scales[1]), float(scales[2]))
        patch = build_multimodal_patch(
            context.image,
            context.seg,
            spacing,
            vertebra,
            context.modality,
            self.patch_config,
        )
        domain = torch.tensor(
            [0 if context.modality == "CT" else 1], device=self.device
        )
        with torch.no_grad():
            probabilities = (
                self.model(patch[None].to(self.device), domain)
                .softmax(dim=1)[0]
                .cpu()
                .tolist()
            )
        class_id = int(probabilities[1] >= self.threshold)
        return Stage1Prediction(
            vertebra, class_id, CLASS_NAMES[class_id], probabilities
        )

    def run(self, context) -> list[Stage1Prediction]:
        from utils import VERTEBRA_LABELS

        return [
            self.predict_vertebra(context, vertebra)
            for vertebra in VERTEBRAE
            if vertebra in self.supported_vertebrae
            and np.any(context.seg == VERTEBRA_LABELS[vertebra])
        ]


class MedicalNetResNet10Binary(nn.Module):
    def __init__(self, input_channels=3):
        super().__init__()
        self.backbone = resnet10(
            pretrained=False,
            progress=False,
            spatial_dims=3,
            n_input_channels=input_channels,
            feed_forward=False,
            shortcut_type="B",
            bias_downsample=False,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(512, 2),
        )

    def forward(self, inputs):
        return self.classifier(self.backbone(inputs))


class Stage1MedicalNetBinaryEnsemble(PipelineStage):
    name = "stage1_medicalnet_binary_ensemble"
    supported_modalities = {"CT"}

    def __init__(
        self,
        run_dir,
        device=None,
        patch_size=(96, 96, 64),
        physical_fov_mm=(128.0, 128.0, 96.0),
    ):
        self.run_dir = Path(run_dir)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.patch_size = tuple(int(value) for value in patch_size)
        self.physical_fov_mm = tuple(float(value) for value in physical_fov_mm)
        self.models = []
        self.thresholds = []

        fold_dirs = sorted(self.run_dir.glob("fold_*"))
        if not fold_dirs:
            raise FileNotFoundError(f"No fold directories found in: {self.run_dir}")
        for fold_dir in fold_dirs:
            checkpoint_path = fold_dir / "best.pth"
            result_path = fold_dir / "result.json"
            if not checkpoint_path.exists() or not result_path.exists():
                raise FileNotFoundError(
                    f"Incomplete Stage 1a fold output: {fold_dir}"
                )
            model = MedicalNetResNet10Binary().to(self.device)
            checkpoint = torch.load(
                checkpoint_path,
                map_location=self.device,
                weights_only=True,
            )
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            self.models.append(model)
            result = json.loads(result_path.read_text(encoding="utf-8"))
            self.thresholds.append(float(result["selected_threshold"]))
        self.threshold = float(np.mean(self.thresholds))

    def predict_vertebra(self, context, vertebra):
        if getattr(context, "modality", "CT") not in self.supported_modalities:
            raise ValueError(
                "The MedicalNet Stage 1a ensemble is CT-specific; "
                f"received {context.modality}."
            )
        spacing = tuple(
            float(value)
            for value in np.linalg.norm(context.affine[:3, :3], axis=0)
        )
        patch = build_stage1_patch_from_arrays(
            context.ct,
            context.seg,
            str(vertebra).upper(),
            patch_size=self.patch_size,
            spacing=spacing,
            patch_mode="physical_context",
            physical_fov_mm=self.physical_fov_mm,
        )
        inputs = torch.from_numpy(patch).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probabilities = [
                torch.softmax(model(inputs), dim=1)[0]
                for model in self.models
            ]
        mean_probabilities = torch.stack(probabilities).mean(dim=0).cpu()
        cancer_probability = float(mean_probabilities[1])
        class_id = int(cancer_probability >= self.threshold)
        return Stage1Prediction(
            vertebra=str(vertebra).upper(),
            class_id=class_id,
            class_name=("cancer" if class_id else "none"),
            probabilities=[float(value) for value in mean_probabilities],
        )


class Stage1Cascade(PipelineStage):
    name = "stage1_cascade"
    class_names = ["none", "blastic", "lytic", "mixed"]

    def __init__(self, detector, cancer_type_classifier):
        self.detector = detector
        self.cancer_type_classifier = cancer_type_classifier

    def predict_vertebra(self, context, vertebra):
        detection = self.detector.predict_vertebra(context, vertebra)
        cancer_probability = float(detection.probabilities[1])
        if not detection.is_suspicious:
            return Stage1Prediction(
                vertebra=str(vertebra).upper(),
                class_id=0,
                class_name="none",
                probabilities=[1.0 - cancer_probability, 0.0, 0.0, 0.0],
            )

        lesion_type = self.cancer_type_classifier.predict_vertebra(
            context,
            vertebra,
        )
        type_probabilities = np.asarray(
            lesion_type.probabilities,
            dtype=np.float64,
        )
        joint_probabilities = np.concatenate(
            ([1.0 - cancer_probability], cancer_probability * type_probabilities)
        )
        class_id = int(lesion_type.class_id) + 1
        return Stage1Prediction(
            vertebra=str(vertebra).upper(),
            class_id=class_id,
            class_name=self.class_names[class_id],
            probabilities=[float(value) for value in joint_probabilities],
        )


def stage1_variant_for_classes(num_classes):
    if int(num_classes) == 2:
        return Stage1Binary
    if int(num_classes) == 3:
        return Stage1CancerType
    return Stage1FourClass
