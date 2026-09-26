import argparse

from pipeline.core.context import PatientContext
from pipeline.core.results import SINSResult
from pipeline.explainability.cam import CAMExplainer
from pipeline.stages.stage1 import Stage1FourClass
from pipeline.stages.stage2 import Stage2Collapse
from pipeline.stages.stage3 import Stage3Posterolateral
from pipeline.stages.stage4 import Stage4Alignment
from pipeline.visualization.overlays import VolumeOverlayViewer
from pipeline.scripts.stage3_postlateral import prepare_ct_seg_arrays
from utils import DEFAULT_DATA_ROOT, VERTEBRAE


DEFAULT_STAGE1_MODEL = "output/stage1/4_class/2026-03-20/last.pth"


def location_score(vertebra):
    if vertebra in ["T1", "T2", "T11", "T12", "L1", "L5"]:
        return 3
    if vertebra in ["L2", "L3", "L4"]:
        return 2
    if vertebra in ["T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10"]:
        return 1
    return 0


def bone_score(class_id):
    if int(class_id) == 2:
        return 1
    if int(class_id) == 3:
        return 2
    return 0


def sins_category(total):
    if total >= 10:
        return "Unstable"
    if total >= 4:
        return "Potentially Unstable"
    return "Stable"


class SINSPipeline:
    def __init__(self, stage1, stage2=None, stage3=None, stage4=None, viewer=None):
        self.stage1 = stage1
        self.stage2 = stage2 or Stage2Collapse()
        self.stage3 = stage3 or Stage3Posterolateral()
        self.stage4 = stage4 or Stage4Alignment()
        self.viewer = viewer or VolumeOverlayViewer()

    def run_patient(self, context, show_cam=False):
        context.ct, context.seg = prepare_ct_seg_arrays(context.ct, context.seg)
        align_score = self.stage4.run_patient(context)
        thresholds = self.stage3.prepare(context)

        predictions = []
        explanations = []
        results = []

        for vertebra in VERTEBRAE:
            if not self._has_vertebra(context, vertebra):
                continue

            prediction = self.stage1.predict_vertebra(context, vertebra)
            predictions.append(prediction)

            if show_cam:
                explanations.append(self.stage1.explain_vertebra(context, vertebra))

            if not prediction.is_suspicious:
                continue

            loc = location_score(vertebra)
            bone = bone_score(prediction.class_id)
            collapse = self.stage2.run_vertebra(context, vertebra)
            posterolateral = self.stage3.run_vertebra(context, vertebra, thresholds=thresholds)
            total = loc + bone + collapse + posterolateral + align_score
            results.append(
                SINSResult(
                    vertebra=vertebra,
                    total=int(total),
                    category=sins_category(total),
                    location=int(loc),
                    bone=int(bone),
                    collapse=int(collapse),
                    posterolateral=int(posterolateral),
                    alignment=int(align_score),
                )
            )

        results.sort(key=lambda item: item.total, reverse=True)
        if show_cam:
            self.print_stage1_predictions(predictions)
        self.print_results(results)
        if show_cam:
            self.show_cam_overlay(context, explanations)
        return results

    def print_stage1_predictions(self, predictions):
        if not predictions:
            return
        parts = [f"{p.vertebra}: {p.class_name} {p.confidence:.2f}" for p in predictions]
        print("Stage1 predictions")
        print(" | ".join(parts[:8]))
        if len(parts) > 8:
            print(" | ".join(parts[8:]))

    def print_results(self, results):
        if not results:
            print("No suspicious vertebrae found.")
            return
        print("Top 3")
        for result in results[:3]:
            print(
                f"{result.vertebra}: SINS={result.total} "
                f"({result.category}) | "
                f"loc={result.location} bone={result.bone} collapse={result.collapse} "
                f"posterolateral={result.posterolateral} align={result.alignment}"
            )

    def show_cam_overlay(self, context, explanations):
        if not explanations:
            print("[WARN] No vertebrae found for full-volume CAM overlay.")
            return

        cam_points = [self.stage1.cam.patient_points(item) for item in explanations]
        summary = [
            f"{item.prediction.vertebra}: {item.prediction.class_name} {item.prediction.confidence:.2f}"
            for item in explanations
        ]
        print(f"Showing stage1 CAMs over the original CT volume for {len(cam_points)} vertebrae.")
        self.viewer.show_stage1_patient_cams(
            context,
            cam_points,
            title=(
                f"{context.patient_id} | original CT + stage1 {self.stage1.cam.method} CAMs | no CAM threshold\n"
                + " | ".join(summary[:8])
                + ("\n" + " | ".join(summary[8:16]) if len(summary) > 8 else "")
            ),
        )

    @staticmethod
    def _has_vertebra(context, vertebra):
        from utils import VERTEBRA_LABELS

        return (context.seg == VERTEBRA_LABELS[str(vertebra).upper()]).any()


def build_pipeline(model_path=None, cam_method="gradcam", experimental_mri=False):
    cam = CAMExplainer(method=cam_method)
    stage1 = Stage1FourClass(
        model_path or DEFAULT_STAGE1_MODEL,
        cam=cam,
        allow_unsupported_modality=experimental_mri,
    )
    return SINSPipeline(stage1=stage1)


def parse_args():
    parser = argparse.ArgumentParser(description="Run the SINS pipeline for one patient.")
    parser.add_argument("patient_id", nargs="?", default="", help="Patient id. If omitted, prompts interactively.")
    parser.add_argument("--root-dir", type=str, default=DEFAULT_DATA_ROOT, help="Dataset root directory.")
    parser.add_argument("--model-path", type=str, default="", help="Stage1 checkpoint path.")
    parser.add_argument("--show-cam", action="store_true", help="Show every vertebra stage1 CAM over the original CT volume.")
    parser.add_argument("--cam-method", type=str, default="gradcam", choices=["gradcam", "gradcam++", "layercam"], help="CAM method.")
    parser.add_argument(
        "--experimental-mri-ct-models",
        action="store_true",
        help="Allow the CT-trained Stage 1 model on MRI for exploratory output only. Stage 3 remains CT-only.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    patient_id = args.patient_id or input("Patient ID: ")
    context = PatientContext.load(patient_id, root_dir=args.root_dir, canonical=True)
    pipeline = build_pipeline(
        model_path=args.model_path or None,
        cam_method=args.cam_method,
        experimental_mri=args.experimental_mri_ct_models,
    )
    pipeline.run_patient(context, show_cam=args.show_cam)


if __name__ == "__main__":
    main()
