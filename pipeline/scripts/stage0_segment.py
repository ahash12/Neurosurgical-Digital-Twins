import argparse

from pipeline.stages.stage0 import Stage0Config, Stage0Segmentation
from utils import DEFAULT_DATA_ROOT


def parse_args():
    parser = argparse.ArgumentParser(description="Run TotalSegmentator vertebra segmentation for stage 0.")
    parser.add_argument("--root-dir", type=str, default=DEFAULT_DATA_ROOT, help="Dataset root with patient CT folders.")
    parser.add_argument("--patient-id", type=str, default="", help="Optional single patient id.")
    parser.add_argument("--output-root", type=str, default="output/stage0", help="Output directory for TotalSegmentator masks.")
    parser.add_argument("--device", type=str, default="gpu", choices=["gpu", "cpu"], help="TotalSegmentator device.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip patients with non-empty stage0 output.")
    parser.add_argument("--ct-task", type=str, default="total", help="TotalSegmentator task for CT scans.")
    parser.add_argument("--mr-task", type=str, default="vertebrae_mr", help="TotalSegmentator task for MR scans.")
    parser.add_argument("--no-combine", action="store_true", help="Do not create the shared labeled segmentation file.")
    return parser.parse_args()


def main():
    args = parse_args()
    stage0 = Stage0Segmentation(
        Stage0Config(
            output_root=args.output_root,
            device=args.device,
            ct_task=args.ct_task,
            mr_task=args.mr_task,
            write_combined_segmentation=not args.no_combine,
        )
    )
    if args.patient_id:
        stage0.run_patient(args.patient_id, root_dir=args.root_dir, skip_existing=args.skip_existing)
    else:
        stage0.run_directory(root_dir=args.root_dir, skip_existing=args.skip_existing)


if __name__ == "__main__":
    main()
