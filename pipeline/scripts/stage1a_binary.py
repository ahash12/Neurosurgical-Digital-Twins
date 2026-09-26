import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


EXPERIMENT = "ct_mri"
EXPERIMENTS = ("ct_holdout", "ct_cv", "ct_mri")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 1a cancer/normal training: CT holdout, CT benchmark/CV, or one CT-or-MRI model."
    )
    parser.add_argument("--experiment", choices=EXPERIMENTS, default=EXPERIMENT)
    parser.add_argument(
        "--check-data",
        action="store_true",
        help="CT-or-MRI only: validate GT, files and patient splits without training.",
    )
    parser.add_argument(
        "--write-label-template",
        action="store_true",
        help="CT-or-MRI only: create blank MRI GT candidates from Stage 0 masks.",
    )
    args = parser.parse_args()
    if args.experiment != "ct_mri" and (args.check_data or args.write_label_template):
        parser.error(
            "--check-data and --write-label-template require --experiment ct_mri."
        )
    if args.check_data and args.write_label_template:
        parser.error("Select either --check-data or --write-label-template.")
    if args.experiment == "ct_holdout":
        from pipeline.training.stage1a_holdout import main as train

        train()
    elif args.experiment == "ct_cv":
        from pipeline.training.stage1a_ct_cv import main as train

        train()
    else:
        from pipeline.training.stage1a_multimodal import main as train_multimodal

        train_multimodal(
            check_data=args.check_data, write_template=args.write_label_template
        )


if __name__ == "__main__":
    main()
