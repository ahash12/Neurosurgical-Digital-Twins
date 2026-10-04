"""Build a structured MRI case folder (axial/sagittal x T1/T2) from a raw study.

Examples
--------
# One study (DICOM folder, or a folder of already-converted .nii.gz / .nrrd)
python -m pipeline.scripts.build_mri_case --source "../trial_b_deid'd/pt-00" --case-id pt-00

# Every sub-folder of a dataset root becomes one case
python -m pipeline.scripts.build_mri_case --source-root "../trial_b_deid'd"

After the first run, open <case>/contact_sheet.png, fill any empty 'weighting'
cells in <case>/series_labels.csv (T1 / T2 / STIR / T1_POST / OTHER) and run
the same command again. Typed labels are kept on every re-run.
"""

import argparse
from pathlib import Path

from pipeline.core.mri_case import build_case

DEFAULT_CASES_ROOT = "output/mri_cases"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--source", type=str, help="Raw study folder (DICOM or NIfTI/NRRD series).")
    group.add_argument("--source-root", type=str, help="Folder whose sub-folders are studies (one case each).")
    parser.add_argument("--case-id", type=str, default="", help="Case id (default: source folder name).")
    parser.add_argument("--cases-root", type=str, default=DEFAULT_CASES_ROOT, help="Where case folders are written.")
    parser.add_argument("--no-contact-sheet", action="store_true", help="Skip the review thumbnail PNG.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.source:
        source = Path(args.source)
        build_case(source, args.case_id or source.name, args.cases_root, contact_sheet=not args.no_contact_sheet)
        return
    for study in sorted(Path(args.source_root).iterdir()):
        if not study.is_dir() or study.name.startswith("."):
            continue
        try:
            build_case(study, study.name, args.cases_root, contact_sheet=not args.no_contact_sheet)
        except Exception as exc:  # keep going across a dataset
            print(f"[ERROR] {study.name}: {exc}")


if __name__ == "__main__":
    main()
