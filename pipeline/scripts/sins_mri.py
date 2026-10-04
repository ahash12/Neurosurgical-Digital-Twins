"""Moved: the MRI pipeline now lives in sins_steps/ (one folder per step).

This wrapper keeps the old command working:
    python -m pipeline.scripts.sins_mri --case output/mri_cases/pt-00 [--model-path ...] [--cam-method ...]
is the same as
    python -m sins_steps --case output/mri_cases/pt-00 --steps 1,2,3,4,5
"""

import argparse
import sys

from sins_steps.__main__ import main as run_steps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--case", required=True)
    p.add_argument("--model-path", default="")
    p.add_argument("--cam-method", default="gradcam")
    p.add_argument("--show-cam", action="store_true")
    p.add_argument("--save-cam", action="store_true")  # always on now
    args, _ = p.parse_known_args()
    argv = ["sins_steps", "--case", args.case, "--steps", "1,2,3,4,5", "--cam-method", args.cam_method]
    if args.model_path:
        argv += ["--model-path", args.model_path]
    if args.show_cam:
        argv.append("--show-cam")
    print("[note] pipeline.scripts.sins_mri now runs sins_steps (see sins_steps/README.md)")
    sys.argv = argv
    run_steps()


if __name__ == "__main__":
    main()
