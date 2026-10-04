"""Moved to sins_steps/step0_segmentation. Same as:
    python -m sins_steps.step0_segmentation --case output/mri_cases/pt-00 --device mps
"""

import runpy
import sys

if __name__ == "__main__":
    sys.argv = [a.replace("--mr-task", "--task") for a in sys.argv]
    runpy.run_module("sins_steps.step0_segmentation", run_name="__main__")
