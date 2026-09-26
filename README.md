# Neuro

This repo is an staged automatic Spinal Instability Neoplastic Scale (SINS) scoring pipeline built around using CT scan.

## Repository Layout

```bash
pipeline/      # Pipeline code and runnable stage 
|-- scripts
validate/      # Per-stage validation commands
output/        # Model checkpoints and generated
|-- outputs

utils.py       # Shared labels and NIfTI helpers
dataset_class.py
cams.py
requirements.txt
enviroment.yml
```

## Environment

Create the environment:

```bash
# Windows PowerShell, Linux, or WSL
mamba env create -f enviroment.yml
conda activate radiel
```

PyTorch can be sensitive to CUDA/CPU setup, so if `torch` causes install or OpenMP issues, reinstall it using the command that matches your machine from the PyTorch install page.

## Data Layout

The default dataset root is defined in `utils.py`:

```python
DEFAULT_DATA_ROOT = "../../datasets/radiel/Spine-Mets-CT-SEG-Nifti"
```

Expected patient layout:

```text
../../datasets/radiel/Spine-Mets-CT-SEG-Nifti/
  14151/
    14151_ct.nii.gz
    14151_seg-1.nii.gz
```

Stage 1 metadata / labels are expected at:

```text
../../datasets/radiel/vertebra_dataset.csv
```

## Stage Scripts

Stage workflow scripts live under `pipeline/scripts/`.

Run Stage 0 TotalSegmentator vertebra segmentation:

```bash
# Run TotalSegmentator for one patient
python -m pipeline.scripts.stage0_segment --patient-id 14151
```

Train/evaluate the stage 1 model variants:

```bash
# Four-class stage 1 model: none / blastic / lytic / mixed
python -m pipeline.scripts.stage1_cancer

# Binary stage 1: shared CT-or-MRI model (requires reviewed MRI GT)
python -m pipeline.scripts.stage1a_binary

# CT-only model benchmark with patient-grouped cross-validation
python -m pipeline.scripts.stage1a_binary --experiment ct_cv

# Original CT DenseNet holdout experiment
python -m pipeline.scripts.stage1a_binary --experiment ct_holdout

# Cancer-type stage 1 model: blastic / lytic / mixed
python -m pipeline.scripts.stage1b_type
```

Run other stages scoring scripts:

```bash
# Stage 2 vertebral collapse scoring
python -m pipeline.scripts.stage2_collapse 

# Stage 3 posterolateral involvement scoring
python -m pipeline.scripts.stage3_postlateral 

# Stage 4 alignment scoring
python -m pipeline.scripts.stage4_alignment 
```

Run SINS inference:

```bash
# Run SINS
python -m pipeline.scripts.sins 14151
```

Run SINS with CAM overlay:

```bash
# Run SINS and show full-CT CAM overlay
python -m pipeline.scripts.sins 14151 --show-cam --cam-method gradcam++
```

CAM methods:

- gradcam
- gradcam++
- layercam


## Validation

Stage 0 validation:

```bash
# Compare TotalSegmentator output against ground truth
python -m validate.stage0_validate --patient-id 14151 --show --show-ct
```

Stage 1 validation with CAM:

```bash
# Predict one vertebra and show the Stage 1 CAM overlay
python -m validate.stage1_validate --patient-id 14151 --vertebra T8
```

Stage 2 collapse geometry validation:

```bash
# Show vertebral height/collapse geometry
python -m validate.stage2_validate --patient-id 14151 --vertebra T8
```

Stage 3 posterolateral geometry validation:

```bash
# Show posterolateral involvement geometry
python -m validate.stage3_validate --patient-id 14151 --vertebra T8
```

Stage 4 alignment validation:

```bash
# Show whole-spine alignment geometry
python -m validate.stage4_validate --patient-id 14151
```
