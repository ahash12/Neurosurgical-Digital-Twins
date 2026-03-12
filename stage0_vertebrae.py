import os
import subprocess

CT_ROOT = "nii_ct"
OUTPUT_ROOT = "seg_output"

# make sure output root exists
os.makedirs(OUTPUT_ROOT, exist_ok=True)


def run_totalsegmentator(input_path, output_path):
    cmd = [
        "TotalSegmentator",
        "-i", input_path,
        "-o", output_path,
        "--fast",
        "--roi_subset", "vertebrae"
    ]

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    for file in os.listdir(CT_ROOT):

        if not file.endswith("_ct.nii.gz"):
            continue

        patient_id = file.replace("_ct.nii.gz", "")
        input_path = os.path.join(CT_ROOT, file)
        output_path = os.path.join(OUTPUT_ROOT, patient_id)

        # Skip if already processed
        if os.path.exists(output_path) and len(os.listdir(output_path)) > 0:
            print(f"Skipping {patient_id}, already processed.")
            continue

        os.makedirs(output_path, exist_ok=True)

        try:
            run_totalsegmentator(input_path, output_path)
            print(f"Finished {patient_id}")
        except subprocess.CalledProcessError:
            print(f"Error processing {patient_id}")


if __name__ == "__main__":
    main()
