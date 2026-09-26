import csv
import re
from dataclasses import dataclass
from pathlib import Path

import SimpleITK as sitk


TRIAL_A_METADATA_NAME = "Trial A Metadata.xlsx - Scan Inventory.csv"
ORIGINAL_OID_PATTERN = re.compile(r"(?im)^original oid:=\s*(\S+)\s*$")
TRIAL_A_CT_STORAGE_OFFSET = 32768


@dataclass(frozen=True)
class TrialAScan:
    scan_id: str
    case_id: str
    modality: str
    sequence: str
    source_path: Path
    source_relative_path: str
    rows: int
    columns: int
    slices: int
    spacing_x: float
    spacing_y: float
    spacing_z: float


def read_nrrd_header(path: Path) -> str:
    header = bytearray()
    with path.open("rb") as source:
        previous = -1
        while True:
            value = source.read(1)
            if not value:
                break
            byte = value[0]
            header.append(byte)
            if byte == 10 and previous == 10:
                break
            if byte != 13:
                previous = byte
    return header.decode("ascii", errors="replace")


def nrrd_original_oid(path: Path) -> str:
    match = ORIGINAL_OID_PATTERN.search(read_nrrd_header(path))
    if match is None:
        raise ValueError(f"NRRD header has no original OID: {path}")
    return match.group(1)


def source_object_oid(value: str) -> str:
    name = Path(value.replace("\\", "/")).name
    return name.removesuffix(".vol")


def scan_modality(scan_type: str) -> str:
    modality = scan_type.split("|", maxsplit=1)[0].strip().upper()
    if modality not in {"CT", "MR"}:
        raise ValueError(f"Unsupported modality in scan type: {scan_type}")
    return modality


def scan_sequence(scan_type: str) -> str:
    return scan_type.rsplit("|", maxsplit=1)[-1].strip()


def load_trial_a_scans(dataset_root: Path) -> list[TrialAScan]:
    metadata_path = dataset_root / TRIAL_A_METADATA_NAME
    with metadata_path.open(newline="", encoding="utf-8-sig") as source:
        rows = list(csv.DictReader(source))

    nrrd_by_oid: dict[str, Path] = {}
    for path in sorted(dataset_root.rglob("*.nrrd")):
        oid = nrrd_original_oid(path)
        if oid in nrrd_by_oid:
            raise ValueError(f"Duplicate NRRD original OID {oid}: {path}")
        nrrd_by_oid[oid] = path

    case_counts: dict[str, int] = {}
    scans: list[TrialAScan] = []
    for row in rows:
        oid = source_object_oid(row["Original source object"])
        source_path = nrrd_by_oid.get(oid)
        if source_path is None:
            if row["File type"] == "DICOM image stack":
                continue
            raise FileNotFoundError(f"No local NRRD matches metadata OID {oid}")

        case_id = row["Our case ID"].strip()
        case_counts[case_id] = case_counts.get(case_id, 0) + 1
        scan_id = f"{case_id}-S{case_counts[case_id]:02d}"
        scans.append(
            TrialAScan(
                scan_id=scan_id,
                case_id=case_id,
                modality=scan_modality(row["Scan type"]),
                sequence=scan_sequence(row["Scan type"]),
                source_path=source_path,
                source_relative_path=source_path.relative_to(dataset_root).as_posix(),
                rows=int(row["Rows"]),
                columns=int(row["Columns"]),
                slices=int(row["Slices"]),
                spacing_x=float(row["Voxel spacing X (mm)"]),
                spacing_y=float(row["Voxel spacing Y (mm)"]),
                spacing_z=float(row["Voxel spacing Z (mm)"]),
            )
        )
    return scans


def prepare_trial_a_image(image: sitk.Image, modality: str) -> tuple[sitk.Image, str]:
    if modality != "CT":
        return image, "none"
    if image.GetPixelID() != sitk.sitkUInt16:
        raise ValueError(
            "Trial A CT decoding expects uint16 storage, "
            f"received {image.GetPixelIDTypeAsString()}"
        )
    decoder = sitk.ShiftScaleImageFilter()
    decoder.SetShift(-TRIAL_A_CT_STORAGE_OFFSET)
    decoder.SetScale(1.0)
    decoder.SetOutputPixelType(sitk.sitkInt16)
    return decoder.Execute(image), f"stored_value-{TRIAL_A_CT_STORAGE_OFFSET}"


def convert_trial_a(dataset_root: Path, output_root: Path) -> Path:
    scans = load_trial_a_scans(dataset_root)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.csv"
    fieldnames = [
        "scan_id",
        "case_id",
        "modality",
        "sequence",
        "image_path",
        "source_relative_path",
        "rows",
        "columns",
        "slices",
        "spacing_x_mm",
        "spacing_y_mm",
        "spacing_z_mm",
        "intensity_transform",
    ]

    manifest_rows = []
    for scan in scans:
        scan_dir = output_root / scan.scan_id
        scan_dir.mkdir(parents=True, exist_ok=True)
        suffix = "ct" if scan.modality == "CT" else "mr"
        image_path = scan_dir / f"{scan.scan_id}_{suffix}.nii.gz"
        image = sitk.ReadImage(str(scan.source_path))
        expected_size = (scan.columns, scan.rows, scan.slices)
        if image.GetSize() != expected_size:
            raise ValueError(
                f"Metadata size mismatch for {scan.scan_id}: "
                f"image={image.GetSize()} metadata={expected_size}"
            )
        image, intensity_transform = prepare_trial_a_image(image, scan.modality)
        sitk.WriteImage(image, str(image_path), useCompression=True)
        manifest_rows.append(
            {
                "scan_id": scan.scan_id,
                "case_id": scan.case_id,
                "modality": scan.modality,
                "sequence": scan.sequence,
                "image_path": image_path.relative_to(output_root).as_posix(),
                "source_relative_path": scan.source_relative_path,
                "rows": scan.rows,
                "columns": scan.columns,
                "slices": scan.slices,
                "spacing_x_mm": scan.spacing_x,
                "spacing_y_mm": scan.spacing_y,
                "spacing_z_mm": scan.spacing_z,
                "intensity_transform": intensity_transform,
            }
        )

    with manifest_path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)
    return manifest_path
