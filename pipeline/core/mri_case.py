"""Structured MRI case folders.

A raw MRI study (a DICOM folder, or series already converted to NIfTI/NRRD) is
turned into one case folder with every axial / sagittal T1 and T2 series in a
predictable place::

    <cases_root>/<case_id>/
        case.json               manifest used by the rest of the pipeline
        series_labels.csv       one row per unique source series (editable)
        contact_sheet.png       mid-slice thumbnail of every series, for review
        sagittal/T1/<case>_sag_T1_<series>.nii.gz
        sagittal/T2/<case>_sag_T2_<series>.nii.gz
        axial/T1/<case>_ax_T1_<series>.nii.gz
        axial/T2/<case>_ax_T2_<series>.nii.gz
        seg/sagittal/...        written by stage0_segment_mri
        seg/axial/...
        steps/<step>/           written by python -m sins_steps (one folder per step)

Plane (axial / sagittal / coronal) is always derived from the image geometry.
Weighting (T1 / T2 / STIR / T1_POST / OTHER) comes from, in order:
    1. a value typed into series_labels.csv (always wins, survives re-runs)
    2. DICOM tags (SeriesDescription, ProtocolName, TE, TR, TI, contrast agent)
    3. a dcm2niix-style JSON sidecar or NIfTI "descrip" field (TE=..)
    4. tokens in the file name ("t2", "stir", ...)
If none of these exist the series is left as UNKNOWN and is not placed into a
T1/T2 folder until its weighting is filled in in series_labels.csv.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np


PLANES = ("sagittal", "axial")
WEIGHTINGS = ("T1", "T2")
ALL_WEIGHTINGS = ("T1", "T2", "STIR", "T1_POST", "OTHER", "UNKNOWN")
PLANE_SHORT = {"sagittal": "sag", "axial": "ax", "coronal": "cor"}
IMAGE_SUFFIXES = (".nii.gz", ".nii", ".nrrd", ".nhdr", ".mha", ".mhd")
LABELS_FILENAME = "series_labels.csv"
MANIFEST_FILENAME = "case.json"
LABEL_FIELDS = [
    "series_id",
    "weighting",
    "include",
    "plane",
    "auto_weighting",
    "weighting_source",
    "shape",
    "spacing_mm",
    "obliquity_deg",
    "duplicate_of",
    "exclude_reason",
    "source",
    "notes",
]
MIN_SLICES = 6            # fewer slices -> localizer / scout
ISOTROPIC_RATIO = 1.5     # slice/in-plane spacing below this -> 3D volume
MAX_OBLIQUITY_DEG = 45.0  # beyond this the plane is ambiguous


@dataclass
class SeriesRecord:
    series_id: str
    source: str
    kind: str                      # "nifti" | "itk" | "dicom"
    plane: str = ""
    obliquity_deg: float = 0.0
    shape: tuple = ()
    spacing_mm: tuple = ()
    auto_weighting: str = "UNKNOWN"
    weighting_source: str = ""
    weighting: str = "UNKNOWN"
    include: bool = True
    exclude_reason: str = ""
    duplicate_of: str = ""
    digest: str = ""
    notes: str = ""
    metadata: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------
def plane_from_affine(affine: np.ndarray) -> tuple[str, float]:
    """Return (plane, obliquity in degrees) from the slice normal direction.

    The slice axis is the voxel axis with the largest spacing (thick-slice 2D
    acquisitions) or, if spacings tie, the third axis.
    """
    affine = np.asarray(affine, dtype=np.float64)
    zooms = np.linalg.norm(affine[:3, :3], axis=0)
    slice_axis = int(np.argmax(zooms)) if zooms.max() > zooms.min() * 1.05 else 2
    normal = affine[:3, slice_axis] / zooms[slice_axis]
    world_axis = int(np.argmax(np.abs(normal)))
    plane = ("sagittal", "coronal", "axial")[world_axis]
    obliquity = float(np.degrees(np.arccos(np.clip(abs(normal[world_axis]), 0.0, 1.0))))
    return plane, obliquity


# --------------------------------------------------------------------------
# Weighting from metadata / text
# --------------------------------------------------------------------------
_POST_TOKENS = ("post", "+c", "c+", "gad", "gd", "contrast", "+gd", "_pc")
_TOKEN_SPLIT = re.compile(r"[^a-z0-9+]+")


def weighting_from_text(text: str) -> str:
    text = str(text or "").lower()
    if not text.strip():
        return "UNKNOWN"
    tokens = set(t for t in _TOKEN_SPLIT.split(text) if t)
    joined = " ".join(sorted(tokens))
    if any(t in tokens for t in ("loc", "localizer", "localiser", "scout", "survey", "3pl")):
        return "OTHER"
    if any(t in tokens for t in ("adc", "dwi", "diff", "diffusion", "trace", "mip", "swi")):
        return "OTHER"
    if "stir" in tokens or "tirm" in tokens or "stir" in text:
        return "STIR"
    is_post = any(tok in text for tok in _POST_TOKENS)
    if re.search(r"t2", joined) or "t2w" in text:
        return "T2"
    if re.search(r"t1", joined) or "t1w" in text:
        return "T1_POST" if is_post else "T1"
    return "UNKNOWN"


def weighting_from_timing(te_ms, tr_ms, ti_ms=None, contrast=False) -> str:
    try:
        te = float(te_ms)
        tr = float(tr_ms)
    except (TypeError, ValueError):
        return "UNKNOWN"
    if not np.isfinite(te) or not np.isfinite(tr) or te <= 0 or tr <= 0:
        return "UNKNOWN"
    try:
        ti = float(ti_ms) if ti_ms not in (None, "") else None
    except (TypeError, ValueError):
        ti = None
    if ti is not None and 100.0 <= ti <= 250.0 and te >= 30.0:
        return "STIR"
    if tr <= 1000.0 and te <= 30.0:
        return "T1_POST" if contrast else "T1"
    if te >= 60.0 and tr >= 1500.0:
        return "T2"
    return "UNKNOWN"


def _sidecar_metadata(path: Path) -> dict:
    name = path.name
    for suffix in IMAGE_SUFFIXES:
        if name.endswith(suffix):
            stem = name[: -len(suffix)]
            break
    else:
        stem = path.stem
    sidecar = path.with_name(stem + ".json")
    if sidecar.is_file():
        try:
            return json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
    return {}


def _descrip_metadata(img) -> dict:
    try:
        raw = img.header["descrip"]
        text = raw.item().decode("latin-1") if hasattr(raw, "item") else str(raw)
    except Exception:
        return {}
    out = {}
    for key, value in re.findall(r"(TE|TR|TI)=([0-9.]+)", text):
        out[{"TE": "EchoTime", "TR": "RepetitionTime", "TI": "InversionTime"}[key]] = float(value)
    return out


def classify_weighting(metadata: dict, filename: str) -> tuple[str, str]:
    """Return (weighting, source) using metadata first, file name second."""
    description = " ".join(
        str(metadata.get(key, ""))
        for key in ("SeriesDescription", "ProtocolName", "SequenceName", "ScanningSequence")
    )
    contrast = bool(str(metadata.get("ContrastBolusAgent", "")).strip())
    from_text = weighting_from_text(description)
    if from_text == "T1" and contrast:
        from_text = "T1_POST"
    if from_text != "UNKNOWN":
        return from_text, "series_description"

    te = metadata.get("EchoTime")
    tr = metadata.get("RepetitionTime")
    ti = metadata.get("InversionTime")
    # dcm2niix JSON stores seconds; DICOM stores milliseconds.
    if te is not None and tr is not None:
        te, tr = float(te), float(tr)
        if te < 1.0 and tr < 50.0:
            te, tr = te * 1000.0, tr * 1000.0
            ti = float(ti) * 1000.0 if ti not in (None, "") else None
        from_timing = weighting_from_timing(te, tr, ti, contrast)
        if from_timing != "UNKNOWN":
            return from_timing, "te_tr"

    from_name = weighting_from_text(Path(filename).name)
    if from_name != "UNKNOWN":
        return from_name, "filename"
    return "UNKNOWN", ""


# --------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------
def _is_image_file(path: Path) -> bool:
    name = path.name.lower()
    return any(name.endswith(s) for s in IMAGE_SUFFIXES) and not name.startswith(".")


def _looks_like_dicom(path: Path) -> bool:
    if path.suffix.lower() in (".dcm", ".ima"):
        return True
    if path.suffix:
        return False
    try:
        with path.open("rb") as stream:
            stream.seek(128)
            return stream.read(4) == b"DICM"
    except OSError:
        return False


def _series_id_from_name(name: str) -> str:
    stem = name
    for suffix in IMAGE_SUFFIXES:
        if stem.lower().endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    lead = re.match(r"^\s*(\d+)", stem)
    trail = re.search(r"_(\d+)\s*$", stem)
    number = lead.group(1) if lead else (trail.group(1) if trail else "")
    block = re.search(r"imageOrientationPatient\s*(\d+)", stem, flags=re.I)
    series_id = f"se{int(number):03d}" if number else re.sub(r"[^A-Za-z0-9]+", "", stem)[:16] or "series"
    if block:
        series_id += f"_b{block.group(1)}"
    return series_id


def discover_sources(source: Path) -> list[SeriesRecord]:
    """Find every image series below ``source`` (files or DICOM folders)."""
    source = Path(source)
    records: list[SeriesRecord] = []
    if source.is_file():
        files = [source]
    else:
        # skip hidden files, e.g. the "._*" metadata macOS writes on external (exFAT) drives
        files = sorted(p for p in source.rglob("*")
                       if p.is_file() and not any(part.startswith(".") for part in p.relative_to(source).parts))

    dicom_dirs = sorted({p.parent for p in files if _looks_like_dicom(p)})
    for path in files:
        if _is_image_file(path):
            kind = "nifti" if path.name.lower().endswith((".nii", ".nii.gz")) else "itk"
            records.append(SeriesRecord(series_id=_series_id_from_name(path.name), source=str(path), kind=kind))
    for directory in dicom_dirs:
        records.extend(_discover_dicom_series(directory))

    used: dict[str, int] = {}
    for record in records:
        base = record.series_id
        if base in used:
            used[base] += 1
            record.series_id = f"{base}_{used[base]}"
        else:
            used[base] = 1
    return records


def _discover_dicom_series(directory: Path) -> list[SeriesRecord]:
    import SimpleITK as sitk

    reader = sitk.ImageSeriesReader()
    out = []
    for uid in reader.GetGDCMSeriesIDs(str(directory)) or []:
        files = reader.GetGDCMSeriesFileNames(str(directory), uid)
        if not files:
            continue
        metadata = _read_dicom_metadata(files[0])
        number = metadata.get("SeriesNumber")
        series_id = f"se{int(number):03d}" if str(number).isdigit() else f"uid{uid[-6:]}"
        out.append(
            SeriesRecord(
                series_id=series_id,
                source=str(directory),
                kind="dicom",
                metadata={**metadata, "_series_uid": uid, "_files": list(files)},
            )
        )
    return out


def _read_dicom_metadata(path: str) -> dict:
    try:
        import pydicom
    except ImportError:
        return {}
    ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
    keys = (
        "SeriesDescription",
        "ProtocolName",
        "SequenceName",
        "ScanningSequence",
        "EchoTime",
        "RepetitionTime",
        "InversionTime",
        "ContrastBolusAgent",
        "SeriesNumber",
        "MRAcquisitionType",
    )
    out = {}
    for key in keys:
        value = getattr(ds, key, None)
        if value is None or value == "":
            continue
        out[key] = str(value) if not isinstance(value, (int, float)) else value
    return out


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_series_image(record: SeriesRecord):
    """Return a nibabel image for any supported record."""
    if record.kind == "nifti":
        return nib.load(record.source)
    import SimpleITK as sitk

    if record.kind == "dicom":
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(record.metadata["_files"])
        image = reader.Execute()
    else:
        image = sitk.ReadImage(record.source)
    if image.GetNumberOfComponentsPerPixel() != 1:
        raise ValueError("multi-component (RGB) image")
    return _sitk_to_nib(image)


def _sitk_to_nib(image):
    import SimpleITK as sitk

    if image.GetDimension() == 4:
        size = list(image.GetSize())
        if size[3] != 1:
            raise ValueError("4D series")
        image = image[:, :, :, 0]
    data = sitk.GetArrayFromImage(image).transpose(2, 1, 0)  # -> i, j, k
    direction = np.asarray(image.GetDirection(), dtype=np.float64).reshape(3, 3)
    spacing = np.asarray(image.GetSpacing(), dtype=np.float64)
    origin = np.asarray(image.GetOrigin(), dtype=np.float64)
    affine = np.eye(4)
    affine[:3, :3] = direction * spacing
    affine[:3, 3] = origin
    lps_to_ras = np.diag([-1.0, -1.0, 1.0, 1.0])
    return nib.Nifti1Image(data, lps_to_ras @ affine)


def _digest(img) -> str:
    data = np.ascontiguousarray(np.asarray(img.dataobj))
    h = hashlib.sha1()
    h.update(str(data.shape).encode())
    h.update(str(data.dtype).encode())
    h.update(np.round(np.asarray(img.affine), 3).tobytes())
    h.update(data.tobytes())
    return h.hexdigest()


# --------------------------------------------------------------------------
# Inventory
# --------------------------------------------------------------------------
def inspect_series(record: SeriesRecord) -> tuple[SeriesRecord, object]:
    try:
        img = load_series_image(record)
    except Exception as exc:  # unreadable, RGB, 4D ...
        record.include = False
        record.exclude_reason = f"unreadable: {exc}"
        return record, None
    if len(img.shape) != 3:
        if len(img.shape) == 4 and img.shape[3] == 1:
            img = nib.Nifti1Image(np.asarray(img.dataobj)[..., 0], img.affine, img.header)
        else:
            record.include = False
            record.exclude_reason = f"not a 3D volume (shape {tuple(img.shape)})"
            return record, None
    zooms = np.linalg.norm(np.asarray(img.affine)[:3, :3], axis=0)
    record.shape = tuple(int(v) for v in img.shape)
    record.spacing_mm = tuple(round(float(v), 3) for v in zooms)
    record.plane, record.obliquity_deg = plane_from_affine(img.affine)
    record.obliquity_deg = round(record.obliquity_deg, 1)
    record.digest = _digest(img)

    metadata = dict(record.metadata)
    if record.kind != "dicom":
        metadata.update(_sidecar_metadata(Path(record.source)))
        metadata.update({k: v for k, v in _descrip_metadata(img).items() if k not in metadata})
    record.auto_weighting, record.weighting_source = classify_weighting(metadata, Path(record.source).name)
    record.weighting = record.auto_weighting

    slice_axis = int(np.argmax(zooms))
    n_slices = img.shape[slice_axis]
    if n_slices < MIN_SLICES:
        record.include = False
        record.exclude_reason = f"localizer ({n_slices} slices)"
    elif zooms.max() / max(zooms.min(), 1e-6) < ISOTROPIC_RATIO:
        record.include = False
        record.exclude_reason = "3D / isotropic volume (not a 2D axial or sagittal stack)"
    elif record.plane not in PLANES:
        record.include = False
        record.exclude_reason = f"{record.plane} plane"
    elif record.obliquity_deg > MAX_OBLIQUITY_DEG:
        record.include = False
        record.exclude_reason = f"too oblique ({record.obliquity_deg} deg)"
    return record, img


def _read_existing_labels(case_dir: Path) -> dict[str, dict]:
    path = case_dir / LABELS_FILENAME
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return {row["series_id"]: row for row in csv.DictReader(stream) if row.get("series_id")}


def _normalize_weighting(value: str) -> str:
    value = str(value or "").strip().upper().replace(" ", "_").replace("-", "_")
    aliases = {"T1W": "T1", "T2W": "T2", "T1POST": "T1_POST", "T1_C": "T1_POST", "T1+C": "T1_POST", "": "UNKNOWN"}
    value = aliases.get(value, value)
    if value not in ALL_WEIGHTINGS:
        raise ValueError(f"Unknown weighting '{value}'. Use one of {ALL_WEIGHTINGS}.")
    return value


def _truthy(value) -> bool:
    return str(value).strip().lower() in ("1", "true", "yes", "y", "t")


def apply_user_labels(records: list[SeriesRecord], existing: dict[str, dict]) -> None:
    """Values typed into series_labels.csv override auto-detection."""
    for record in records:
        row = existing.get(record.series_id)
        if not row:
            continue
        if row.get("source") and Path(row["source"]).name != Path(record.source).name:
            continue  # different file now has this id; do not carry labels over
        typed = _normalize_weighting(row.get("weighting", ""))
        if typed != "UNKNOWN" and typed != record.auto_weighting:
            record.weighting = typed
            record.weighting_source = "series_labels.csv"
        elif typed != "UNKNOWN":
            record.weighting = typed
        if row.get("include", "").strip() != "" and not record.duplicate_of:
            wanted = _truthy(row["include"])
            if wanted != record.include:
                record.include = wanted
                record.exclude_reason = "" if wanted else (record.exclude_reason or "excluded in series_labels.csv")
        if row.get("notes"):
            record.notes = row["notes"]


def write_labels(case_dir: Path, records: list[SeriesRecord]) -> Path:
    path = case_dir / LABELS_FILENAME
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=LABEL_FIELDS)
        writer.writeheader()
        for r in records:
            writer.writerow(
                {
                    "series_id": r.series_id,
                    "weighting": r.weighting if r.weighting != "UNKNOWN" else "",
                    "include": "yes" if r.include else "no",
                    "plane": r.plane,
                    "auto_weighting": r.auto_weighting,
                    "weighting_source": r.weighting_source,
                    "shape": "x".join(str(v) for v in r.shape),
                    "spacing_mm": "x".join(f"{v:g}" for v in r.spacing_mm),
                    "obliquity_deg": r.obliquity_deg,
                    "duplicate_of": r.duplicate_of,
                    "exclude_reason": r.exclude_reason,
                    "source": r.source,
                    "notes": r.notes,
                }
            )
    return path


# --------------------------------------------------------------------------
# Contact sheet
# --------------------------------------------------------------------------
def _mid_slice(img) -> np.ndarray:
    canonical = nib.as_closest_canonical(img)
    data = np.asarray(canonical.dataobj, dtype=np.float32)
    zooms = np.linalg.norm(canonical.affine[:3, :3], axis=0)
    axis = int(np.argmax(zooms))
    index = data.shape[axis] // 2
    plane = np.take(data, index, axis=axis)
    in_plane = [z for i, z in enumerate(zooms) if i != axis]
    return plane.T, float(in_plane[1] / in_plane[0])


def write_contact_sheet(case_dir: Path, items: list[tuple[SeriesRecord, object]]) -> Path | None:
    items = [(r, img) for r, img in items if img is not None and not r.duplicate_of]
    if not items:
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = min(6, len(items))
    rows = int(np.ceil(len(items) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 3.6 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (record, img) in zip(axes.ravel(), items):
        plane_img, aspect = _mid_slice(img)
        finite = plane_img[np.isfinite(plane_img)]
        vmax = float(np.percentile(finite, 99.5)) if finite.size else 1.0
        ax.imshow(plane_img, cmap="gray", origin="lower", vmin=0, vmax=max(vmax, 1e-6), aspect=aspect)
        status = record.weighting if record.include else f"excluded: {record.exclude_reason[:28]}"
        ax.set_title(f"{record.series_id} | {record.plane}\n{status}", fontsize=8,
                     color=("black" if record.include else "gray"))
    fig.suptitle(f"{case_dir.name}: fill 'weighting' in {LABELS_FILENAME} (T1/T2/STIR/T1_POST/OTHER)", fontsize=10)
    fig.tight_layout()
    out = case_dir / "contact_sheet.png"
    fig.savefig(out, dpi=80)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------
def _organized_path(case_dir: Path, case_id: str, record: SeriesRecord) -> Path:
    short = PLANE_SHORT[record.plane]
    return case_dir / record.plane / record.weighting / f"{case_id}_{short}_{record.weighting}_{record.series_id}.nii.gz"


def build_case(source, case_id: str, cases_root, contact_sheet: bool = True, verbose: bool = True) -> dict:
    """Create / refresh ``<cases_root>/<case_id>`` from a raw MRI study."""
    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(f"Source not found: {source}")
    case_dir = Path(cases_root) / str(case_id)
    case_dir.mkdir(parents=True, exist_ok=True)

    records = discover_sources(source)
    if not records:
        raise RuntimeError(f"No image series (NIfTI / NRRD / DICOM) found under {source}")

    inspected = [inspect_series(record) for record in records]
    seen: dict[str, str] = {}
    for record, _img in inspected:
        if not record.digest:
            continue
        if record.digest in seen:
            record.duplicate_of = seen[record.digest]
            record.include = False
            record.exclude_reason = f"duplicate of {record.duplicate_of}"
        else:
            seen[record.digest] = record.series_id

    apply_user_labels(records, _read_existing_labels(case_dir))

    # Clear previously organized volumes so relabelled series move cleanly.
    previous = _load_manifest(case_dir)
    for plane_entries in (previous.get("series") or {}).values():
        for entries in plane_entries.values():
            for entry in entries:
                stale = case_dir / entry["path"]
                if stale.is_file():
                    stale.unlink()

    series: dict[str, dict[str, list]] = {plane: {w: [] for w in WEIGHTINGS} for plane in PLANES}
    pending = []
    for record, img in inspected:
        if not record.include or record.plane not in PLANES:
            continue
        if record.weighting not in WEIGHTINGS:
            if record.weighting == "UNKNOWN":
                pending.append(record.series_id)
            continue
        out_path = _organized_path(case_dir, str(case_id), record)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if record.kind == "nifti" and record.source.endswith(".nii.gz"):
            shutil.copy2(record.source, out_path)
        else:
            nib.save(img, str(out_path))
        series[record.plane][record.weighting].append(
            {
                "series_id": record.series_id,
                "path": out_path.relative_to(case_dir).as_posix(),
                "shape": list(record.shape),
                "spacing_mm": list(record.spacing_mm),
                "obliquity_deg": record.obliquity_deg,
                "weighting_source": record.weighting_source,
                "source": record.source,
            }
        )

    labels_path = write_labels(case_dir, records)
    sheet = write_contact_sheet(case_dir, inspected) if contact_sheet else None

    manifest = {
        "case_id": str(case_id),
        "source": str(source),
        "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "series": series,
        "unlabelled_series": pending,
        "segmentations": previous.get("segmentations", {}),
    }
    # Drop segmentations whose image is no longer part of the case.
    live = {e["path"] for p in series.values() for es in p.values() for e in es}
    manifest["segmentations"] = {k: v for k, v in manifest["segmentations"].items() if k in live}
    (case_dir / MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if verbose:
        print(f"Case folder: {case_dir}")
        for plane in PLANES:
            for weighting in WEIGHTINGS:
                ids = [e["series_id"] for e in series[plane][weighting]]
                print(f"  {plane:8s} {weighting}: {', '.join(ids) if ids else '-'}")
        dupes = sum(1 for r in records if r.duplicate_of)
        excluded = sum(1 for r in records if not r.include and not r.duplicate_of)
        print(f"  {len(records)} source series | {dupes} duplicates | {excluded} excluded (localizer/3D/coronal/other)")
        if pending:
            print(
                f"  [ACTION] {len(pending)} axial/sagittal series have no T1/T2 label in their metadata: "
                f"{', '.join(pending)}\n"
                f"           Fill the 'weighting' column in {labels_path} (see {sheet.name if sheet else 'contact sheet'})"
                f" and re-run this command."
            )
    return manifest


def _load_manifest(case_dir: Path) -> dict:
    path = Path(case_dir) / MANIFEST_FILENAME
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_manifest(case_dir: Path, manifest: dict) -> None:
    (Path(case_dir) / MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------
# Reading a built case
# --------------------------------------------------------------------------
class MRICase:
    """Read-only view of a case folder produced by :func:`build_case`."""

    def __init__(self, case_dir):
        self.case_dir = Path(case_dir)
        self.manifest = _load_manifest(self.case_dir)
        if not self.manifest:
            raise FileNotFoundError(f"No {MANIFEST_FILENAME} in {self.case_dir}; run build_mri_case first.")
        self.case_id = self.manifest["case_id"]

    def series(self, plane: str, weighting: str) -> list[dict]:
        return list(self.manifest.get("series", {}).get(plane, {}).get(weighting, []))

    def image_path(self, entry: dict) -> Path:
        return self.case_dir / entry["path"]

    def seg_path(self, entry: dict) -> Path | None:
        rel = self.manifest.get("segmentations", {}).get(entry["path"])
        path = self.case_dir / rel if rel is not None else self.default_seg_path(entry)
        return path if path.is_file() else None

    def default_seg_path(self, entry: dict) -> Path:
        image = Path(entry["path"])
        stem = image.name[: -len(".nii.gz")]
        plane = image.parts[0]
        return self.case_dir / "seg" / plane / f"{stem}_seg.nii.gz"

    def load_context(self, entry: dict, plane: str, weighting: str):
        """PatientContext (canonical orientation) for one series + its segmentation."""
        from nibabel.processing import resample_from_to

        from pipeline.core.context import PatientContext

        seg_path = self.seg_path(entry)
        if seg_path is None:
            raise FileNotFoundError(
                f"No segmentation for {entry['path']}; run stage0_segment_mri --case {self.case_dir}"
            )
        image_path = self.image_path(entry)
        img = nib.as_closest_canonical(nib.load(str(image_path)))
        seg_img = nib.as_closest_canonical(nib.load(str(seg_path)))
        if seg_img.shape != img.shape or not np.allclose(seg_img.affine, img.affine, atol=1e-4):
            seg_img = resample_from_to(seg_img, img, order=0)
        image = np.asarray(img.dataobj, dtype=np.float32)
        seg = np.rint(np.asarray(seg_img.dataobj)).astype(np.int16)
        context = PatientContext(
            patient_id=f"{self.case_id}:{entry['series_id']}",
            ct=image,
            seg=seg,
            affine=np.asarray(img.affine, dtype=np.float64),
            ct_path=image_path,
            seg_path=seg_path,
            modality="MR",
            sequence=f"{weighting} {plane}",
        )
        context.plane = plane
        context.series_id = entry["series_id"]
        return context
