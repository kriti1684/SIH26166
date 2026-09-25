"""FastAPI routes for registration jobs, progress, and generated artifacts."""
import hashlib
import mimetypes
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Literal

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy import desc
from sqlalchemy.orm import Session

from backend.config import MAX_JOB_UPLOAD_BYTES, OUTPUTS_DIR, PREVIEWS_DIR, UPLOADS_DIR
from backend.database import get_db
from backend.models import JobStatus, RegistrationJob
from backend.schemas import (
    ArtifactInfo,
    ArtifactListResponse,
    JobDetailResponse,
    JobListResponse,
    JobStatusResponse,
    SensorRef,
    SensorSrc,
)
from backend.worker import run_registration_pipeline


router = APIRouter()
SUPPORTED_INPUT_SUFFIXES = {".xml", ".lbl", ".img", ".tif", ".tiff", ".h5", ".hdf5"}
SUPPORTED_SIDECAR_SUFFIXES = SUPPORTED_INPUT_SUFFIXES | {".dat", ".bin"}


def _safe_upload_name(upload: UploadFile, *, sidecar: bool = False) -> str:
    original = (upload.filename or "").replace("\\", "/").rsplit("/", 1)[-1].strip()
    name = re.sub(r'[<>:"|?*\x00-\x1f]', "_", original)
    if not name or name in {".", ".."}:
        raise HTTPException(status_code=400, detail="Every upload must have a valid filename")
    supported = SUPPORTED_SIDECAR_SUFFIXES if sidecar else SUPPORTED_INPUT_SUFFIXES
    if Path(name).suffix.lower() not in supported:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported input format for {name}. Supported formats: PDS labels/images, GeoTIFF, and HDF5.",
        )
    return name


async def _save_upload(upload: UploadFile, destination: Path, remaining_bytes: int) -> int:
    copied = 0
    try:
        with destination.open("wb") as target:
            while True:
                chunk = await upload.read(1024 * 1024)
                if not chunk:
                    break
                copied += len(chunk)
                if copied > remaining_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Combined upload exceeds the {MAX_JOB_UPLOAD_BYTES // (1024**3)} GiB job upload limit.",
                    )
                target.write(chunk)
    finally:
        await upload.close()
    return copied


def _job_or_404(db: Session, job_id: str) -> RegistrationJob:
    job = db.query(RegistrationJob).filter(RegistrationJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.post("/jobs", response_model=JobDetailResponse, status_code=201)
async def create_registration_job(
    source_file: UploadFile = File(...),
    reference_file: UploadFile = File(...),
    source_sidecars: Optional[List[UploadFile]] = File(default=None),
    reference_sidecars: Optional[List[UploadFile]] = File(default=None),
    sensor_src: SensorSrc = Form(...),
    sensor_ref: SensorRef = Form(...),
    method: Literal["loftr", "ensemble", "crater"] = Form("loftr"),
    structural_method: Literal["phase_congruency", "gradient"] = Form("phase_congruency"),
    coarse_method: Literal["auto", "fft", "crater"] = Form("auto"),
    poly_degree: int = Form(2, ge=0, le=4),
    tps_smoothing: float = Form(0.05, ge=0.0, le=10.0),
    warp_order: int = Form(3, ge=0, le=5),
    ransac_threshold: float = Form(1.2, gt=0.0, le=100.0),
    wac_band: int = Form(7, ge=1, le=7),
    grid_rows: int = Form(4, ge=1, le=16),
    grid_cols: int = Form(4, ge=1, le=16),
    export_native: bool = Form(False),
    db: Session = Depends(get_db),
):
    """Upload two products (plus optional sidecars) and enqueue the full pipeline."""
    source_sidecars = source_sidecars or []
    reference_sidecars = reference_sidecars or []
    if len(source_sidecars) + len(reference_sidecars) > 16:
        raise HTTPException(status_code=400, detail="A job can include at most 16 sidecar files")

    source_uploads = [source_file, *source_sidecars]
    reference_uploads = [reference_file, *reference_sidecars]
    names: Dict[str, List[str]] = {"source": [], "reference": []}
    for side, uploads in (("source", source_uploads), ("reference", reference_uploads)):
        for index, upload in enumerate(uploads):
            name = _safe_upload_name(upload, sidecar=index > 0)
            if name.casefold() in {existing.casefold() for existing in names[side]}:
                raise HTTPException(status_code=400, detail=f"Duplicate {side} product filename: {name}")
            names[side].append(name)

    config: Dict[str, Any] = {
        "sensor_src": sensor_src.value,
        "sensor_ref": sensor_ref.value,
        "method": method,
        "structural_method": structural_method,
        "coarse_method": coarse_method,
        "poly_degree": poly_degree,
        "tps_smoothing": tps_smoothing,
        "warp_order": warp_order,
        "ransac_threshold": ransac_threshold,
        "wac_band": wac_band,
        "grid_rows": grid_rows,
        "grid_cols": grid_cols,
        "export_native": export_native,
    }

    # Use generated job directories and sanitized basenames; a client filename
    # must never be able to select a path outside its own upload directory.
    import uuid

    job_id = str(uuid.uuid4())
    upload_root = UPLOADS_DIR / job_id
    source_dir, reference_dir = upload_root / "source", upload_root / "reference"
    source_dir.mkdir(parents=True, exist_ok=True)
    reference_dir.mkdir(parents=True, exist_ok=True)
    saved: Dict[str, List[Path]] = {"source": [], "reference": []}
    total_bytes = 0
    try:
        for side, uploads, destination_dir in (
            ("source", source_uploads, source_dir),
            ("reference", reference_uploads, reference_dir),
        ):
            for upload, name in zip(uploads, names[side]):
                remaining = MAX_JOB_UPLOAD_BYTES - total_bytes
                if remaining <= 0:
                    raise HTTPException(status_code=413, detail="Combined upload exceeds the job upload limit")
                destination = destination_dir / name
                total_bytes += await _save_upload(upload, destination, remaining)
                saved[side].append(destination)
    except Exception:
        import shutil

        shutil.rmtree(upload_root, ignore_errors=True)
        raise

    job = RegistrationJob(
        id=job_id,
        sensor_src=sensor_src.value,
        sensor_ref=sensor_ref.value,
        source_filename=names["source"][0],
        reference_filename=names["reference"][0],
        source_path=str(saved["source"][0]),
        reference_path=str(saved["reference"][0]),
        method=method,
        structural_method=structural_method,
        coarse_method=coarse_method,
        poly_degree=poly_degree,
        tps_smoothing=tps_smoothing,
        warp_order=warp_order,
        wac_band=wac_band,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        run_config_json={
            **config,
            "source_files": names["source"],
            "reference_files": names["reference"],
        },
        stage_events_json=[],
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    params = {
        **config,
        "source_path": str(saved["source"][0]),
        "reference_path": str(saved["reference"][0]),
        "source_files": [str(path) for path in saved["source"]],
        "reference_files": [str(path) for path in saved["reference"]],
    }
    try:
        task = run_registration_pipeline.delay(job_id=job.id, params=params)
        job.celery_task_id = task.id
        db.commit()
        db.refresh(job)
    except Exception as exc:
        job.status = JobStatus.FAILED
        job.error_message = f"Could not enqueue pipeline task: {exc}"
        job.completed_at = datetime.now(timezone.utc)
        db.commit()
        db.refresh(job)

    return _to_job_detail(job)


@router.get("/jobs", response_model=JobListResponse)
def list_jobs(skip: int = 0, limit: int = 50, db: Session = Depends(get_db)):
    limit = max(1, min(limit, 100))
    total = db.query(RegistrationJob).count()
    jobs = (
        db.query(RegistrationJob)
        .order_by(desc(RegistrationJob.created_at))
        .offset(max(0, skip))
        .limit(limit)
        .all()
    )
    return {"total": total, "jobs": [_to_job_detail(job) for job in jobs]}


@router.get("/jobs/{job_id}", response_model=JobDetailResponse)
def get_job(job_id: str, db: Session = Depends(get_db)):
    return _to_job_detail(_job_or_404(db, job_id))


@router.get("/jobs/{job_id}/status", response_model=JobStatusResponse)
def get_job_status(job_id: str, db: Session = Depends(get_db)):
    job = _job_or_404(db, job_id)
    return {
        "id": job.id,
        "status": job.status.value if isinstance(job.status, JobStatus) else str(job.status),
        "progress_pct": job.progress_pct or 0,
        "current_stage": job.current_stage,
        "error_message": job.error_message,
        "created_at": job.created_at,
        "started_at": job.started_at,
        "completed_at": job.completed_at,
    }


@router.get("/jobs/{job_id}/artifacts", response_model=ArtifactListResponse)
def list_job_artifacts(job_id: str, db: Session = Depends(get_db)):
    _job_or_404(db, job_id)
    return {"job_id": job_id, "artifacts": _artifact_manifest(job_id)}


@router.get("/jobs/{job_id}/artifacts/{artifact_id}/download")
def download_artifact(job_id: str, artifact_id: str, db: Session = Depends(get_db)):
    _job_or_404(db, job_id)
    entry = _find_artifact(job_id, artifact_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Artifact not found")
    return FileResponse(entry["path"], media_type=entry["media_type"], filename=entry["file_name"])


@router.get("/jobs/{job_id}/artifacts/{artifact_id}/preview")
def preview_artifact(job_id: str, artifact_id: str, db: Session = Depends(get_db)):
    _job_or_404(db, job_id)
    entry = _find_artifact(job_id, artifact_id)
    if not entry or not entry["previewable"]:
        raise HTTPException(status_code=404, detail="Preview is not available for this artifact")
    return FileResponse(
        entry["path"],
        media_type=entry["media_type"],
        headers={"Content-Disposition": f'inline; filename="{entry["file_name"]}"'},
    )


@router.get("/downloads/{job_id}/{file_type}")
def download_output_file(job_id: str, file_type: str, db: Session = Depends(get_db)):
    """Backward-compatible download route for the original output links."""
    job = _job_or_404(db, job_id)
    path_map = {
        "registered_tif": job.registered_tif_path,
        "native_tif": job.native_tif_path,
        "heatmap_png": job.heatmap_preview_path,
        "candidate_csv": job.candidate_csv_path,
        "subpixel_csv": job.subpixel_csv_path,
        "hybrid_model_json": job.hybrid_model_path,
        "metrics_json": str(OUTPUTS_DIR / job_id / "diagnostics" / "verification_metrics.json"),
        "registered_preview": str(PREVIEWS_DIR / job_id / "registered_preview.png"),
        "sidebyside_png": job.sidebyside_preview_path,
        "falsecolor_png": job.falsecolor_preview_path,
    }
    path_str = path_map.get(file_type)
    if file_type not in path_map:
        raise HTTPException(status_code=400, detail=f"Invalid file_type: {file_type}")
    if not path_str or not Path(path_str).is_file():
        raise HTTPException(status_code=404, detail=f"File not found for type: {file_type}")
    media_type = mimetypes.guess_type(path_str)[0] or "application/octet-stream"
    return FileResponse(path_str, media_type=media_type, filename=Path(path_str).name)


def _artifact_stage(relative_path: str) -> tuple[str, str]:
    name = Path(relative_path).name.lower()
    normalized = relative_path.lower().replace("\\", "/")
    if name.startswith("stage_01_") or name.startswith("bbox_overlap_meta") or "/georeferenced/" in normalized or "/harmonized/" in normalized or "iirs_selected_band" in name:
        return "stage_1", "harmonization"
    if "coarse_alignment" in name:
        return "stage_2", "coarse_alignment"
    if any(token in normalized for token in ("match_visualizations", "tile_pngs")) or name in {
        "candidate_matches.csv", "subpixel_tie_points.csv", "tie_points_inliers.csv", "hybrid_transform_model.json"
    }:
        return "stage_3", "matching"
    if name.startswith("registered_") or "warp_composite_overlay" in name:
        return "stage_4", "warping"
    if name in {
        "verification_metrics.json", "registration_verification.png", "difference_heatmap.png",
        "overview_side_by_side.png", "overview_false_color.png"
    } or "heatmap" in name:
        return "stage_5", "verification"
    if "preview" in name or "sidebyside" in name or "falsecolor" in name:
        return "stage_4", "preview"
    return "pipeline", "intermediate"


def _artifact_manifest(job_id: str) -> List[Dict[str, Any]]:
    roots = [("output", OUTPUTS_DIR / job_id), ("preview", PREVIEWS_DIR / job_id)]
    artifacts: List[Dict[str, Any]] = []
    for root_key, root in roots:
        if not root.is_dir():
            continue
        resolved_root = root.resolve()
        for path in root.rglob("*"):
            if not path.is_file() or path.is_symlink() or path.name.startswith("."):
                continue
            resolved = path.resolve()
            try:
                relative = resolved.relative_to(resolved_root).as_posix()
            except ValueError:
                continue
            artifact_id = hashlib.sha256(f"{root_key}:{relative}".encode("utf-8")).hexdigest()[:24]
            stage, category = _artifact_stage(relative)
            media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            preview_suffix = path.suffix.lower() in {".png", ".jpg", ".jpeg"}
            preview_name = path.name.lower()
            bounded_preview = (
                preview_name.startswith("stage_01_")
                or preview_name.endswith("_preview.png")
                or preview_name in {
                    "warp_composite_overlay.png",
                    "registration_verification.png",
                    "difference_heatmap.png",
                    "overview_side_by_side.png",
                    "overview_false_color.png",
                }
            )
            previewable = preview_suffix and bounded_preview and path.stat().st_size <= 8 * 1024 * 1024
            base = f"/api/v1/jobs/{job_id}/artifacts/{artifact_id}"
            artifacts.append({
                "artifact_id": artifact_id,
                "file_name": path.name,
                "relative_path": relative,
                "stage": stage,
                "category": category,
                "media_type": media_type,
                "size_bytes": path.stat().st_size,
                "previewable": previewable,
                "download_url": f"{base}/download",
                "preview_url": f"{base}/preview" if previewable else None,
                "path": str(path),
            })
    return sorted(artifacts, key=lambda item: (item["stage"], item["relative_path"].casefold()))


def _find_artifact(job_id: str, artifact_id: str) -> Optional[Dict[str, Any]]:
    return next((item for item in _artifact_manifest(job_id) if item["artifact_id"] == artifact_id), None)


def _to_job_detail(job: RegistrationJob) -> dict:
    base_url = f"/api/v1/downloads/{job.id}"
    outputs = {}
    links = {
        "registered_tif": ("registered_tif", job.registered_tif_path),
        "native_tif": ("native_tif", job.native_tif_path),
        "candidate_csv": ("candidate_csv", job.candidate_csv_path),
        "subpixel_csv": ("subpixel_csv", job.subpixel_csv_path),
        "hybrid_model_json": ("hybrid_model_json", job.hybrid_model_path),
        "metrics_json": ("metrics_json", OUTPUTS_DIR / job.id / "diagnostics" / "verification_metrics.json"),
        "registered_preview": ("registered_preview", PREVIEWS_DIR / job.id / "registered_preview.png"),
        "heatmap_preview": ("heatmap_png", job.heatmap_preview_path),
        "sidebyside_preview": ("sidebyside_png", job.sidebyside_preview_path),
        "falsecolor_preview": ("falsecolor_png", job.falsecolor_preview_path),
    }
    for key, (file_type, path) in links.items():
        if path and Path(path).is_file():
            outputs[key] = f"{base_url}/{file_type}"
    metrics = dict(job.full_metrics_json or {})
    # Do not send machine-local paths in a browser API response.
    for path_key in ("registered_raster", "reference_raster"):
        if metrics.get(path_key):
            metrics[path_key] = Path(metrics[path_key]).name
    metrics_value = metrics or None
    return {
        "id": job.id,
        "celery_task_id": job.celery_task_id,
        "sensor_src": job.sensor_src,
        "sensor_ref": job.sensor_ref,
        "source_filename": job.source_filename,
        "reference_filename": job.reference_filename,
        "method": job.method,
        "status": job.status.value if isinstance(job.status, JobStatus) else str(job.status),
        "progress_pct": job.progress_pct or 0,
        "current_stage": job.current_stage,
        "error_message": job.error_message,
        "created_at": job.created_at,
        "started_at": job.started_at,
        "completed_at": job.completed_at,
        "wall_time_seconds": job.wall_time_seconds,
        "run_config": job.run_config_json or {},
        "stages": job.stage_events_json or [],
        "metrics": metrics_value,
        "outputs": outputs or None,
    }
