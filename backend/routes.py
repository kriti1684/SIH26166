"""
FastAPI Routes for Lunar Registration Backend.
Handles job submission, status polling, and file downloads.
"""

import os
import shutil
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from sqlalchemy import desc

from backend.config import UPLOADS_DIR
from backend.database import get_db
from backend.models import RegistrationJob, JobStatus
from backend.schemas import (
    JobDetailResponse,
    JobListResponse,
    JobStatusResponse,
    SensorSrc,
    SensorRef,
)
from backend.worker import run_registration_pipeline


router = APIRouter()


@router.post("/jobs", response_model=JobDetailResponse, status_code=201)
async def create_registration_job(
    source_file: UploadFile = File(...),
    reference_file: UploadFile = File(...),
    sensor_src: SensorSrc = Form(...),
    sensor_ref: SensorRef = Form(...),
    method: str = Form("loftr"),
    structural_method: str = Form("phase_congruency"),
    coarse_method: str = Form("auto"),
    poly_degree: int = Form(2),
    tps_smoothing: float = Form(0.05),
    warp_order: int = Form(3),
    wac_band: int = Form(7),
    grid_rows: int = Form(4),
    grid_cols: int = Form(4),
    db: Session = Depends(get_db)
):
    job = RegistrationJob(
        sensor_src=sensor_src,
        sensor_ref=sensor_ref,
        source_filename=source_file.filename or "source.tif",
        reference_filename=reference_file.filename or "reference.tif",
        method=method,
        structural_method=structural_method,
        coarse_method=coarse_method,
        poly_degree=poly_degree,
        tps_smoothing=tps_smoothing,
        warp_order=warp_order,
        wac_band=wac_band,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    job_upload_dir = UPLOADS_DIR / job.id
    job_upload_dir.mkdir(parents=True, exist_ok=True)
    
    src_path = job_upload_dir / job.source_filename
    ref_path = job_upload_dir / job.reference_filename

    with src_path.open("wb") as buffer:
        shutil.copyfileobj(source_file.file, buffer)
    with ref_path.open("wb") as buffer:
        shutil.copyfileobj(reference_file.file, buffer)

    job.source_path = str(src_path)
    job.reference_path = str(ref_path)
    db.commit()

    params = {
        "source_path": str(src_path),
        "reference_path": str(ref_path),
        "sensor_src": sensor_src,
        "sensor_ref": sensor_ref,
        "method": method,
        "structural_method": structural_method,
        "coarse_method": coarse_method,
        "poly_degree": poly_degree,
        "tps_smoothing": tps_smoothing,
        "warp_order": warp_order,
        "wac_band": wac_band,
        "grid_rows": grid_rows,
        "grid_cols": grid_cols,
    }

    task = run_registration_pipeline.delay(job_id=job.id, params=params)
    
    job.celery_task_id = task.id
    db.commit()
    db.refresh(job)

    return _to_job_detail(job)


@router.get("/jobs", response_model=JobListResponse)
def list_jobs(skip: int = 0, limit: int = 50, db: Session = Depends(get_db)):
    total = db.query(RegistrationJob).count()
    jobs = (
        db.query(RegistrationJob)
        .order_by(desc(RegistrationJob.created_at))
        .offset(skip)
        .limit(limit)
        .all()
    )
    return {"total": total, "jobs": [_to_job_detail(j) for j in jobs]}


@router.get("/jobs/{job_id}", response_model=JobDetailResponse)
def get_job(job_id: str, db: Session = Depends(get_db)):
    job = db.query(RegistrationJob).filter(RegistrationJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return _to_job_detail(job)


@router.get("/jobs/{job_id}/status", response_model=JobStatusResponse)
def get_job_status(job_id: str, db: Session = Depends(get_db)):
    job = db.query(RegistrationJob).filter(RegistrationJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.get("/downloads/{job_id}/{file_type}")
def download_output_file(job_id: str, file_type: str, db: Session = Depends(get_db)):
    job = db.query(RegistrationJob).filter(RegistrationJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    path_str = None
    if file_type == "registered_tif":
        path_str = job.registered_tif_path
    elif file_type == "native_tif":
        path_str = job.native_tif_path
    elif file_type == "heatmap_png":
        path_str = job.heatmap_preview_path
    else:
        raise HTTPException(status_code=400, detail=f"Invalid file_type: {file_type}")

    if not path_str or not os.path.exists(path_str):
        raise HTTPException(status_code=404, detail=f"File not found for type: {file_type}")

    filename = Path(path_str).name
    media_type = "image/png" if path_str.endswith(".png") else "application/octet-stream"
    return FileResponse(path_str, media_type=media_type, filename=filename)


def _to_job_detail(job: RegistrationJob) -> dict:
    d = {
        "id": job.id,
        "celery_task_id": job.celery_task_id,
        "sensor_src": job.sensor_src,
        "sensor_ref": job.sensor_ref,
        "source_filename": job.source_filename,
        "reference_filename": job.reference_filename,
        "method": job.method,
        "status": job.status,
        "progress_pct": job.progress_pct,
        "current_stage": job.current_stage,
        "error_message": job.error_message,
        "created_at": job.created_at,
        "started_at": job.started_at,
        "completed_at": job.completed_at,
        "wall_time_seconds": job.wall_time_seconds,
    }

    if job.status == JobStatus.SUCCESS or job.full_metrics_json:
        d["metrics"] = {
            "rmse_px": job.rmse_px,
            "rmse_meters": job.rmse_meters,
            "verdict": job.verdict,
        }
    
    links = {}
    base_url = f"/api/v1/downloads/{job.id}"
    if job.registered_tif_path:
        links["registered_tif"] = f"{base_url}/registered_tif"
    if job.native_tif_path:
        links["native_tif"] = f"{base_url}/native_tif"
    if job.heatmap_preview_path:
        links["heatmap_preview"] = f"{base_url}/heatmap_png"
        
    d["outputs"] = links if links else None
    return d
