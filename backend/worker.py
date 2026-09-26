"""Background worker that runs the lunar registration pipeline and persists progress."""
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from backend.config import OUTPUTS_DIR, PREVIEWS_DIR
from backend.database import SessionLocal
from backend.models import JobStatus, RegistrationJob
from src.registration.web_previews import export_raster_preview
from run_pipeline import run_pipeline as pipeline


def _update_job(db, job_id, **kwargs):
    job = db.query(RegistrationJob).filter(RegistrationJob.id == job_id).first()
    if job:
        for key, value in kwargs.items():
            setattr(job, key, value)
        db.commit()
    return job


def _append_stage_event(db, job_id: str, event: dict):
    job = db.query(RegistrationJob).filter(RegistrationJob.id == job_id).first()
    if not job:
        return
    stored = dict(event)
    stored["created_at"] = datetime.now(timezone.utc).isoformat()
    events = list(job.stage_events_json or [])
    events.append(stored)
    job.stage_events_json = events[-40:]
    job.current_stage = event.get("title", job.current_stage)
    job.progress_pct = int(event.get("progress_pct", job.progress_pct or 0))
    db.commit()


def _safe_metrics(metrics: dict) -> dict:
    result = dict(metrics or {})
    for path_key in ("registered_raster", "reference_raster"):
        if result.get(path_key):
            result[path_key] = Path(result[path_key]).name
    return result


def run_registration_pipeline(job_id: str, params: dict):
    db = SessionLocal()
    job_out_dir = OUTPUTS_DIR / job_id
    job_out_dir.mkdir(parents=True, exist_ok=True)
    try:
        _update_job(
            db,
            job_id,
            status=JobStatus.PROCESSING,
            celery_task_id=f"local-{job_id[:8]}",
            started_at=datetime.now(timezone.utc),
            current_stage="Preparing pipeline",
            progress_pct=1,
        )

        args = SimpleNamespace(
            source=Path(params["source_path"]),
            reference=Path(params["reference_path"]),
            sensor_src=params["sensor_src"],
            sensor_ref=params["sensor_ref"],
            out_dir=job_out_dir,
            method=params.get("method", "loftr"),
            structural_method=params.get("structural_method", "phase_congruency"),
            coarse_method=params.get("coarse_method", "auto"),
            coarse_dx=None,
            coarse_dy=None,
            poly_degree=params.get("poly_degree", 2),
            tps_smoothing=params.get("tps_smoothing", 0.05),
            ransac_threshold=params.get("ransac_threshold", 1.2),
            warp_order=params.get("warp_order", 3),
            wac_band=params.get("wac_band", 7),
            grid_size=[params.get("grid_rows", 4), params.get("grid_cols", 4)],
            force=False,
            skip_verify=False,
            export_native=params.get("export_native", False),
        )

        start = datetime.now(timezone.utc)

        def publish_stage(event):
            _append_stage_event(db, job_id, event)

        pipeline(args, progress_callback=publish_stage)
        wall_time = (datetime.now(timezone.utc) - start).total_seconds()

        diag_dir = job_out_dir / "diagnostics"
        preview_dir = PREVIEWS_DIR / job_id
        preview_dir.mkdir(parents=True, exist_ok=True)
        registered_tif = job_out_dir / "registered_subpixel.tif"
        registered_preview = preview_dir / "registered_preview.png"
        if registered_tif.is_file():
            try:
                export_raster_preview(registered_tif, registered_preview)
            except Exception as preview_error:
                print(f"[PREVIEW] Could not create registered quicklook: {preview_error}")

        # Keep the old preview URL working while the artifact manifest exposes
        # all stage and final products through the new API.
        heatmap_png = preview_dir / "heatmap.png"
        diag_heatmap = diag_dir / "difference_heatmap.png"
        if diag_heatmap.is_file():
            shutil.copy2(diag_heatmap, heatmap_png)

        metrics_file = diag_dir / "verification_metrics.json"
        metrics = {}
        if metrics_file.is_file():
            with metrics_file.open("r", encoding="utf-8") as stream:
                metrics = json.load(stream)

        native_tifs = sorted(job_out_dir.glob("registered_native_*m.tif"))
        candidate_csv = job_out_dir / "candidate_matches.csv"
        subpixel_csv = job_out_dir / "subpixel_tie_points.csv"
        model_json = job_out_dir / "hybrid_transform_model.json"
        sidebyside = diag_dir / "overview_side_by_side.png"
        falsecolor = diag_dir / "overview_false_color.png"
        _append_stage_event(db, job_id, {
            "stage_id": "complete",
            "title": "Pipeline complete",
            "state": "complete",
            "progress_pct": 100,
            "message": "All pipeline stages finished. Pipeline execution status is separate from the scientific verification verdict.",
            "details": {"verification_verdict": metrics.get("verdict")},
        })
        _update_job(
            db,
            job_id,
            status=JobStatus.SUCCESS,
            completed_at=datetime.now(timezone.utc),
            progress_pct=100,
            current_stage="Complete",
            wall_time_seconds=wall_time,
            output_dir=str(job_out_dir),
            registered_tif_path=str(registered_tif) if registered_tif.is_file() else None,
            native_tif_path=str(native_tifs[0]) if native_tifs else None,
            candidate_csv_path=str(candidate_csv) if candidate_csv.is_file() else None,
            subpixel_csv_path=str(subpixel_csv) if subpixel_csv.is_file() else None,
            hybrid_model_path=str(model_json) if model_json.is_file() else None,
            heatmap_preview_path=str(heatmap_png) if heatmap_png.is_file() else None,
            sidebyside_preview_path=str(sidebyside) if sidebyside.is_file() else None,
            falsecolor_preview_path=str(falsecolor) if falsecolor.is_file() else None,
            rmse_px=metrics.get("rmse_px"),
            rmse_meters=metrics.get("rmse_meters"),
            inlier_match_count=metrics.get("inlier_match_count"),
            candidate_match_count=metrics.get("total_candidate_matches"),
            inlier_ratio=metrics.get("inlier_ratio"),
            spatial_entropy_h=metrics.get("spatial_entropy_h"),
            grid_coverage_pct=metrics.get("convex_hull_coverage_pct"),
            composite_confidence=metrics.get("composite_scientific_confidence"),
            verdict=metrics.get("verdict"),
            full_metrics_json=_safe_metrics(metrics),
        )
        return {"job_id": job_id, "status": "SUCCESS", "verdict": metrics.get("verdict")}
    except Exception as exc:
        job = db.query(RegistrationJob).filter(RegistrationJob.id == job_id).first()
        if job:
            events = list(job.stage_events_json or [])
            events.append({
                "stage_id": "pipeline",
                "title": job.current_stage or "Pipeline",
                "state": "failed",
                "progress_pct": int(job.progress_pct or 0),
                "message": str(exc),
                "details": {},
                "created_at": datetime.now(timezone.utc).isoformat(),
            })
            job.stage_events_json = events[-40:]
            job.status = JobStatus.FAILED
            job.completed_at = datetime.now(timezone.utc)
            job.error_message = str(exc)
            job.current_stage = "FAILED"
            db.commit()
        raise
    finally:
        db.close()
