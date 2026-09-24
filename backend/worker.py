"""
Celery Task Worker for Lunar Registration Pipeline.
"""
import os
import sys
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import rasterio
import cv2
from celery import Celery

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from backend.config import CELERY_BROKER_URL, CELERY_RESULT_BACKEND, OUTPUTS_DIR, PREVIEWS_DIR
from backend.database import SessionLocal
from backend.models import RegistrationJob, JobStatus

celery_app = Celery("lunar_reg", broker=CELERY_BROKER_URL, backend=CELERY_RESULT_BACKEND)
celery_app.conf.update(
    task_serializer="json", result_serializer="json", accept_content=["json"],
    timezone="UTC", enable_utc=True, task_track_started=True,
    worker_prefetch_multiplier=1, task_soft_time_limit=7200, task_time_limit=7800
)

def _update_job(db, job_id, **kwargs):
    job = db.query(RegistrationJob).filter(RegistrationJob.id == job_id).first()
    if job:
        for k, v in kwargs.items():
            setattr(job, k, v)
        db.commit()
    return job

def _export_preview_png(tif_path: Path, out_png: Path, max_dim: int = 1024):
    with rasterio.open(tif_path) as src:
        h, w = src.height, src.width
        if h > max_dim * 2 or w > max_dim * 2:
            size = min(max_dim * 2, min(h, w))
            r0, c0 = (h - size) // 2, (w - size) // 2
            win = rasterio.windows.Window(c0, r0, size, size)
            data = src.read(1, window=win)
        else:
            dec = max(1, max(h, w) // max_dim)
            data = src.read(1, out_shape=(max(1, h // dec), max(1, w // dec)), resampling=rasterio.enums.Resampling.bilinear)

    vals = data[data > 0].ravel()
    if vals.size > 100:
        p1, p99 = np.percentile(vals, (1, 99))
        p99 = max(p99, p1 + 1)
        scaled = np.clip((data.astype(np.float32) - p1) / (p99 - p1) * 255.0, 0, 255).astype(np.uint8)
    else:
        scaled = np.zeros(data.shape, dtype=np.uint8)

    if scaled.shape[0] > max_dim or scaled.shape[1] > max_dim:
        scaled = cv2.resize(scaled, (max_dim, max_dim), interpolation=cv2.INTER_AREA)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_png), scaled)

def _export_heatmap_preview(registered_tif: Path, ref_tif: Path, out_png: Path, max_dim: int = 1024):
    try:
        with rasterio.open(registered_tif) as s, rasterio.open(ref_tif) as r:
            scale = max_dim / max(s.width, s.height)
            ow = max(1, int(s.width  * scale))
            oh = max(1, int(s.height * scale))
            i1 = s.read(1, out_shape=(oh, ow), resampling=rasterio.enums.Resampling.bilinear)
            i2 = r.read(1, out_shape=(oh, ow), resampling=rasterio.enums.Resampling.bilinear)

        def _stretch(arr):
            v = arr[arr > 0].ravel()
            if v.size < 100: return arr.astype(np.uint8)
            p1, p99 = np.percentile(v, (1, 99))
            return np.clip((arr.astype(np.float32) - p1) / (max(p99, p1 + 1) - p1) * 255, 0, 255).astype(np.uint8)

        diff = cv2.absdiff(_stretch(i1), _stretch(i2))
        colored = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_png), colored)
    except Exception as e:
        print(f"[HEATMAP] Warning: {e}")

@celery_app.task(bind=True, name="backend.worker.run_registration_pipeline")
def run_registration_pipeline(self, job_id: str, params: dict):
    db = SessionLocal()
    job_out_dir = OUTPUTS_DIR / job_id
    job_out_dir.mkdir(parents=True, exist_ok=True)

    try:
        _update_job(db, job_id, status=JobStatus.PROCESSING, celery_task_id=self.request.id, started_at=datetime.now(timezone.utc), current_stage="Stage 1/5: Scale Harmonization", progress_pct=5)
        
        args = SimpleNamespace(
            source=Path(params["source_path"]), reference=Path(params["reference_path"]),
            sensor_src=params["sensor_src"], sensor_ref=params["sensor_ref"], out_dir=job_out_dir,
            method=params.get("method", "loftr"), structural_method=params.get("structural_method", "phase_congruency"),
            coarse_method=params.get("coarse_method", "auto"), poly_degree=params.get("poly_degree", 2),
            tps_smoothing=params.get("tps_smoothing", 0.05), warp_order=params.get("warp_order", 3),
            wac_band=params.get("wac_band", 7), grid_size=[params.get("grid_rows", 4), params.get("grid_cols", 4)],
            force=False, skip_verify=False
        )

        import time as _time
        t_start = _time.time()
        
        from run_pipeline import run_pipeline as _run_pipeline
        _run_pipeline(args)
        wall_time = _time.time() - t_start

        registered_tif = job_out_dir / "registered_subpixel.tif"
        diag_dir       = job_out_dir / "diagnostics"
        preview_dir = PREVIEWS_DIR / job_id
        preview_dir.mkdir(parents=True, exist_ok=True)

        heatmap_png   = preview_dir / "heatmap.png"
        reg_preview_png = preview_dir / "registered_preview.png"
        ref_cropped = job_out_dir / "harmonized" / "bbox_overlap_ref_cropped.tif"

        if registered_tif.exists():
            _export_preview_png(registered_tif, reg_preview_png)
            if ref_cropped.exists():
                _export_heatmap_preview(registered_tif, ref_cropped, heatmap_png)

        diag_heatmap = diag_dir / "difference_heatmap.png"
        if diag_heatmap.exists() and not heatmap_png.exists():
            shutil.copy(diag_heatmap, heatmap_png)

        metrics = {}
        metrics_file = diag_dir / "verification_metrics.json"
        if metrics_file.exists():
            with open(metrics_file, "r") as f:
                metrics = json.load(f)

        native_tifs = sorted(job_out_dir.glob("registered_native_*m.tif"))
        
        _update_job(
            db, job_id, status=JobStatus.SUCCESS, completed_at=datetime.now(timezone.utc), progress_pct=100,
            current_stage="Complete", wall_time_seconds=wall_time, output_dir=str(job_out_dir),
            registered_tif_path=str(registered_tif) if registered_tif.exists() else None,
            native_tif_path=str(native_tifs[0]) if native_tifs else None,
            heatmap_preview_path=str(heatmap_png) if heatmap_png.exists() else None,
            rmse_px=metrics.get("rmse_px"), rmse_meters=metrics.get("rmse_meters"), verdict=metrics.get("verdict"),
            full_metrics_json=metrics
        )
        return {"job_id": job_id, "status": "SUCCESS"}
    except Exception as exc:
        _update_job(db, job_id, status=JobStatus.FAILED, completed_at=datetime.now(timezone.utc), error_message=str(exc), current_stage="FAILED")
        raise exc
    finally:
        db.close()
