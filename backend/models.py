"""
SQLAlchemy ORM Models.
"""
import uuid
from datetime import datetime, timezone
import enum

from sqlalchemy import Column, String, Float, Integer, DateTime, JSON, Text, Enum as SAEnum
from backend.database import Base

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)

class JobStatus(str, enum.Enum):
    PENDING    = "PENDING"
    PROCESSING = "PROCESSING"
    SUCCESS    = "SUCCESS"
    FAILED     = "FAILED"

class RegistrationJob(Base):
    __tablename__ = "registration_jobs"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    celery_task_id = Column(String(36), nullable=True, index=True)

    sensor_src   = Column(String(16), nullable=False)
    sensor_ref   = Column(String(16), nullable=False)
    source_filename   = Column(String(512), nullable=False)
    reference_filename = Column(String(512), nullable=False)
    source_path    = Column(String(1024), nullable=False)
    reference_path = Column(String(1024), nullable=False)

    method              = Column(String(16), default="loftr")
    structural_method   = Column(String(32), default="phase_congruency")
    coarse_method       = Column(String(16), default="auto")
    poly_degree         = Column(Integer,   default=2)
    tps_smoothing       = Column(Float,     default=0.05)
    warp_order          = Column(Integer,   default=3)
    wac_band            = Column(Integer,   default=7)
    grid_rows           = Column(Integer,   default=4)
    grid_cols           = Column(Integer,   default=4)

    created_at    = Column(DateTime(timezone=True), default=_utcnow)
    started_at    = Column(DateTime(timezone=True), nullable=True)
    completed_at  = Column(DateTime(timezone=True), nullable=True)

    status = Column(SAEnum(JobStatus), nullable=False, default=JobStatus.PENDING, index=True)
    progress_pct  = Column(Integer, default=0)
    current_stage = Column(String(128), nullable=True)
    error_message = Column(Text, nullable=True)

    output_dir           = Column(String(1024), nullable=True)
    registered_tif_path  = Column(String(1024), nullable=True)
    native_tif_path      = Column(String(1024), nullable=True)
    candidate_csv_path   = Column(String(1024), nullable=True)
    subpixel_csv_path    = Column(String(1024), nullable=True)
    hybrid_model_path    = Column(String(1024), nullable=True)
    heatmap_preview_path = Column(String(1024), nullable=True)
    sidebyside_preview_path = Column(String(1024), nullable=True)
    falsecolor_preview_path = Column(String(1024), nullable=True)

    rmse_px                      = Column(Float, nullable=True)
    rmse_meters                  = Column(Float, nullable=True)
    subpixel_error_px            = Column(Float, nullable=True)
    inlier_match_count           = Column(Integer, nullable=True)
    candidate_match_count        = Column(Integer, nullable=True)
    inlier_ratio                 = Column(Float, nullable=True)
    spatial_entropy_h            = Column(Float, nullable=True)
    grid_coverage_pct            = Column(Float, nullable=True)
    composite_confidence         = Column(Float, nullable=True)
    verdict                      = Column(String(64), nullable=True)
    full_metrics_json            = Column(JSON, nullable=True)

    wall_time_seconds = Column(Float, nullable=True)
