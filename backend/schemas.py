"""
Pydantic Schemas for the API.
"""
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field

import enum

class SensorSrc(str, enum.Enum):
    OHRC = "OHRC"
    IIRS = "IIRS"
    TMC  = "TMC"
    TMC2 = "TMC2"

class SensorRef(str, enum.Enum):
    NAC    = "NAC"
    WAC    = "WAC"
    SELENE = "SELENE"
    TC     = "TC"

class JobStatusResponse(BaseModel):
    id:            str
    status:        str
    progress_pct:  int
    current_stage: Optional[str]
    error_message: Optional[str]
    created_at:    datetime
    started_at:    Optional[datetime]
    completed_at:  Optional[datetime]

    model_config = {"from_attributes": True}

class VerificationMetrics(BaseModel):
    rmse_px:                Optional[float]
    rmse_meters:            Optional[float]
    subpixel_error_px:      Optional[float]
    inlier_match_count:     Optional[int]
    candidate_match_count:  Optional[int]
    inlier_ratio:           Optional[float]
    spatial_entropy_h:      Optional[float]
    grid_coverage_pct:      Optional[float]
    composite_confidence:   Optional[float]
    verdict:                Optional[str]

class JobOutputLinks(BaseModel):
    registered_tif:   Optional[str]
    native_tif:       Optional[str]
    candidate_csv:    Optional[str]
    subpixel_csv:     Optional[str]
    hybrid_model_json: Optional[str]
    heatmap_preview:  Optional[str]
    sidebyside_preview: Optional[str]
    falsecolor_preview: Optional[str]

class JobDetailResponse(BaseModel):
    id:             str
    celery_task_id: Optional[str]
    sensor_src:     str
    sensor_ref:     str
    source_filename:    str
    reference_filename: str
    method:         str
    status:         str
    progress_pct:   int
    current_stage:  Optional[str]
    error_message:  Optional[str]
    created_at:     datetime
    started_at:     Optional[datetime]
    completed_at:   Optional[datetime]
    wall_time_seconds: Optional[float]
    metrics:        Optional[VerificationMetrics]
    outputs:        Optional[JobOutputLinks]

    model_config = {"from_attributes": True}

class JobListResponse(BaseModel):
    total: int
    jobs:  List[JobDetailResponse]
