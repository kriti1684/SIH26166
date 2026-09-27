from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
import enum


class SensorSrc(str, enum.Enum):
    OHRC = "OHRC"
    IIRS = "IIRS"
    TMC = "TMC"
    TMC2 = "TMC2"


class SensorRef(str, enum.Enum):
    NAC = "NAC"
    WAC = "WAC"
    SELENE = "SELENE"
    TC = "TC"


class JobStatusResponse(BaseModel):
    id: str
    status: str
    progress_pct: int
    current_stage: Optional[str] = None
    error_message: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


class PipelineStageEvent(BaseModel):
    stage_id: str
    title: str
    state: str
    progress_pct: int
    message: str
    details: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class ArtifactInfo(BaseModel):
    artifact_id: str
    file_name: str
    relative_path: str
    stage: str
    category: str
    media_type: str
    size_bytes: int
    previewable: bool
    download_url: str
    preview_url: Optional[str] = None


class ArtifactListResponse(BaseModel):
    job_id: str
    artifacts: List[ArtifactInfo] = Field(default_factory=list)


class JobOutputLinks(BaseModel):
    registered_tif: Optional[str] = None
    native_tif: Optional[str] = None
    candidate_csv: Optional[str] = None
    subpixel_csv: Optional[str] = None
    hybrid_model_json: Optional[str] = None
    metrics_json: Optional[str] = None
    registered_preview: Optional[str] = None
    heatmap_preview: Optional[str] = None
    sidebyside_preview: Optional[str] = None
    falsecolor_preview: Optional[str] = None


class JobDetailResponse(BaseModel):
    id: str
    sensor_src: str
    sensor_ref: str
    source_filename: str
    reference_filename: str
    method: str
    status: str
    progress_pct: int
    current_stage: Optional[str] = None
    error_message: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    wall_time_seconds: Optional[float] = None
    run_config: Dict[str, Any] = Field(default_factory=dict)
    stages: List[PipelineStageEvent] = Field(default_factory=list)
    metrics: Optional[Dict[str, Any]] = None
    outputs: Optional[JobOutputLinks] = None

    model_config = {"from_attributes": True}


class JobListResponse(BaseModel):
    total: int
    jobs: List[JobDetailResponse]
