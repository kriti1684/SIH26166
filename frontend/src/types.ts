export type JobState = 'PENDING' | 'PROCESSING' | 'SUCCESS' | 'FAILED'

export interface PipelineStageEvent {
  stage_id: string
  title: string
  state: 'running' | 'complete' | 'failed' | string
  progress_pct: number
  message: string
  details: Record<string, unknown>
  created_at: string
}

export interface JobOutputLinks {
  registered_tif?: string
  native_tif?: string
  candidate_csv?: string
  subpixel_csv?: string
  hybrid_model_json?: string
  metrics_json?: string
  registered_preview?: string
  heatmap_preview?: string
  sidebyside_preview?: string
  falsecolor_preview?: string
}

export interface RegistrationJob {
  id: string
  celery_task_id?: string | null
  sensor_src: string
  sensor_ref: string
  source_filename: string
  reference_filename: string
  method: string
  status: JobState
  progress_pct: number
  current_stage?: string | null
  error_message?: string | null
  created_at: string
  started_at?: string | null
  completed_at?: string | null
  wall_time_seconds?: number | null
  run_config: Record<string, unknown>
  stages: PipelineStageEvent[]
  metrics?: Record<string, unknown> | null
  outputs?: JobOutputLinks | null
}

export interface ArtifactInfo {
  artifact_id: string
  file_name: string
  relative_path: string
  stage: string
  category: string
  media_type: string
  size_bytes: number
  previewable: boolean
  download_url: string
  preview_url?: string | null
}

export interface ArtifactList {
  job_id: string
  artifacts: ArtifactInfo[]
}

export interface HealthStatus {
  status: string
  version?: string
  db?: string
  celery_queue?: string
  gpu_available?: boolean
  gpu_name?: string | null
}

export interface RegistrationConfig {
  source: File | null
  reference: File | null
  sourceSidecars: File[]
  referenceSidecars: File[]
  sensorSrc: string
  sensorRef: string
  method: string
  structuralMethod: string
  coarseMethod: string
  polyDegree: number
  tpsSmoothing: number
  ransacThreshold: number
  warpOrder: number
  wacBand: number
  gridRows: number
  gridCols: number
  exportNative: boolean
}
