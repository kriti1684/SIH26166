import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { FormEvent } from 'react'
import { apiUrl, createJob, fetchArtifacts, fetchHealth, fetchJob } from './api'
import { AppSidebar, type ActiveTab } from './components/AppSidebar'
import {
  ArtifactRow, formatBytes, formatMetric, formatPreviewTitle, formatTime, ImageLightbox, Mark, MetricCard,
  ProductDrop, QuickBatchUploader, SpaceWaitingCard, StageDetails,
} from './components'
import { NewProjectModal, ProjectsModal } from './components/ProjectModals'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'
import { Progress } from '@/components/ui/progress'
import type { ArtifactInfo, HealthStatus, PipelineStageEvent, RegistrationConfig, RegistrationJob } from './types'

const SOURCE_SENSORS = [
  { value: 'OHRC', label: 'OHRC · Optical High Resolution Camera' },
  { value: 'IIRS', label: 'IIRS · Imaging Infrared Spectrometer' },
  { value: 'TMC', label: 'TMC · Terrain Mapping Camera' },
  { value: 'TMC2', label: 'TMC2 · Terrain Mapping Camera 2' },
]

const REFERENCE_SENSORS = [
  { value: 'NAC', label: 'NAC · LRO Narrow Angle Camera' },
  { value: 'WAC', label: 'WAC · LRO Wide Angle Camera' },
  { value: 'SELENE', label: 'SELENE · Terrain Camera' },
  { value: 'TC', label: 'TC · Terrain Camera' },
]

export interface StageDefinition {
  id: string
  index: string
  shortName: string
  title: string
  description: string
  input: string
  output: string
}

export const STAGES: StageDefinition[] = [
  {
    id: 'stage_1',
    index: '01',
    shortName: 'Harmonize',
    title: 'Ingest & Scale Harmonization',
    description: 'Parses raw PDS / GeoTIFF sensor labels, computes SPICE orbital ray-tracing, projects both rasters onto a shared coordinate frame, and crops the mutual bounding-box overlap.',
    input: 'Source raster + reference raster + auxiliary sidecar metadata',
    output: 'Georeferenced overlap rasters + GSD scale ratio + initial quicklooks',
  },
  {
    id: 'stage_2',
    index: '02',
    shortName: 'Coarse Align',
    title: 'Coarse Alignment & Global Drift',
    description: 'Estimates integer translational pixel offset (dx, dy) and strip-wise drift using structural phase congruency and consensus voting on detected crater rim profiles.',
    input: 'Harmonized source and reference rasters',
    output: 'Global pixel offset (dx, dy) + coarse confidence + drift profile',
  },
  {
    id: 'stage_3',
    index: '03',
    shortName: 'Dense Match',
    title: 'Dense Feature Matching & Sub-Pixel ECC',
    description: 'Performs coarse-to-fine dense attention matching using LoFTR, followed by continuous Gauss-Newton Enhanced Correlation Coefficient (ECC) optimization for sub-pixel accuracy (< 0.2 px).',
    input: 'Overlap tiles + coarse alignment priors',
    output: 'Sub-pixel tie points (< 0.2 px) + candidate match coordinates',
  },
  {
    id: 'stage_4',
    index: '04',
    shortName: 'Hybrid Warp',
    title: 'Hybrid Transformation & Warping',
    description: 'Fits a 3-layer physics-grounded hybrid transformation (Affine global + local polynomial orbital drift + Thin Plate Spline) and applies streaming block-wise bicubic resampling onto the reference grid.',
    input: 'Sub-pixel tie points + source raster + sensor geometry',
    output: 'Registered GeoTIFF + transformation model JSON + residual vectors',
  },
  {
    id: 'stage_5',
    index: '05',
    shortName: 'Verification',
    title: 'Multi-Pillar Scientific Verification',
    description: 'Quantifies spatial leave-one-out cross-validation RMSE, convex hull coverage percentage, structural feature entropy, and issues the scientific verification verdict.',
    input: 'Registered raster + reference raster + tie-point evidence',
    output: 'Scientific verification verdict + metric report + difference heatmaps',
  },
]

const initialConfig: RegistrationConfig = {
  source: null,
  reference: null,
  sourceSidecars: [],
  referenceSidecars: [],
  sensorSrc: 'OHRC',
  sensorRef: 'NAC',
  method: 'loftr',
  structuralMethod: 'phase_congruency',
  coarseMethod: 'auto',
  polyDegree: 2,
  tpsSmoothing: 0.05,
  ransacThreshold: 1.2,
  warpOrder: 3,
  wacBand: 7,
  gridRows: 4,
  gridCols: 4,
  exportNative: false,
}
function App() {
  const [config, setConfig] = useState<RegistrationConfig>(initialConfig)
  const [job, setJob] = useState<RegistrationJob | null>(null)
  const [artifacts, setArtifacts] = useState<ArtifactInfo[]>([])
  const [health, setHealth] = useState<HealthStatus | null>(null)
  const [submitError, setSubmitError] = useState('')
  const [uploadProgress, setUploadProgress] = useState(0)
  const [submitting, setSubmitting] = useState(false)
  const [activeTab, setActiveTab] = useState<ActiveTab>('input')
  const [lightboxState, setLightboxState] = useState<{ list: ArtifactInfo[]; index: number } | null>(null)
  const [isNewProjectOpen, setIsNewProjectOpen] = useState(false)
  const [isProjectsOpen, setIsProjectsOpen] = useState(false)
  const [batchFeedback, setBatchFeedback] = useState<string | null>(null)
  const [theme, setTheme] = useState<'dark' | 'light'>(() => {
    const saved = localStorage.getItem('cs_theme')
    return (saved as 'dark' | 'light') || 'dark'
  })
  const pollingRef = useRef(false)

  useEffect(() => {
    const root = document.documentElement
    if (theme === 'dark') {
      root.classList.add('dark')
      root.classList.remove('light')
    } else {
      root.classList.add('light')
      root.classList.remove('dark')
    }
    localStorage.setItem('cs_theme', theme)
  }, [theme])

  const activeId = job?.id
  const busy = job?.status === 'PENDING' || job?.status === 'PROCESSING'

  const stageEvents = useMemo(() => {
    const byId = new Map<string, PipelineStageEvent>()
    for (const event of job?.stages ?? []) byId.set(event.stage_id, event)
    return byId
  }, [job?.stages])

  const verificationEvent = stageEvents.get('stage_5')
  const metrics = useMemo(() => {
    if (job?.metrics) return job.metrics
    const eventMetrics = verificationEvent?.details?.metrics
    return eventMetrics && typeof eventMetrics === 'object' ? eventMetrics as Record<string, unknown> : null
  }, [job?.metrics, verificationEvent])

  const failedStage = job?.stages.find((event) => event.state === 'failed')

  const refreshArtifacts = useCallback(async (jobId: string) => {
    try {
      const response = await fetchArtifacts(jobId)
      setArtifacts(response.artifacts)
    } catch {
    }
  }, [])

  useEffect(() => {
    let mounted = true
    window.localStorage.removeItem('chandrashakti.activeJob')
    const storedId = window.sessionStorage.getItem('chandrashakti.activeJob')
    void fetchHealth().then((h) => { if (mounted) setHealth(h) }).catch(() => { if (mounted) setHealth({ status: 'offline' }) })
    if (storedId) {
      void fetchJob(storedId).then((restored) => {
        if (mounted) {
          setJob(restored)
          void refreshArtifacts(storedId)
          if (restored.status === 'SUCCESS') setActiveTab('output')
        }
      }).catch(() => {
        window.sessionStorage.removeItem('chandrashakti.activeJob')
      })
    }
    const healthTimer = window.setInterval(() => {
      void fetchHealth().then((h) => { if (mounted) setHealth(h) }).catch(() => { if (mounted) setHealth({ status: 'offline' }) })
    }, 12000)
    return () => { mounted = false; window.clearInterval(healthTimer) }
  }, [refreshArtifacts])

  useEffect(() => {
    if (!activeId) {
      setArtifacts([])
      window.sessionStorage.removeItem('chandrashakti.activeJob')
      window.localStorage.removeItem('chandrashakti.activeJob')
      return
    }
    window.sessionStorage.setItem('chandrashakti.activeJob', activeId)
    void refreshArtifacts(activeId)
  }, [activeId, refreshArtifacts])

  useEffect(() => {
    if (!activeId || !busy) return
    const timer = window.setInterval(async () => {
      if (pollingRef.current) return
      pollingRef.current = true
      try {
        const updated = await fetchJob(activeId)
        setJob((current) => current?.id === activeId ? updated : current)
        const artifactResponse = await fetchArtifacts(activeId)
        setArtifacts((current) => JSON.stringify(current) === JSON.stringify(artifactResponse.artifacts)
          ? current : artifactResponse.artifacts)
      } catch {
      } finally {
        pollingRef.current = false
      }
    }, 2200)
    return () => window.clearInterval(timer)
  }, [activeId, busy])

  function updateConfig<K extends keyof RegistrationConfig>(key: K, value: RegistrationConfig[K]) {
    setConfig((current) => ({ ...current, [key]: value }))
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    setSubmitError('')
    const sidecarCount = config.sourceSidecars.length + config.referenceSidecars.length
    if (sidecarCount > 16) {
      setSubmitError('A run can include at most 16 related sidecar files.')
      return
    }
    const uploadBundleBytes = [config.source, config.reference, ...config.sourceSidecars, ...config.referenceSidecars]
      .reduce((total, file) => total + (file?.size ?? 0), 0)
    if (uploadBundleBytes > 20 * 1024 ** 3) {
      setSubmitError('The combined upload is over the 20 GiB local job limit.')
      return
    }
    setUploadProgress(0)
    setSubmitting(true)
    try {
      const created = await createJob(config, setUploadProgress)
      setJob(created)
      setArtifacts([])
      window.sessionStorage.setItem('chandrashakti.activeJob', created.id)
      setActiveTab('stage_1')
    } catch (error) {
      setSubmitError((error as Error).message)
    } finally {
      setSubmitting(false)
    }
  }

  function handleBatchUploaded(data: {
    sourceFile?: File
    sourceSidecars: File[]
    referenceFile?: File
    referenceSidecars: File[]
    detectedSensors?: { src?: string; ref?: string }
    message: string
  }) {
    if (data.sourceFile) updateConfig('source', data.sourceFile)
    if (data.sourceSidecars.length) updateConfig('sourceSidecars', data.sourceSidecars)
    if (data.referenceFile) updateConfig('reference', data.referenceFile)
    if (data.referenceSidecars.length) updateConfig('referenceSidecars', data.referenceSidecars)
    if (data.detectedSensors?.src) updateConfig('sensorSrc', data.detectedSensors.src)
    if (data.detectedSensors?.ref) updateConfig('sensorRef', data.detectedSensors.ref)

    setBatchFeedback(data.message)
    setTimeout(() => setBatchFeedback(null), 7000)
  }

  function handleNewProjectConfirm(projectName: string) {
    setJob(null)
    setArtifacts([])
    setConfig({ ...initialConfig, projectName: projectName || undefined })
    window.sessionStorage.removeItem('chandrashakti.activeJob')
    window.localStorage.removeItem('chandrashakti.activeJob')
    setActiveTab('input')
  }

  async function handleSelectProject(selectedJob: RegistrationJob) {
    try {
      const fullJob = await fetchJob(selectedJob.id)
      setJob(fullJob)
      window.sessionStorage.setItem('chandrashakti.activeJob', fullJob.id)
      void refreshArtifacts(fullJob.id)
      if (fullJob.status === 'SUCCESS') {
        setActiveTab('output')
      } else if (fullJob.status === 'PROCESSING' || fullJob.status === 'PENDING') {
        setActiveTab('stage_1')
      } else {
        setActiveTab('output')
      }
    } catch {
      setJob(selectedJob)
      window.sessionStorage.setItem('chandrashakti.activeJob', selectedJob.id)
      void refreshArtifacts(selectedJob.id)
      setActiveTab(selectedJob.status === 'SUCCESS' ? 'output' : 'stage_1')
    }
  }

  const stageProgress = job?.progress_pct ?? 0
  const uploadBundleBytes = [config.source, config.reference, ...config.sourceSidecars, ...config.referenceSidecars]
    .reduce((total, file) => total + (file?.size ?? 0), 0)
  const sidecarCount = config.sourceSidecars.length + config.referenceSidecars.length
  const uploadLimitExceeded = uploadBundleBytes > 20 * 1024 ** 3

  const metricSummary = metrics ? [
    { label: 'Residual RMSE', value: formatMetric(metrics.rmse_px), unit: 'px', accent: true },
    { label: 'RMSE In Meters', value: formatMetric(metrics.rmse_meters), unit: 'm' },
    { label: 'Inlier matches', value: formatMetric(metrics.inlier_match_count) },
    { label: 'Inlier ratio', value: typeof metrics.inlier_ratio === 'number' ? `${(metrics.inlier_ratio * 100).toFixed(1)}%` : '—' },
    { label: 'Spatial coverage', value: formatMetric(metrics.convex_hull_coverage_pct), unit: '%' },
    { label: 'Spatial entropy', value: formatMetric(metrics.spatial_entropy_h) },
  ] : []

  function getStageState(stageId: string): 'waiting' | 'running' | 'complete' | 'failed' {
    if (!job) return 'waiting'
    const ev = stageEvents.get(stageId)
    if (ev) {
      if (ev.state === 'complete') return 'complete'
      if (ev.state === 'failed') return 'failed'
      if (ev.state === 'running') return 'running'
    }
    if (job.status === 'SUCCESS') return 'complete'
    return 'waiting'
  }

  function getOutputState(): 'waiting' | 'running' | 'complete' | 'failed' {
    if (!job) return 'waiting'
    if (job.status === 'SUCCESS') return 'complete'
    if (job.status === 'FAILED') return 'failed'
    if (job.status === 'PROCESSING' || job.status === 'PENDING') return 'running'
    return 'waiting'
  }
  const outputState = getOutputState()
  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="topbar-left">
          <a className="brand" href="#top" onClick={(e) => { e.preventDefault(); setActiveTab('input') }}>
            <span className="brand-mark">
              <img src="/logo.png" alt="ChandraShakti Emblem" className="size-full object-contain p-0.5 rounded-lg" />
            </span>
            <div className="brand-copy">
              <strong>CHANDRASHAKTI</strong>
              <small>LUNAR SUB-PIXEL CO-REGISTRATION</small>
            </div>
          </a>
        </div>

        <div className="topbar-center">
          {job ? (
            <Button type="button" variant="outline" size="sm" className="current-run-pill bg-primary/5" onClick={() => setActiveTab(job.status === 'SUCCESS' ? 'output' : 'stage_1')}>
              <span className={`run-status-dot ${job.status.toLowerCase()}`} />
              <span className="run-names">{job.project_name || `${job.source_filename} → ${job.reference_filename}`}</span>
              <span className="run-pct">{stageProgress}%</span>
            </Button>
          ) : config.projectName ? (
            <div className="current-project-pill">
              <Mark name="folderPlus" size={13} />
              <span className="truncate max-w-[200px]">Draft: {config.projectName}</span>
            </div>
          ) : null}
        </div>

        <div className="topbar-right">
          <Button
            type="button"
            variant="outline"
            size="sm"
            className="project-action-btn"
            onClick={() => setIsNewProjectOpen(true)}
            title="Start a new registration project"
          >
            <Mark name="folderPlus" size={14} />
            <span>New Project</span>
          </Button>
          <Button
            type="button"
            variant="outline"
            size="sm"
            className="project-action-btn"
            onClick={() => setIsProjectsOpen(true)}
            title="Open saved projects and history"
          >
            <Mark name="folderOpen" size={14} />
            <span>Projects</span>
          </Button>
          <Button
            type="button"
            variant="outline"
            size="sm"
            className="project-action-btn size-8 p-0 justify-center"
            onClick={() => setTheme((t) => (t === 'dark' ? 'light' : 'dark'))}
            title={theme === 'dark' ? 'Switch to Light Mode' : 'Switch to Dark Mode'}
            aria-label="Toggle Dark / Light Theme"
          >
            {theme === 'dark' ? <Mark name="sun" size={14} /> : <Mark name="moon" size={14} />}
          </Button>
        </div>
      </header>

      <div className="layout-body">
        <AppSidebar
          activeTab={activeTab}
          onTabChange={setActiveTab}
          stages={STAGES}
          getStageState={getStageState}
          outputState={outputState}
          productsReady={Boolean(config.source && config.reference)}
          progress={stageProgress}
          currentStage={job?.current_stage ?? 'Ready for run'}
        />

        <main className="content-pane px-4 py-6 sm:px-6 lg:px-8 lg:py-8 xl:px-10">
          {activeTab === 'input' && (
            <section className="view-panel input-view">
              <div className="view-header">
                <div>
                  <span className="eyebrow"><span className="eyebrow-line" /> STEP 00 · INITIAL SETUP</span>
                  <h2>Select Source & Reference Products</h2>
                  <p>Upload raw or map-projected lunar datasets. The pipeline automatically applies optimal ISRO/LRO calibration, coarse alignment, and sub-pixel elastic deformation using scientific backend defaults.</p>
                </div>
              </div>

              {job && (
                <Card className="active-run-alert">
                  <Mark name="activity" size={16} />
                  <div className="active-run-alert-copy">
                    <strong>Active Run: {job.project_name || `${job.source_filename} → ${job.reference_filename}`}</strong>
                    <span>Status: {job.current_stage || job.status} ({stageProgress}%)</span>
                  </div>
                  <Button type="button" size="sm" onClick={() => setActiveTab(job.status === 'SUCCESS' ? 'output' : 'stage_1')}>
                    View Execution
                  </Button>
                </Card>
              )}

              <form className="products-form" onSubmit={handleSubmit}>
                <div className="project-banner-card">
                  <div className="project-banner-header">
                    <span className="project-banner-label">
                      <Mark name="folderPlus" size={14} />
                      <span>Project Name</span>
                    </span>
                    <span className="project-banner-hint">Optional · Used for history identification</span>
                  </div>
                  <input
                    type="text"
                    className="project-name-input"
                    value={config.projectName || ''}
                    onChange={(e) => updateConfig('projectName', e.target.value)}
                    placeholder="e.g. Shackleton Rim High-Res Alignment Run 01"
                    maxLength={128}
                  />
                </div>

                {batchFeedback && (
                  <div className="flex items-center justify-between gap-3 p-3.5 rounded-xl bg-emerald-500/15 border border-emerald-500/30 text-emerald-400 text-xs">
                    <div className="flex items-center gap-2 min-w-0">
                      <Mark name="check" size={16} />
                      <span className="font-medium truncate">{batchFeedback}</span>
                    </div>
                    <button
                      type="button"
                      onClick={() => setBatchFeedback(null)}
                      className="text-emerald-400 hover:text-emerald-300 shrink-0 cursor-pointer p-0.5"
                      aria-label="Dismiss message"
                    >
                      <Mark name="x" size={14} />
                    </button>
                  </div>
                )}

                <QuickBatchUploader onBatchUploaded={handleBatchUploaded} />

                <div className="products-grid">
                  <ProductDrop
                    title="Source product"
                    hint="GeoTIFF · PDS3 / PDS4 XML+IMG · HDF5"
                    file={config.source}
                    sidecars={config.sourceSidecars}
                    onFile={(file) => updateConfig('source', file)}
                    onSidecars={(files) => updateConfig('sourceSidecars', files)}
                    sensor={config.sensorSrc}
                    onSensor={(value) => updateConfig('sensorSrc', value)}
                    sensors={SOURCE_SENSORS}
                  />

                  <ProductDrop
                    title="Reference product"
                    hint="GeoTIFF · LRO NAC IMG · WAC GeoTIFF"
                    file={config.reference}
                    sidecars={config.referenceSidecars}
                    onFile={(file) => updateConfig('reference', file)}
                    onSidecars={(files) => updateConfig('referenceSidecars', files)}
                    sensor={config.sensorRef}
                    onSensor={(value) => updateConfig('sensorRef', value)}
                    sensors={REFERENCE_SENSORS}
                  />
                </div>

                <Card className="form-submit-box">
                  {submitError && (
                    <div className="form-error">
                      <Mark name="alert" size={16} />
                      <span>{submitError}</span>
                    </div>
                  )}

                  {submitting && (
                    <div className="upload-progress">
                      <div><span>Uploading products to registration engine</span><b>{uploadProgress}%</b></div>
                      <Progress value={uploadProgress} className="progress-track" aria-label={`Upload ${uploadProgress}% complete`} />
                      <small>The registration pipeline starts automatically upon upload completion.</small>
                    </div>
                  )}

                  <div className="submit-action-row">
                    <div className="upload-info">
                      <span>UPLOAD BUNDLE</span>
                      <strong>{formatBytes(uploadBundleBytes)}</strong>
                    </div>

                    <Button
                      className="run-btn-primary"
                      type="submit"
                      size="lg"
                      disabled={submitting || !config.source || !config.reference || health?.status !== 'online' || uploadLimitExceeded || sidecarCount > 16}
                      title={health?.status !== 'online' ? 'The registration backend is offline.' : undefined}
                    >
                      {submitting ? (
                        <><span className="button-spinner" /> Uploading & Enqueuing...</>
                      ) : (
                        <><Mark name="spark" size={18} /> Start registration run <Mark name="arrow" size={16} /></>
                      )}
                    </Button>
                  </div>
                </Card>
              </form>
            </section>
          )}

          {activeTab.startsWith('stage_') && (() => {
            const currentStage = STAGES.find(s => s.id === activeTab)!
            const event = stageEvents.get(currentStage.id)
            const state = getStageState(currentStage.id)
            const stageArtifacts = artifacts.filter(a => a.stage === currentStage.id)
            const stagePreviews = stageArtifacts.filter(a => a.previewable && a.preview_url)
            const stageIndex = STAGES.findIndex(s => s.id === activeTab)
            const prevStage = stageIndex > 0 ? STAGES[stageIndex - 1] : null
            const nextStage = stageIndex < STAGES.length - 1 ? STAGES[stageIndex + 1] : null

            return (
              <section className="view-panel stage-view">
                <div className="view-header">
                  <div>
                    <span className="eyebrow"><span className="eyebrow-line" /> STAGE {currentStage.index} OF 05</span>
                    <h2>{currentStage.title}</h2>
                    <p>{currentStage.description}</p>
                  </div>
                  <div className="view-header-right">
                    <Badge variant={state === 'complete' ? 'success' : state === 'failed' ? 'destructive' : state === 'running' ? 'warning' : 'muted'} className={`stage-state-pill state-${state}`}>
                      <i /> {state === 'running' ? 'IN PROGRESS' : state.toUpperCase()}
                    </Badge>
                  </div>
                </div>

                {event ? (
                  <Card className={`stage-log-card ${state === 'failed' ? 'is-error' : ''}`}>
                    <div className="log-top">
                      <div className="log-title">
                        <span className={`log-bullet state-${state}`} />
                        <strong>{state === 'complete' ? 'Stage Execution Succeeded' : state === 'failed' ? 'Stage Execution Failed' : 'Active Stage Process'}</strong>
                      </div>
                    </div>
                    <p className="log-msg">{event.message}</p>
                    {event.details && <StageDetails details={event.details} />}
                  </Card>
                ) : (
                  <SpaceWaitingCard
                    stage={currentStage}
                    jobActive={busy}
                    prevStageName={prevStage?.shortName}
                    stageIndex={stageIndex}
                    onGoToInput={() => setActiveTab('input')}
                  />
                )}

                {stagePreviews.length > 0 && (
                  <div className="stage-section">
                    <h3 className="section-title">
                      {currentStage.id === 'stage_3'
                        ? `Stage 3 Match Visualizations (${stagePreviews.length})`
                        : `Visual Previews & Quicklooks (${stagePreviews.length})`}
                    </h3>
                    <div className="stage-previews-grid">
                      {stagePreviews.map((art, idx) => (
                        <Card
                          className="preview-card"
                          key={art.artifact_id}
                          onClick={() => setLightboxState({ list: stagePreviews, index: idx })}
                        >
                          <div className="preview-card-img-wrap" title="Click to view full screen in this tab">
                            <img src={apiUrl(art.preview_url!)} alt={formatPreviewTitle(art.file_name)} loading="lazy" />
                            <div className="preview-hover-zoom">
                              <Mark name="spark" size={14} /> Fullscreen
                            </div>
                          </div>
                          <div className="preview-card-caption">
                            <span title={art.file_name}>{formatPreviewTitle(art.file_name)}</span>
                          </div>
                        </Card>
                      ))}
                    </div>
                  </div>
                )}

                {stageArtifacts.length > 0 && (
                  <div className="stage-section">
                    <Card asChild className="stage-artifacts-card"><details key={currentStage.id} className="stage-artifacts-dropdown">
                      <summary className="stage-artifacts-summary">
                        <div className="summary-left">
                          <span className="summary-icon"><Mark name="download" size={16} /></span>
                          <span className="summary-title">Stage Artifacts & Exports</span>
                          <span className="artifacts-count-badge">{stageArtifacts.length} files</span>
                        </div>
                        <div className="summary-right">
                          <span className="summary-hint">Show files</span>
                          <span className="summary-chevron"><Mark name="chevron" size={14} /></span>
                        </div>
                      </summary>
                      <div className="stage-files-list">
                        {stageArtifacts.map((art) => (
                          <ArtifactRow key={art.artifact_id} artifact={art} />
                        ))}
                      </div>
                    </details></Card>
                  </div>
                )}

                <div className="stage-nav-footer">
                  {prevStage ? (
                    <Button type="button" variant="outline" className="stage-nav-btn prev h-auto min-h-10 whitespace-normal py-2 text-left" onClick={() => setActiveTab(prevStage.id as ActiveTab)}>
                      ← Previous: {prevStage.shortName}
                    </Button>
                  ) : (
                    <Button type="button" variant="outline" className="stage-nav-btn prev h-auto min-h-10 whitespace-normal py-2 text-left" onClick={() => setActiveTab('input')}>
                      ← Back to Input
                    </Button>
                  )}
                  {nextStage ? (
                    <Button type="button" variant="outline" className="stage-nav-btn next h-auto min-h-10 whitespace-normal py-2 text-left" onClick={() => setActiveTab(nextStage.id as ActiveTab)}>
                      Next: {nextStage.shortName} →
                    </Button>
                  ) : (
                    <Button type="button" variant="outline" className="stage-nav-btn next h-auto min-h-10 whitespace-normal py-2 text-left" onClick={() => setActiveTab('output')}>
                      Final Output & Results →
                    </Button>
                  )}
                </div>
              </section>
            )
          })()}
          {activeTab === 'output' && (
            <section className="view-panel output-view">
              {!job ? (
                <Card className="no-results-card">
                  <div className="empty-orbit-icon"><Mark name="moon" size={36} /></div>
                  <h3>No Registration Run Available</h3>
                  <p>Please setup source and reference products on the Input tab and run the co-registration engine.</p>
                  <Button type="button" className="run-btn-primary" size="lg" onClick={() => setActiveTab('input')}>
                    Go to Input Setup
                  </Button>
                </Card>
              ) : (
                <>
                  <Card className="results-hero">
                    <div className="hero-top">
                      <div>
                        <span className="eyebrow"><span className="eyebrow-line" /> FINAL CO-REGISTRATION OUTPUT</span>
                        <h2>{job.project_name ? `${job.project_name} (${job.source_filename} → ${job.reference_filename})` : `${job.source_filename} → ${job.reference_filename}`}</h2>
                        <div className="hero-meta">
                          <span><b>{job.sensor_src}</b> Source</span> ·
                          <span><b>{job.sensor_ref}</b> Reference</span> ·
                          <span><Mark name="clock" size={13} /> {job.wall_time_seconds ? `${job.wall_time_seconds.toFixed(1)} seconds` : formatTime(job.completed_at || job.created_at)}</span>
                        </div>
                        <div className="flex items-center gap-2 mt-2 pt-2 border-t border-border/40 text-xs flex-wrap">
                          <span className="font-mono text-[11px] text-muted-foreground bg-muted/80 px-2 py-0.5 rounded border border-border/60 select-all">
                            Project ID: <b className="text-foreground">{job.id}</b>
                          </span>
                          <button
                            type="button"
                            className="inline-flex items-center gap-1 text-[11px] text-primary hover:underline px-1.5 py-0.5 cursor-pointer"
                            onClick={() => void navigator.clipboard.writeText(job.id)}
                            title="Copy Project ID UUID"
                          >
                            <Mark name="check" size={12} />
                            <span>Copy ID</span>
                          </button>
                          <button
                            type="button"
                            className="inline-flex items-center gap-1 text-[11px] text-muted-foreground hover:text-foreground px-1.5 py-0.5 cursor-pointer"
                            onClick={() => void navigator.clipboard.writeText(`C:\\Padhai\\E\\SIH1\\data\\storage\\outputs\\${job.id}`)}
                            title="Copy Windows Folder Path to this project's files"
                          >
                            <span>Copy Folder Path</span>
                          </button>
                        </div>
                      </div>
                      <div className="hero-verdict-box">
                        <Badge variant={String(metrics?.verdict ?? '').toLowerCase().includes('verified') ? 'success' : 'warning'} className={`verdict-pill ${String(metrics?.verdict ?? '').toLowerCase().includes('verified') ? 'verified' : 'uncertain'}`}>
                          <i /> {String(metrics?.verdict ?? job.status).replaceAll('_', ' ')}
                        </Badge>
                      </div>
                    </div>

                    {job.status === 'FAILED' && (
                      <div className="run-error">
                        <Mark name="alert" size={18} />
                        <div>
                          <strong>{failedStage?.title ?? 'Pipeline Execution Stopped'}</strong>
                          <p>{job.error_message ?? failedStage?.message ?? 'Registration could not complete.'}</p>
                        </div>
                      </div>
                    )}
                  </Card>

                  {metrics && (
                    <div className="results-section">
                      <div className="results-section-header">
                        <h3 className="section-title">Scientific Precision Metrics</h3>
                        {metrics.metrics_are_measured === false && (
                          <Badge variant="warning" className="measurement-warning-pill">
                            <Mark name="alert" size={13} /> Estimated residuals
                          </Badge>
                        )}
                      </div>

                      <div className="metrics-grid">
                        {metricSummary.map((item) => (
                          <MetricCard key={item.label} label={item.label} value={item.value} unit={item.unit} accent={item.accent} />
                        ))}
                      </div>

                      <details className="all-metrics-accordion">
                        <summary><span>View All Scientific Metric Attributes</span><Mark name="chevron" size={14} /></summary>
                        <div className="metric-table">
                          {flattenMetrics(metrics).map(([label, value]) => (
                            <div className="metric-table-row" key={label}>
                              <span>{label}</span>
                              <strong>{value}</strong>
                            </div>
                          ))}
                        </div>
                      </details>
                    </div>
                  )}

                  {(() => {
                    const outputPreviews = artifacts.filter(a => a.previewable && a.preview_url)
                    if (outputPreviews.length === 0) return null
                    return (
                      <div className="results-section">
                        <h3 className="section-title">Co-Registration Quicklooks & Visual Verification ({outputPreviews.length})</h3>
                        <div className="results-previews-grid">
                          {outputPreviews.map((art, idx) => (
                            <Card
                              className="result-preview-card"
                              key={art.artifact_id}
                              onClick={() => setLightboxState({ list: outputPreviews, index: idx })}
                            >
                              <div className="preview-card-img-wrap" title="Click to view full screen in this tab">
                                <img src={apiUrl(art.preview_url!)} alt={formatPreviewTitle(art.file_name)} loading="lazy" />
                                <div className="preview-hover-zoom">
                                  <Mark name="spark" size={14} /> Fullscreen
                                </div>
                              </div>
                              <div className="result-preview-caption">
                                <strong title={art.file_name}>{formatPreviewTitle(art.file_name)}</strong>
                              </div>
                            </Card>
                          ))}
                        </div>
                      </div>
                    )
                  })()}

                  <div className="results-section">
                    <div className="results-section-header">
                      <h3 className="section-title">Generated Data Products & Scientific Artifacts ({artifacts.length})</h3>
                      <span className="section-subtitle">GeoTIFFs · CSV Tie Points · Transform Models · Metrics JSON</span>
                    </div>

                    {artifacts.length > 0 ? (
                      <div className="all-files-list">
                        {artifacts.map((artifact) => (
                          <ArtifactRow key={artifact.artifact_id} artifact={artifact} />
                        ))}
                      </div>
                    ) : (
                      <p className="files-empty">Outputs will appear here upon completion.</p>
                    )}
                  </div>
                </>
              )}
            </section>
          )}
        </main>
      </div>

      {lightboxState && (
        <ImageLightbox
          list={lightboxState.list}
          index={lightboxState.index}
          onClose={() => setLightboxState(null)}
          onNavigate={(newIndex) => setLightboxState({ list: lightboxState.list, index: newIndex })}
        />
      )}

      <NewProjectModal
        isOpen={isNewProjectOpen}
        onClose={() => setIsNewProjectOpen(false)}
        onConfirm={handleNewProjectConfirm}
        currentJob={job}
      />

      <ProjectsModal
        isOpen={isProjectsOpen}
        onClose={() => setIsProjectsOpen(false)}
        onSelectProject={handleSelectProject}
        currentJobId={job?.id}
      />
    </div>
  )
}

function flattenMetrics(metrics: Record<string, unknown>): Array<[string, string]> {
  const result: Array<[string, string]> = []
  const visit = (value: unknown, prefix: string) => {
    if (value === null || value === undefined) return
    if (Array.isArray(value)) {
      if (value.length && value.every((item) => ['string', 'number', 'boolean'].includes(typeof item))) {
        result.push([prefix, value.map((item) => formatMetric(item)).join(', ')])
      }
      return
    }
    if (typeof value === 'object') {
      for (const [key, child] of Object.entries(value as Record<string, unknown>)) {
        visit(child, prefix ? `${prefix} · ${key.replaceAll('_', ' ')}` : key.replaceAll('_', ' '))
      }
      return
    }
    result.push([prefix.replaceAll('_', ' '), formatMetric(value)])
  }
  for (const [key, value] of Object.entries(metrics)) visit(value, key)
  return result
}

export default App
