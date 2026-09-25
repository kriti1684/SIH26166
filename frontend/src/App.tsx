import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { FormEvent } from 'react'
import { createJob, fetchArtifacts, fetchHealth, fetchJob, fetchJobs } from './api'
import {
  ArtifactRow, EmptyStageList, formatBytes, formatMetric, formatTime, Mark, MetricCard,
  ProductDrop, StageCard,
} from './components'
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
const STAGES = [
  {
    id: 'stage_1', index: '01', title: 'Ingest and harmonize',
    description: 'Read sensor labels, georeference both rasters, and build a shared overlap grid.',
    input: 'Source product + reference product + sidecars', output: 'Georeferenced overlap rasters + scale metadata + quicklooks',
  },
  {
    id: 'stage_2', index: '02', title: 'Estimate coarse alignment',
    description: 'Estimate the global pixel offset and strip-wise drift from image structure.',
    input: 'Harmonized source and reference rasters', output: 'Global shift + confidence + drift profile',
  },
  {
    id: 'stage_3', index: '03', title: 'Find ties and fit transform',
    description: 'Match spatial tiles, refine tie points, reject outliers, and fit the transform.',
    input: 'Overlap rasters + coarse shift and drift', output: 'Candidate / refined tie points + fitted transform',
  },
  {
    id: 'stage_4', index: '04', title: 'Warp registered product',
    description: 'Apply the transform in blocks and write a registered raster on the reference grid.',
    input: 'Source raster + hybrid transformation + reference grid', output: 'Registered GeoTIFF + compact preview + overlay',
  },
  {
    id: 'stage_5', index: '05', title: 'Verify and summarize',
    description: 'Measure residuals and spatial coverage, then generate comparison previews and metrics.',
    input: 'Registered product + reference + tie-point evidence', output: 'Verdict + metric report + verification imagery',
  },
]

const initialConfig: RegistrationConfig = {
  source: null, reference: null, sourceSidecars: [], referenceSidecars: [],
  sensorSrc: 'OHRC', sensorRef: 'NAC', method: 'loftr', structuralMethod: 'phase_congruency', coarseMethod: 'auto',
  polyDegree: 2, tpsSmoothing: 0.05, ransacThreshold: 1.2, warpOrder: 3, wacBand: 7, gridRows: 4, gridCols: 4, exportNative: false,
}

function App() {
  const [config, setConfig] = useState<RegistrationConfig>(initialConfig)
  const [job, setJob] = useState<RegistrationJob | null>(null)
  const [artifacts, setArtifacts] = useState<ArtifactInfo[]>([])
  const [health, setHealth] = useState<HealthStatus | null>(null)
  const [submitError, setSubmitError] = useState('')
  const [uploadProgress, setUploadProgress] = useState(0)
  const [submitting, setSubmitting] = useState(false)
  const [expandedStage, setExpandedStage] = useState('stage_1')
  const pollingRef = useRef(false)

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
    } catch { /* artifact requests retry when the active run is polled */ }
  }, [])

  useEffect(() => {
    let mounted = true
    const storedId = window.localStorage.getItem('chandashakti.activeJob')
    void Promise.allSettled([fetchHealth(), fetchJobs()]).then(async ([healthResult, jobsResult]) => {
      if (!mounted) return
      if (healthResult.status === 'fulfilled') setHealth(healthResult.value)
      else setHealth({ status: 'offline' })
      if (jobsResult.status === 'fulfilled') {
        const resumeId = storedId || jobsResult.value.find((item) => item.status === 'PROCESSING' || item.status === 'PENDING')?.id
        if (resumeId) {
          try {
            const restored = await fetchJob(resumeId)
            if (mounted) {
              setJob(restored)
              void refreshArtifacts(resumeId)
            }
          } catch {
            window.localStorage.removeItem('chandashakti.activeJob')
          }
        }
      }
    })
    const healthInterval = window.setInterval(() => {
      void fetchHealth().then((value) => { if (mounted) setHealth(value) }).catch(() => {
        if (mounted) setHealth({ status: 'offline' })
      })
    }, 12000)
    return () => { mounted = false; window.clearInterval(healthInterval) }
  }, [refreshArtifacts])

  useEffect(() => {
    if (!activeId) {
      setArtifacts([])
      return
    }
    window.localStorage.setItem('chandashakti.activeJob', activeId)
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
        // The health indicator reports connectivity; the next poll retries this run.
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
    if (sidecarCount > 16) {
      setSubmitError('A run can include at most 16 related sidecar files.')
      return
    }
    if (uploadLimitExceeded) {
      setSubmitError('The combined upload is over the 20 GiB local job limit.')
      return
    }
    setUploadProgress(0)
    setSubmitting(true)
    try {
      const created = await createJob(config, setUploadProgress)
      setJob(created)
      setArtifacts([])
      setExpandedStage('stage_1')
      window.localStorage.setItem('chandashakti.activeJob', created.id)
    } catch (error) {
      setSubmitError((error as Error).message)
    } finally {
      setSubmitting(false)
    }
  }

  const stageProgress = job?.progress_pct ?? 0
  const uploadBundleBytes = [config.source, config.reference, ...config.sourceSidecars, ...config.referenceSidecars]
    .reduce((total, file) => total + (file?.size ?? 0), 0)
  const sidecarCount = config.sourceSidecars.length + config.referenceSidecars.length
  const uploadLimitExceeded = uploadBundleBytes > 20 * 1024 ** 3
  const metricSummary = metrics ? [
    { label: 'Residual RMSE', value: formatMetric(metrics.rmse_px), unit: 'px', accent: true },
    { label: 'Inlier matches', value: formatMetric(metrics.inlier_match_count) },
    { label: 'Inlier ratio', value: typeof metrics.inlier_ratio === 'number' ? `${(metrics.inlier_ratio * 100).toFixed(1)}%` : '—' },
    { label: 'Spatial coverage', value: formatMetric(metrics.convex_hull_coverage_pct), unit: '%' },
    { label: 'Spatial entropy', value: formatMetric(metrics.spatial_entropy_h) },
    { label: 'Scientific confidence', value: typeof metrics.composite_scientific_confidence === 'number' ? `${(metrics.composite_scientific_confidence * 100).toFixed(1)}%` : '—' },
  ] : []

  return <div className="app-shell">
    <header className="topbar">
      <a className="brand" href="#top" aria-label="ChandaShakti registration lab home">
        <span className="brand-mark"><Mark name="moon" size={20} /></span>
        <span><strong>CHANDASHAKTI</strong></span>
      </a>
      <h1 className="topbar-title">Lunar co-registration</h1>
      <div className="topbar-right">
        <div className={`api-indicator ${health?.status === 'online' ? 'online' : ''}`} title={`Registration backend ${health?.status ?? 'unavailable'}`}>
          <span className="api-dot" />{health?.status === 'online' ? 'Connected' : 'Offline'}
        </div>
      </div>
    </header>

    <main id="top" className="workspace">
      <section className="panel setup-panel">
        <form className="new-run-form" onSubmit={handleSubmit}>
          <div className="new-run-products">
              <ProductDrop
                title="Source product" hint="GeoTIFF · PDS3 / PDS4 · HDF5"
                file={config.source} sidecars={config.sourceSidecars} onFile={(file) => updateConfig('source', file)}
                onSidecars={(files) => updateConfig('sourceSidecars', files)} sensor={config.sensorSrc}
                onSensor={(value) => updateConfig('sensorSrc', value)} sensors={SOURCE_SENSORS}
              />
              <ProductDrop
                title="Reference product" hint="GeoTIFF · PDS3 / PDS4 · HDF5"
                file={config.reference} sidecars={config.referenceSidecars} onFile={(file) => updateConfig('reference', file)}
                onSidecars={(files) => updateConfig('referenceSidecars', files)} sensor={config.sensorRef}
                onSensor={(value) => updateConfig('sensorRef', value)} sensors={REFERENCE_SENSORS}
              />
          </div>

          <div className="run-controls">
            <div className="advanced-fields">
              <div className="field-grid">
                <label><span>Feature matching</span><select value={config.method} onChange={(e) => updateConfig('method', e.target.value)}><option value="loftr">LoFTR</option><option value="ensemble">Ensemble</option><option value="crater">Crater matching</option></select></label>
                <label><span>Coarse solver</span><select value={config.coarseMethod} onChange={(e) => updateConfig('coarseMethod', e.target.value)}><option value="auto">Automatic</option><option value="fft">FFT phase correlation</option><option value="crater">Crater voting</option></select></label>
                <label><span>Structural representation</span><select value={config.structuralMethod} onChange={(e) => updateConfig('structuralMethod', e.target.value)}><option value="phase_congruency">Phase congruency</option><option value="gradient">Gradient</option></select></label>
                <label><span>Matching grid</span><div className="inline-number-pair"><input type="number" min="1" max="16" value={config.gridRows} onChange={(e) => updateConfig('gridRows', Number(e.target.value))} aria-label="Grid rows"/><i>×</i><input type="number" min="1" max="16" value={config.gridCols} onChange={(e) => updateConfig('gridCols', Number(e.target.value))} aria-label="Grid columns"/></div></label>
                {config.sensorRef === 'WAC' && <label><span>WAC spectral band</span><select value={config.wacBand} onChange={(e) => updateConfig('wacBand', Number(e.target.value))}><option value="7">Band 7 · 689 nm red</option><option value="4">Band 4 · 566 nm green</option><option value="3">Band 3 · 415 nm blue</option><option value="1">Band 1</option><option value="2">Band 2</option><option value="5">Band 5</option><option value="6">Band 6</option></select></label>}
                <label><span>RANSAC threshold <small>PIXELS</small></span><input type="number" min="0.1" max="100" step="0.1" value={config.ransacThreshold} onChange={(e) => updateConfig('ransacThreshold', Number(e.target.value))} /></label>
                <label><span>TPS smoothing</span><input type="number" min="0" max="10" step="0.01" value={config.tpsSmoothing} onChange={(e) => updateConfig('tpsSmoothing', Number(e.target.value))} /></label>
                <label><span>Transform degree</span><input type="number" min="0" max="4" value={config.polyDegree} onChange={(e) => updateConfig('polyDegree', Number(e.target.value))} /></label>
                <label><span>Warp interpolation order</span><select value={config.warpOrder} onChange={(e) => updateConfig('warpOrder', Number(e.target.value))}><option value="3">3 · Bicubic</option><option value="1">1 · Bilinear</option><option value="0">0 · Nearest</option><option value="2">2 · Quadratic</option><option value="4">4 · Quartic</option><option value="5">5 · Quintic</option></select></label>
              </div>
              <label className="checkbox-line"><input type="checkbox" checked={config.exportNative} onChange={(e) => updateConfig('exportNative', e.target.checked)} /><span><b>Export secondary native-GSD raster</b><small>Creates a high-resolution GeoTIFF and may require substantial disk space.</small></span></label>
            </div>
            <div className="run-actions">
              {submitError && <div className="form-error"><Mark name="alert" size={16} /><span>{submitError}</span></div>}
              {submitting && <div className="upload-progress"><div><span>Uploading products</span><b>{uploadProgress}%</b></div><span className="progress-track"><i style={{ width: `${Math.max(uploadProgress, 3)}%` }} /></span><small>The full pipeline will be queued after the upload finishes.</small></div>}
              {uploadBundleBytes > 0 && <div className="upload-bundle"><span>UPLOAD BUNDLE</span><strong>{formatBytes(uploadBundleBytes)}</strong></div>}
              <button className="run-button" type="submit" title={health?.status !== 'online' ? 'The registration backend is offline.' : undefined} disabled={submitting || !config.source || !config.reference || health?.status !== 'online' || uploadLimitExceeded || sidecarCount > 16}>
                {submitting ? <><span className="button-spinner" /> Uploading products…</> : <><Mark name="spark" size={17} /> Start registration run <Mark name="arrow" size={16} /></>}
              </button>
            </div>
          </div>
        </form>
      </section>

      <section className="panel results-shell">
          {!job ? <div className="no-job-panel"><EmptyStageList /></div> : <>
            <section className="panel run-overview">
              <div className="run-overview-top">
                <div>
                  <h2>{job.source_filename} <span>→</span> {job.reference_filename}</h2>
                </div>
                <span className={`run-status ${job.status.toLowerCase()}`}><i />{job.status === 'SUCCESS' ? 'PIPELINE COMPLETE' : job.status}</span>
              </div>
              <div className="run-meta">
                <span><b>{job.sensor_src}</b> source</span><i />
                <span><b>{job.sensor_ref}</b> reference</span><i />
                <span><Mark name="clock" size={13} /> {job.wall_time_seconds ? `${job.wall_time_seconds.toFixed(1)} sec` : formatTime(job.created_at)}</span>
                {job.current_stage && <><i /><span className="current-stage-text"><Mark name="activity" size={13} /> {job.current_stage}</span></>}
              </div>
              {(busy || job.status === 'SUCCESS') && <div className="run-progress">
                <div className="progress-track"><i style={{ width: `${stageProgress}%` }} /></div><span>{stageProgress}%</span>
              </div>}
              {job.status === 'FAILED' && <div className="run-error"><Mark name="alert" size={16} /><div><strong>{failedStage?.title ?? 'Pipeline failed'}</strong><p>{job.error_message ?? failedStage?.message ?? 'The worker could not complete this run.'}</p></div></div>}
              {job.status === 'SUCCESS' && <div className="execution-note"><Mark name="check" size={15} /> Run complete</div>}
            </section>

            {metrics && <section className="panel metrics-panel">
              <div className="metrics-heading">
                <h2>Metrics</h2>
                <span className={`verdict-badge ${String(metrics.verdict ?? '').toLowerCase().includes('verified') ? 'verified' : 'uncertain'}`}><i />{String(metrics.verdict ?? 'METRICS READY').replaceAll('_', ' ')}</span>
              </div>
              {metrics.metrics_are_measured === false && <div className="measurement-warning"><Mark name="alert" size={15} /><span>Residuals are estimated; precision is unverified.</span></div>}
              <div className="metric-grid">{metricSummary.map((item) => <MetricCard key={item.label} label={item.label} value={item.value} unit={item.unit} accent={item.accent} />)}</div>
              <details className="all-metrics">
                <summary><span>All metrics</span><span><Mark name="chevron" size={14} /></span></summary>
                <div className="metric-table">{flattenMetrics(metrics).map(([label, value]) => <div className="metric-table-row" key={label}><span>{label}</span><strong>{value}</strong></div>)}</div>
              </details>
            </section>}

            <section className="panel stage-panel">
              <div className="stage-panel-heading"><h2>Pipeline stages</h2><span className={`trace-indicator ${busy ? 'active' : ''}`}><i />{busy ? 'RUNNING' : job.status === 'SUCCESS' ? '5 / 5 COMPLETE' : job.status}</span></div>
              <div className="stage-list">{STAGES.map((stage) => <StageCard
                key={stage.id} stageId={stage.id} index={stage.index} title={stage.title}
                description={stage.description} inputLabel={stage.input} outputLabel={stage.output}
                event={stageEvents.get(stage.id)} artifacts={artifacts} expanded={expandedStage === stage.id}
                onToggle={() => setExpandedStage((current) => current === stage.id ? '' : stage.id)}
              />)}</div>
            </section>

            <section className="panel all-files-panel">
              <div className="stage-panel-heading"><h2>Files</h2><span className="artifact-count">{artifacts.length}</span></div>
              {artifacts.length ? <div className="all-files-list">{artifacts.map((artifact) => <ArtifactRow key={artifact.artifact_id} artifact={artifact} />)}</div>
                : <p className="files-empty">Pipeline outputs will appear here.</p>}
            </section>
          </>}
      </section>
    </main>
  </div>
}

function flattenMetrics(metrics: Record<string, unknown>): Array<[string, string]> {
  const result: Array<[string, string]> = []
  const visit = (value: unknown, prefix: string) => {
    if (value === null || value === undefined) return
    if (Array.isArray(value)) {
      if (value.length && value.every((item) => ['string', 'number', 'boolean'].includes(typeof item))) result.push([prefix, value.map((item) => formatMetric(item)).join(', ')])
      return
    }
    if (typeof value === 'object') {
      for (const [key, child] of Object.entries(value as Record<string, unknown>)) visit(child, prefix ? `${prefix} · ${key.replaceAll('_', ' ')}` : key.replaceAll('_', ' '))
      return
    }
    result.push([prefix.replaceAll('_', ' '), formatMetric(value)])
  }
  for (const [key, value] of Object.entries(metrics)) visit(value, key)
  return result
}

export default App
