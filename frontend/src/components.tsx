import { useId, useRef, useState } from 'react'
import type { ChangeEvent, DragEvent, ReactNode } from 'react'
import type { ArtifactInfo, PipelineStageEvent } from './types'
import { apiUrl } from './api'

export function Mark({ name, size = 18 }: { name: 'moon' | 'upload' | 'plus' | 'x' | 'chevron' | 'download' | 'image' | 'file' | 'activity' | 'clock' | 'arrow' | 'check' | 'alert' | 'gpu' | 'history' | 'spark' | 'layers' | 'external'; size?: number }) {
  const shared = { width: size, height: size, viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor', strokeWidth: 1.7, strokeLinecap: 'round' as const, strokeLinejoin: 'round' as const, 'aria-hidden': true as const }
  const paths: Record<typeof name, ReactNode> = {
    moon: <><path d="M20.2 15.1A8.5 8.5 0 0 1 8.9 3.8 8.7 8.7 0 1 0 20.2 15.1Z"/><path d="m16.5 4 .8 1.6L19 6.4l-1.7.8-.8 1.7-.7-1.7-1.7-.8 1.7-.8.7-1.6Z"/></>,
    upload: <><path d="M12 16V4m0 0L7.5 8.5M12 4l4.5 4.5"/><path d="M5 14v5a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1v-5"/></>,
    plus: <><path d="M12 5v14M5 12h14"/></>,
    x: <><path d="m18 6-12 12M6 6l12 12"/></>,
    chevron: <><path d="m7 10 5 5 5-5"/></>,
    download: <><path d="M12 3v12m0 0 4-4m-4 4-4-4"/><path d="M5 17v3h14v-3"/></>,
    image: <><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><path d="m21 15-5-5L5 21"/></>,
    file: <><path d="M13 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9Z"/><path d="M13 2v7h7M8 13h8M8 17h8"/></>,
    activity: <><path d="M3 12h4l3-8 4 16 3-8h4"/></>,
    clock: <><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/></>,
    arrow: <><path d="M5 12h14m-6-6 6 6-6 6"/></>,
    check: <><path d="m5 12 4 4L19 6"/></>,
    alert: <><path d="M12 9v4m0 4h.01"/><path d="M10.3 3.9 2.5 17.4A1.8 1.8 0 0 0 4 20h16a1.8 1.8 0 0 0 1.5-2.6L13.7 3.9a2 2 0 0 0-3.4 0Z"/></>,
    gpu: <><rect x="5" y="5" width="14" height="14" rx="2"/><path d="M9 9h6v6H9zM9 1v4m6-4v4m-6 14v4m6-4v4M1 9h4m-4 6h4m14-6h4m-4 6h4"/></>,
    history: <><path d="M3 12a9 9 0 1 0 2.6-6.4L3 8"/><path d="M3 3v5h5m4-1v5l3 2"/></>,
    spark: <><path d="m12 3 1.9 5.8L20 11l-6.1 2.2L12 19l-2-5.8L4 11l6-2.2L12 3Z"/><path d="m19 15 .8 2.2L22 18l-2.2.8L19 21l-.8-2.2L16 18l2.2-.8L19 15Z"/></>,
    layers: <><path d="m12 3 9 5-9 5-9-5 9-5Z"/><path d="m3 12 9 5 9-5M3 16l9 5 9-5"/></>,
    external: <><path d="M14 3h7v7m-11 4L21 3"/><path d="M19 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V7a2 2 0 0 1 2-2h6"/></>,
  }
  return <svg {...shared}>{paths[name]}</svg>
}

export function SectionLabel({ children, trailing }: { children: ReactNode; trailing?: ReactNode }) {
  return <div className="section-label"><span>{children}</span>{trailing}</div>
}

export function ProductDrop({
  title, hint, file, sidecars, onFile, onSidecars, sensor, onSensor, sensors,
}: {
  title: string; hint: string; file: File | null; sidecars: File[]; onFile: (file: File | null) => void
  onSidecars: (files: File[]) => void; sensor: string; onSensor: (value: string) => void
  sensors: Array<{ value: string; label: string }>
}) {
  const inputId = useId()
  const extrasId = useId()
  const fileRef = useRef<HTMLInputElement>(null)
  const sidecarRef = useRef<HTMLInputElement>(null)
  const [dragging, setDragging] = useState(false)

  function setFirstFile(list: FileList | null) {
    if (list?.[0]) onFile(list[0])
    if (fileRef.current) fileRef.current.value = ''
  }
  function onDrop(event: DragEvent<HTMLButtonElement>) {
    event.preventDefault(); setDragging(false); setFirstFile(event.dataTransfer.files)
  }
  function onSidecarChange(event: ChangeEvent<HTMLInputElement>) {
    const next = Array.from(event.target.files ?? [])
    if (next.length) onSidecars([...sidecars, ...next])
    event.target.value = ''
  }

  return <div className={`product-block ${dragging ? 'is-dragging' : ''}`}>
    <div className="product-topline">
      <span className="product-title">{title}</span>
    </div>
    <label className="field-label" htmlFor={`${inputId}-sensor`}>Instrument</label>
    <select id={`${inputId}-sensor`} value={sensor} onChange={(e) => onSensor(e.target.value)}>
      {sensors.map((item) => <option key={item.value} value={item.value}>{item.label}</option>)}
    </select>
    <input id={inputId} ref={fileRef} type="file" accept=".xml,.lbl,.img,.tif,.tiff,.h5,.hdf5" onChange={(e) => setFirstFile(e.target.files)} hidden />
    <input id={extrasId} ref={sidecarRef} type="file" accept=".xml,.lbl,.img,.tif,.tiff,.h5,.hdf5,.dat,.bin" multiple onChange={onSidecarChange} hidden />
    {file ? <div className="selected-file-card">
      <div className="file-icon"><Mark name="file" size={17} /></div>
      <div className="selected-file-copy"><strong title={file.name}>{file.name}</strong><span>{formatBytes(file.size)} · {file.name.split('.').pop()?.toUpperCase()} product</span></div>
      <button className="icon-button" type="button" aria-label={`Remove ${title}`} onClick={() => onFile(null)}><Mark name="x" size={16} /></button>
    </div> : <button
      className="dropzone" type="button" onClick={() => fileRef.current?.click()}
      onDragOver={(e) => { e.preventDefault(); setDragging(true) }} onDragLeave={() => setDragging(false)} onDrop={onDrop}
    >
      <span className="drop-icon"><Mark name="upload" size={19} /></span>
      <span className="drop-title">Drop a product here or <b>browse files</b></span>
      <span className="drop-hint">{hint}</span>
    </button>}
    <div className="sidecar-line">
      <span><Mark name="layers" size={14} /> Related files{sidecars.length ? ` · ${sidecars.length}` : ''}</span>
      <button type="button" className="text-button" onClick={() => sidecarRef.current?.click()}><Mark name="plus" size={14} /> Add files</button>
    </div>
    {sidecars.length > 0 && <div className="sidecar-list">{sidecars.map((item, index) => <div className="sidecar-item" key={`${item.name}-${index}`}>
      <span><Mark name="file" size={13} /> {item.name} <small>{formatBytes(item.size)}</small></span>
      <button type="button" className="icon-button compact" aria-label={`Remove ${item.name}`} onClick={() => onSidecars(sidecars.filter((_, i) => i !== index))}><Mark name="x" size={13} /></button>
    </div>)}</div>}
  </div>
}

export function StageCard({
  stageId, index, title, description, inputLabel, outputLabel, event, artifacts, expanded, onToggle,
}: {
  stageId: string; index: string; title: string; description: string; inputLabel: string; outputLabel: string
  event?: PipelineStageEvent; artifacts: ArtifactInfo[]; expanded: boolean; onToggle: () => void
}) {
  const state = event?.state ?? 'waiting'
  const visibleArtifacts = artifacts.filter((artifact) => artifact.stage === stageId)
  const previewArtifacts = visibleArtifacts.filter((artifact) => artifact.previewable && artifact.preview_url)
  const [showAllPreviews, setShowAllPreviews] = useState(false)
  const previews = showAllPreviews ? previewArtifacts : previewArtifacts.slice(0, 4)

  return <article className={`stage-card stage-${state}`}>
    <button className="stage-heading" type="button" onClick={onToggle} aria-expanded={expanded}>
      <span className={`stage-marker ${state}`}>
        {state === 'complete' ? <Mark name="check" size={17} /> : state === 'failed' ? <Mark name="alert" size={16} /> : <span>{index}</span>}
      </span>
      <span className="stage-heading-copy"><strong>{title}</strong></span>
      <span className={`state-chip ${state}`}>{state === 'waiting' ? 'Waiting' : state}</span>
      <span className={`stage-chevron ${expanded ? 'open' : ''}`}><Mark name="chevron" size={16} /></span>
    </button>
    {expanded && <div className="stage-content">
      <p className="stage-description">{description}</p>
      <div className="stage-flow">
        <div><span>INPUT</span><p>{inputLabel}</p></div><Mark name="arrow" size={15} /><div><span>OUTPUT</span><p>{outputLabel}</p></div>
      </div>
      {event && <div className={`stage-message ${state === 'failed' ? 'error' : ''}`}>
        <span className="message-bullet" />
        <div><strong>{state === 'complete' ? 'Stage result' : state === 'failed' ? 'Stage stopped' : 'Currently running'}</strong><p>{event.message}</p></div>
        <time>{formatTime(event.created_at)}</time>
      </div>}
      {event?.details && <StageDetails details={event.details} />}
      {previews.length > 0 && <div className="preview-grid">{previews.map((artifact) => <figure className="artifact-preview" key={artifact.artifact_id}>
        <a href={apiUrl(artifact.preview_url!)} target="_blank" rel="noreferrer" title="Open quicklook full size"><img loading="lazy" src={apiUrl(artifact.preview_url!)} alt={artifact.file_name} /></a>
        <figcaption><span>{artifact.file_name}</span><a href={apiUrl(artifact.download_url)} download title="Download preview"><Mark name="download" size={14} /></a></figcaption>
      </figure>)}</div>}
      {previewArtifacts.length > 4 && <button type="button" className="show-more" onClick={() => setShowAllPreviews((value) => !value)}>{showAllPreviews ? 'Show fewer previews' : `Show ${previewArtifacts.length - 4} more previews`}</button>}
      {visibleArtifacts.length > 0 && <div className="artifact-list">
        <SectionLabel trailing={<span className="artifact-count">{visibleArtifacts.length} files</span>}>Stage files</SectionLabel>
        {visibleArtifacts.map((artifact) => <ArtifactRow key={artifact.artifact_id} artifact={artifact} />)}
      </div>}
      {event && visibleArtifacts.length === 0 && state === 'complete' && <p className="artifact-empty">The stage completed without downloadable side-products.</p>}
    </div>}
  </article>
}

export function ArtifactRow({ artifact }: { artifact: ArtifactInfo }) {
  return <div className="artifact-row">
    <span className={`artifact-type ${artifact.previewable ? 'visual' : ''}`}>{artifact.previewable ? <Mark name="image" size={14} /> : <Mark name="file" size={14} />}</span>
    <div className="artifact-copy"><strong title={artifact.relative_path}>{artifact.file_name}</strong><span>{artifact.relative_path}</span></div>
    <span className="artifact-size">{formatBytes(artifact.size_bytes)}</span>
    <a className="artifact-download" href={apiUrl(artifact.download_url)} download aria-label={`Download ${artifact.file_name}`} title="Download"><Mark name="download" size={16} /></a>
  </div>
}

export function MetricCard({ label, value, unit, accent }: { label: string; value: ReactNode; unit?: string; accent?: boolean }) {
  return <div className={`metric-card ${accent ? 'accent' : ''}`}><span>{label}</span><strong>{value ?? '—'}{unit && value !== undefined && value !== null && <small>{unit}</small>}</strong></div>
}

export function EmptyStageList() {
  return <div className="stage-empty-state">
    <div className="empty-orbit"><span /><i /><b><Mark name="moon" size={24} /></b></div>
    <h2>Ready for a run</h2>
    <p>Select both instruments, then add a source and reference product.</p>
  </div>
}

export function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes < 0) return '—'
  if (bytes < 1024) return `${bytes} B`
  const units = ['KB', 'MB', 'GB', 'TB']
  let value = bytes / 1024; let unit = 0
  while (value >= 1024 && unit < units.length - 1) { value /= 1024; unit += 1 }
  return `${value.toFixed(value >= 100 ? 0 : value >= 10 ? 1 : 2)} ${units[unit]}`
}

export function formatTime(value?: string | null): string {
  if (!value) return ''
  const date = new Date(value)
  return Number.isNaN(date.getTime()) ? '' : date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
}

export function formatMetric(value: unknown, precision = 3): string {
  if (typeof value === 'number') return Number.isInteger(value) ? value.toLocaleString() : value.toFixed(precision)
  if (typeof value === 'boolean') return value ? 'Yes' : 'No'
  if (typeof value === 'string') return value.replaceAll('_', ' ')
  return '—'
}

function StageDetails({ details }: { details: Record<string, unknown> }) {
  const ignored = new Set(['previews', 'preview', 'preview_dimensions', 'files_created', 'metrics'])
  const entries = Object.entries(details).filter(([key, value]) => !ignored.has(key) && value !== null && value !== undefined)
  if (!entries.length) return null
  return <div className="stage-detail-grid">{entries.map(([key, value]) => <div className="stage-detail" key={key}>
    <span>{key.replaceAll('_', ' ')}</span><strong>{displayDetail(value)}</strong>
  </div>)}</div>
}

function displayDetail(value: unknown): ReactNode {
  if (typeof value === 'number') return Number.isInteger(value) ? value.toLocaleString() : value.toFixed(3)
  if (typeof value === 'boolean') return value ? 'Yes' : 'No'
  if (typeof value === 'string') return value
  if (Array.isArray(value)) return value.map((item) => typeof item === 'number' ? item.toLocaleString() : String(item)).join(' × ')
  if (value && typeof value === 'object') {
    return <span className="detail-object">{Object.entries(value as Record<string, unknown>).slice(0, 4).map(([key, item]) => `${key.replaceAll('_', ' ')}: ${displayDetail(item)}`).join(' · ')}</span>
  }
  return String(value)
}
