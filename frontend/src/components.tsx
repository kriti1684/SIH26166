import { useEffect, useId, useRef, useState } from 'react'
import type { ChangeEvent, DragEvent, ReactNode } from 'react'
import { Activity, ArrowRight, Check, ChevronDown, Clock3, Cpu, Download, ExternalLink, FileImage, FileText, History, Layers3, Moon, Plus, Sparkles, TriangleAlert, Upload, X } from 'lucide-react'
import type { ArtifactInfo, PipelineStageEvent } from './types'
import { apiUrl } from './api'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'

const icons = {
  moon: Moon,
  upload: Upload,
  plus: Plus,
  x: X,
  chevron: ChevronDown,
  download: Download,
  image: FileImage,
  file: FileText,
  activity: Activity,
  clock: Clock3,
  arrow: ArrowRight,
  check: Check,
  alert: TriangleAlert,
  gpu: Cpu,
  history: History,
  spark: Sparkles,
  layers: Layers3,
  external: ExternalLink,
}

export function Mark({ name, size = 18 }: { name: keyof typeof icons; size?: number }) {
  const Icon = icons[name]
  return <Icon size={size} strokeWidth={1.8} aria-hidden="true" />
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

  return <Card className={`product-block ${dragging ? 'is-dragging' : ''}`}>
    <div className="product-topline">
      <span className="product-title">{title}</span>
    </div>
    <label className="field-label" htmlFor={`${inputId}-sensor`}>Instrument</label>
    <Select value={sensor} onValueChange={onSensor}>
      <SelectTrigger id={`${inputId}-sensor`} className="form-select" aria-label="Instrument">
        <SelectValue />
      </SelectTrigger>
      <SelectContent>
        {sensors.map((item) => <SelectItem key={item.value} value={item.value}>{item.label}</SelectItem>)}
      </SelectContent>
    </Select>
    <input id={inputId} ref={fileRef} type="file" accept=".xml,.lbl,.img,.tif,.tiff,.h5,.hdf5" onChange={(e) => setFirstFile(e.target.files)} hidden />
    <input id={extrasId} ref={sidecarRef} type="file" accept=".xml,.lbl,.img,.tif,.tiff,.h5,.hdf5,.dat,.bin" multiple onChange={onSidecarChange} hidden />
    {file ? <div className="selected-file-card">
      <div className="file-icon"><Mark name="file" size={17} /></div>
      <div className="selected-file-copy"><strong title={file.name}>{file.name}</strong><span>{formatBytes(file.size)} · {file.name.split('.').pop()?.toUpperCase()} product</span></div>
      <Button className="icon-button" variant="ghost" size="icon-sm" type="button" aria-label={`Remove ${title}`} onClick={() => onFile(null)}><Mark name="x" size={16} /></Button>
    </div> : <Button
      className="dropzone h-auto min-h-36 w-full flex-col border-dashed bg-muted/40 p-5 text-center shadow-none" variant="outline" type="button" onClick={() => fileRef.current?.click()}
      onDragOver={(e) => { e.preventDefault(); setDragging(true) }} onDragLeave={() => setDragging(false)} onDrop={onDrop}
    >
      <span className="drop-icon"><Mark name="upload" size={19} /></span>
      <span className="drop-title">Drop a product here or <b>browse files</b></span>
      <span className="drop-hint">{hint}</span>
    </Button>}
    <div className="sidecar-line">
      <span><Mark name="layers" size={14} /> Related files{sidecars.length ? ` · ${sidecars.length}` : ''}</span>
      <Button type="button" variant="link" size="sm" className="text-button h-auto px-0" onClick={() => sidecarRef.current?.click()}><Mark name="plus" size={14} /> Add files</Button>
    </div>
    {sidecars.length > 0 && <div className="sidecar-list">{sidecars.map((item, index) => <div className="sidecar-item" key={`${item.name}-${index}`}>
      <span><Mark name="file" size={13} /> {item.name} <small>{formatBytes(item.size)}</small></span>
      <Button type="button" variant="ghost" size="icon-sm" className="icon-button compact" aria-label={`Remove ${item.name}`} onClick={() => onSidecars(sidecars.filter((_, i) => i !== index))}><Mark name="x" size={13} /></Button>
    </div>)}</div>}
  </Card>
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

  return <Card className={`stage-card stage-${state}`}>
    <Button className="stage-heading h-auto w-full justify-start rounded-none px-4 py-3 text-left shadow-none sm:px-5" variant="ghost" type="button" onClick={onToggle} aria-expanded={expanded}>
      <span className={`stage-marker ${state}`}>
        {state === 'complete' ? <Mark name="check" size={17} /> : state === 'failed' ? <Mark name="alert" size={16} /> : <span>{index}</span>}
      </span>
      <span className="stage-heading-copy"><strong>{title}</strong></span>
      <Badge variant={state === 'complete' ? 'success' : state === 'failed' ? 'destructive' : state === 'running' ? 'warning' : 'muted'} className={`state-chip ${state}`}>{state === 'waiting' ? 'Waiting' : state}</Badge>
      <span className={`stage-chevron ${expanded ? 'open' : ''}`}><Mark name="chevron" size={16} /></span>
    </Button>
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
        <div className="artifact-preview-wrap"><img loading="lazy" src={apiUrl(artifact.preview_url!)} alt={artifact.file_name} /></div>
        <figcaption><span>{artifact.file_name}</span><a href={apiUrl(artifact.download_url)} download title="Download preview"><Mark name="download" size={14} /></a></figcaption>
      </figure>)}</div>}
      {previewArtifacts.length > 4 && <Button type="button" variant="link" className="show-more h-auto justify-self-start px-0 py-0" onClick={() => setShowAllPreviews((value) => !value)}>{showAllPreviews ? 'Show fewer previews' : `Show ${previewArtifacts.length - 4} more previews`}</Button>}
      {visibleArtifacts.length > 0 && <div className="artifact-list">
        <SectionLabel trailing={<span className="artifact-count">{visibleArtifacts.length} files</span>}>Stage files</SectionLabel>
        {visibleArtifacts.map((artifact) => <ArtifactRow key={artifact.artifact_id} artifact={artifact} />)}
      </div>}
      {event && visibleArtifacts.length === 0 && state === 'complete' && <p className="artifact-empty">The stage completed without downloadable side-products.</p>}
    </div>}
  </Card>
}

export function ArtifactRow({ artifact }: { artifact: ArtifactInfo }) {
  return <Card className="artifact-row">
    <Badge variant="secondary" className={`artifact-type ${artifact.previewable ? 'visual' : ''}`}>{artifact.previewable ? <Mark name="image" size={14} /> : <Mark name="file" size={14} />}</Badge>
    <div className="artifact-copy"><strong title={artifact.relative_path}>{artifact.file_name}</strong><span>{artifact.relative_path}</span></div>
    <span className="artifact-size">{formatBytes(artifact.size_bytes)}</span>
    <Button asChild variant="ghost" size="icon-sm" className="artifact-download size-9 p-0"><a href={apiUrl(artifact.download_url)} download aria-label={`Download ${artifact.file_name}`} title="Download"><Mark name="download" size={16} /></a></Button>
  </Card>
}

export function MetricCard({ label, value, unit, accent }: { label: string; value: ReactNode; unit?: string; accent?: boolean }) {
  return <Card className={`metric-card ${accent ? 'accent' : ''}`}><span>{label}</span><strong>{value ?? '—'}{unit && value !== undefined && value !== null && <small>{unit}</small>}</strong></Card>
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

export function StageDetails({ details }: { details: Record<string, unknown> }) {
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

export function ImageLightbox({
  list,
  index,
  onClose,
  onNavigate,
}: {
  list: ArtifactInfo[]
  index: number
  onClose: () => void
  onNavigate: (newIndex: number) => void
}) {
  const current = list[index]
  const hasPrev = index > 0
  const hasNext = index < list.length - 1

  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      if (e.key === 'Escape') onClose()
      else if (e.key === 'ArrowLeft' && hasPrev) onNavigate(index - 1)
      else if (e.key === 'ArrowRight' && hasNext) onNavigate(index + 1)
    }
    window.addEventListener('keydown', handleKeyDown)
    const prevOverflow = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    return () => {
      window.removeEventListener('keydown', handleKeyDown)
      document.body.style.overflow = prevOverflow
    }
  }, [index, hasPrev, hasNext, onClose, onNavigate])

  if (!current || !current.preview_url) return null

  return (
    <div className="image-lightbox-overlay" onClick={onClose} role="dialog" aria-modal="true">
      <div className="lightbox-content" onClick={(e) => e.stopPropagation()}>
        <div className="lightbox-header">
          <div className="lightbox-meta">
            <span className="lightbox-badge">
              <Mark name="image" size={14} />
              <span>{list.length > 1 ? `${index + 1} of ${list.length}` : 'Preview'}</span>
            </span>
            <span className="lightbox-filename" title={current.file_name}>
              {current.file_name}
            </span>
          </div>
          <div className="lightbox-actions">
            <a
              href={apiUrl(current.download_url)}
              download
              className="lightbox-btn"
              title="Download image file"
            >
              <Mark name="download" size={15} />
              <span>Download</span>
            </a>
            <button
              type="button"
              className="lightbox-btn close-btn"
              onClick={onClose}
              title="Close full-screen preview (Esc)"
            >
              <Mark name="x" size={17} />
              <span className="kbd-shortcut">ESC</span>
            </button>
          </div>
        </div>

        <div className="lightbox-body">
          {hasPrev && (
              <Button
                type="button"
                variant="outline"
                size="icon"
                className="lightbox-nav-btn prev p-0 sm:size-12"
              onClick={() => onNavigate(index - 1)}
              title="Previous image (← Arrow key)"
              aria-label="Previous image"
            >
              <Mark name="arrow" size={20} />
              </Button>
          )}

          <div className="lightbox-img-container">
            <img
              src={apiUrl(current.preview_url)}
              alt={current.file_name}
              className="lightbox-main-img"
            />
          </div>

          {hasNext && (
              <Button
                type="button"
                variant="outline"
                size="icon"
                className="lightbox-nav-btn next p-0 sm:size-12"
              onClick={() => onNavigate(index + 1)}
              title="Next image (→ Arrow key)"
              aria-label="Next image"
            >
              <Mark name="arrow" size={20} />
              </Button>
          )}
        </div>

        <div className="lightbox-footer">
          <span className="lightbox-stage-pill">{current.stage.toUpperCase().replace('_', ' ')}</span>
          <span className="lightbox-path">{current.relative_path}</span>
          <span className="lightbox-size">{formatBytes(current.size_bytes)}</span>
        </div>
      </div>
    </div>
  )
}
