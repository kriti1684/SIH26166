import { useEffect, useId, useRef, useState } from 'react'
import type { DragEvent, ReactNode } from 'react'
import { Activity, ArrowRight, Check, ChevronDown, Clock3, Copy, Cpu, Download, ExternalLink, FileImage, FileText, FolderOpen, FolderPlus, Globe, History, Layers3, Maximize2, Moon, Move, Orbit, Percent, Plus, Radio, Ruler, Satellite, Scan, ShieldCheck, Sparkles, Sun, TriangleAlert, Upload, X } from 'lucide-react'
import type { ArtifactInfo, PipelineStageEvent } from './types'
import { apiUrl } from './api'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'

const icons = {
  moon: Moon,
  sun: Sun,
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
  folderPlus: FolderPlus,
  folderOpen: FolderOpen,
  satellite: Satellite,
  orbit: Orbit,
  radio: Radio,
}

export function Mark({ name, size = 18, className }: { name: keyof typeof icons; size?: number; className?: string }) {
  const Icon = icons[name]
  return <Icon size={size} strokeWidth={1.8} aria-hidden="true" className={className} />
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
  const fileRef = useRef<HTMLInputElement>(null)
  const [dragging, setDragging] = useState(false)

  function handleIncomingFiles(incoming: File[]) {
    if (!incoming.length) return

    if (!file) {
      let primaryIndex = incoming.findIndex((f) => {
        const ext = f.name.split('.').pop()?.toLowerCase()
        return ext === 'xml' || ext === 'lbl' || ext === 'tif' || ext === 'tiff'
      })
      if (primaryIndex === -1) {
        primaryIndex = incoming.findIndex((f) => {
          const ext = f.name.split('.').pop()?.toLowerCase()
          return ext === 'img' || ext === 'h5' || ext === 'hdf5'
        })
      }
      if (primaryIndex === -1) primaryIndex = 0

      const primary = incoming[primaryIndex]
      const others = incoming.filter((_, idx) => idx !== primaryIndex)

      onFile(primary)

      if (others.length > 0) {
        const existingNames = new Set(sidecars.map((s) => s.name.toLowerCase()))
        const newSidecars = others.filter((f) => !existingNames.has(f.name.toLowerCase()))
        if (newSidecars.length > 0) {
          onSidecars([...sidecars, ...newSidecars])
        }
      }
    } else {
      const existingNames = new Set([file.name.toLowerCase(), ...sidecars.map((s) => s.name.toLowerCase())])
      const newSidecars = incoming.filter((f) => !existingNames.has(f.name.toLowerCase()))
      if (newSidecars.length > 0) {
        onSidecars([...sidecars, ...newSidecars])
      }
    }
  }

  function onDropCard(event: DragEvent<HTMLDivElement>) {
    event.preventDefault()
    setDragging(false)
    const dropped = Array.from(event.dataTransfer.files ?? [])
    handleIncomingFiles(dropped)
  }

  function setAsPrimary(index: number) {
    const target = sidecars[index]
    if (!target) return
    const remaining = sidecars.filter((_, i) => i !== index)
    if (file) {
      remaining.unshift(file)
    }
    onFile(target)
    onSidecars(remaining)
  }

  function removePrimary() {
    if (sidecars.length > 0) {
      const nextPrimary = sidecars[0]
      onFile(nextPrimary)
      onSidecars(sidecars.slice(1))
    } else {
      onFile(null)
    }
  }

  function clearAll() {
    onFile(null)
    onSidecars([])
  }

  const totalFiles = (file ? 1 : 0) + sidecars.length

  return (
    <Card
      className={`product-block ${dragging ? 'is-dragging' : ''}`}
      onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
      onDragLeave={(e) => {
        if (!e.currentTarget.contains(e.relatedTarget as Node)) {
          setDragging(false)
        }
      }}
      onDrop={onDropCard}
    >
      <div className="product-topline">
        <div className="flex items-center gap-2">
          <span className="product-title">{title}</span>
          {totalFiles > 0 && (
            <Badge variant="secondary" className="font-mono text-[11px] px-2 py-0.5">
              {totalFiles} {totalFiles === 1 ? 'file' : 'files'}
            </Badge>
          )}
        </div>
        {totalFiles > 0 && (
          <div className="flex items-center gap-1.5">
            <Button
              type="button"
              variant="outline"
              size="sm"
              className="h-7 px-2 text-xs font-semibold text-primary hover:bg-primary/10 gap-1"
              onClick={() => fileRef.current?.click()}
              title="Add more files to this product"
            >
              <Mark name="plus" size={13} />
              <span>Add files</span>
            </Button>
            <Button
              type="button"
              variant="ghost"
              size="sm"
              className="h-7 px-2 text-xs text-muted-foreground hover:text-destructive hover:bg-destructive/10"
              onClick={clearAll}
              title={`Remove all files from ${title}`}
            >
              Clear
            </Button>
          </div>
        )}
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

      <input
        id={inputId}
        ref={fileRef}
        type="file"
        multiple
        accept=".xml,.lbl,.img,.tif,.tiff,.h5,.hdf5,.dat,.bin,.tfw,.prj,.hdr"
        onChange={(e) => {
          handleIncomingFiles(Array.from(e.target.files ?? []))
          if (fileRef.current) fileRef.current.value = ''
        }}
        hidden
      />

      {file ? (
        <div className="space-y-3">
          {/* Main Product Card */}
          <div className="selected-file-card">
            <div className="file-icon"><Mark name="file" size={17} /></div>
            <div className="selected-file-copy">
              <div className="flex items-center gap-1.5 flex-wrap">
                <strong title={file.name}>{file.name}</strong>
                <span className="inline-flex items-center rounded-md bg-primary/15 text-primary text-[10px] font-bold px-1.5 py-0.5 uppercase tracking-wide">
                  Main Product
                </span>
              </div>
              <span>{formatBytes(file.size)} · {file.name.split('.').pop()?.toUpperCase()} product</span>
            </div>
            <Button
              className="icon-button"
              variant="ghost"
              size="icon-sm"
              type="button"
              aria-label={`Remove ${file.name}`}
              onClick={removePrimary}
              title="Remove primary file"
            >
              <Mark name="x" size={16} />
            </Button>
          </div>

          {/* Sidecars / Related Files */}
          {sidecars.length > 0 && (
            <div className="space-y-1.5">
              <div className="sidecar-line">
                <span className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground flex items-center gap-1.5">
                  <Mark name="layers" size={13} />
                  Related files ({sidecars.length})
                </span>
              </div>
              <div className="sidecar-list">
                {sidecars.map((item, index) => (
                  <div className="sidecar-item" key={`${item.name}-${index}`}>
                    <div className="flex items-center gap-2 min-w-0 flex-1">
                      <Mark name="file" size={13} />
                      <span className="truncate" title={item.name}>{item.name}</span>
                      <small>{formatBytes(item.size)}</small>
                    </div>
                    <div className="flex items-center gap-1 shrink-0">
                      <Button
                        type="button"
                        variant="ghost"
                        size="sm"
                        className="h-6 px-1.5 text-[10px] text-muted-foreground hover:text-primary hover:bg-primary/10"
                        onClick={() => setAsPrimary(index)}
                        title="Set this file as the primary product"
                      >
                        Set as Main
                      </Button>
                      <Button
                        type="button"
                        variant="ghost"
                        size="icon-sm"
                        className="icon-button compact"
                        aria-label={`Remove ${item.name}`}
                        onClick={() => onSidecars(sidecars.filter((_, i) => i !== index))}
                        title="Remove file"
                      >
                        <Mark name="x" size={13} />
                      </Button>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Compact drop-strip for adding more files */}
          <div
            className="mini-drop-strip"
            onClick={() => fileRef.current?.click()}
            role="button"
            tabIndex={0}
            onKeyDown={(e) => { if (e.key === 'Enter') fileRef.current?.click() }}
            title="Click or drag to add more files to this product"
          >
            <Mark name="upload" size={14} className="text-primary/70" />
            <span>Drop more files here or <b>browse</b> to add</span>
          </div>
        </div>
      ) : (
        <Button
          className="dropzone h-auto min-h-36 w-full flex-col border-dashed bg-muted/40 p-5 text-center shadow-none"
          variant="outline"
          type="button"
          onClick={() => fileRef.current?.click()}
        >
          <span className="drop-icon"><Mark name="upload" size={19} /></span>
          <span className="drop-title">Drop product files here or <b>browse files</b></span>
          <span className="drop-hint">{hint}</span>
          <span className="mt-1 text-[11px] font-mono text-muted-foreground bg-secondary/80 border border-border/60 px-2 py-0.5 rounded-full">
            Supports multi-file selection (.xml + .img, GeoTIFF + sidecars)
          </span>
        </Button>
      )}
    </Card>
  )
}

export function QuickBatchUploader({
  onBatchUploaded,
}: {
  onBatchUploaded: (data: {
    sourceFile?: File
    sourceSidecars: File[]
    referenceFile?: File
    referenceSidecars: File[]
    detectedSensors?: { src?: string; ref?: string }
    message: string
  }) => void
}) {
  const batchInputRef = useRef<HTMLInputElement>(null)
  const [dragging, setDragging] = useState(false)

  function processFiles(files: File[]) {
    if (!files.length) return

    const sourceSensors = ['ohrc', 'iirs', 'tmc', 'tmc2', 'ch2', 'chandrayaan', 'src', 'source']
    const refSensors = ['nac', 'wac', 'selene', 'tc', 'ref', 'reference', 'lro']

    const isSource = (name: string) => sourceSensors.some((k) => name.toLowerCase().includes(k))
    const isRef = (name: string) => refSensors.some((k) => name.toLowerCase().includes(k))

    let sourcePool: File[] = []
    let refPool: File[] = []
    const unclassified: File[] = []

    for (const f of files) {
      if (isSource(f.name)) {
        sourcePool.push(f)
      } else if (isRef(f.name)) {
        refPool.push(f)
      } else {
        unclassified.push(f)
      }
    }

    if (sourcePool.length === 0 && refPool.length === 0) {
      if (unclassified.length === 1) {
        sourcePool.push(unclassified[0])
      } else if (unclassified.length >= 2) {
        sourcePool.push(unclassified[0])
        refPool = unclassified.slice(1)
      }
    } else if (unclassified.length > 0) {
      for (const u of unclassified) {
        const stem = u.name.split('.')[0].toLowerCase()
        const matchesSrc = sourcePool.some((s) => s.name.toLowerCase().startsWith(stem) || stem.startsWith(s.name.split('.')[0].toLowerCase()))
        const matchesRef = refPool.some((r) => r.name.toLowerCase().startsWith(stem) || stem.startsWith(r.name.split('.')[0].toLowerCase()))

        if (matchesSrc && !matchesRef) {
          sourcePool.push(u)
        } else if (matchesRef && !matchesSrc) {
          refPool.push(u)
        } else if (sourcePool.length === 0) {
          sourcePool.push(u)
        } else {
          refPool.push(u)
        }
      }
    }

    function pickPrimaryAndSidecars(pool: File[]) {
      if (!pool.length) return { primary: undefined, sidecars: [] }
      let primaryIdx = pool.findIndex((f) => {
        const ext = f.name.split('.').pop()?.toLowerCase()
        return ext === 'xml' || ext === 'lbl' || ext === 'tif' || ext === 'tiff'
      })
      if (primaryIdx === -1) {
        primaryIdx = pool.findIndex((f) => {
          const ext = f.name.split('.').pop()?.toLowerCase()
          return ext === 'img' || ext === 'h5' || ext === 'hdf5'
        })
      }
      if (primaryIdx === -1) primaryIdx = 0

      return {
        primary: pool[primaryIdx],
        sidecars: pool.filter((_, i) => i !== primaryIdx),
      }
    }

    const srcRes = pickPrimaryAndSidecars(sourcePool)
    const refRes = pickPrimaryAndSidecars(refPool)

    // Detect sensor auto-suggestions
    const detectedSensors: { src?: string; ref?: string } = {}
    if (srcRes.primary) {
      const n = srcRes.primary.name.toLowerCase()
      if (n.includes('ohrc')) detectedSensors.src = 'OHRC'
      else if (n.includes('iirs')) detectedSensors.src = 'IIRS'
      else if (n.includes('tmc2')) detectedSensors.src = 'TMC2'
      else if (n.includes('tmc')) detectedSensors.src = 'TMC'
    }
    if (refRes.primary) {
      const n = refRes.primary.name.toLowerCase()
      if (n.includes('wac')) detectedSensors.ref = 'WAC'
      else if (n.includes('nac')) detectedSensors.ref = 'NAC'
      else if (n.includes('selene')) detectedSensors.ref = 'SELENE'
      else if (n.includes('tc')) detectedSensors.ref = 'TC'
    }

    const parts: string[] = []
    if (srcRes.primary) parts.push(`Source: ${srcRes.primary.name}${srcRes.sidecars.length ? ` (+${srcRes.sidecars.length} sidecars)` : ''}`)
    if (refRes.primary) parts.push(`Reference: ${refRes.primary.name}${refRes.sidecars.length ? ` (+${refRes.sidecars.length} sidecars)` : ''}`)

    const message = `Auto-assigned ${files.length} file(s) — ${parts.join(' · ')}`

    onBatchUploaded({
      sourceFile: srcRes.primary,
      sourceSidecars: srcRes.sidecars,
      referenceFile: refRes.primary,
      referenceSidecars: refRes.sidecars,
      detectedSensors,
      message,
    })
  }

  return (
    <div
      className={`quick-batch-card ${dragging ? 'is-dragging' : ''}`}
      onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
      onDragLeave={(e) => {
        if (!e.currentTarget.contains(e.relatedTarget as Node)) {
          setDragging(false)
        }
      }}
      onDrop={(e) => {
        e.preventDefault()
        setDragging(false)
        processFiles(Array.from(e.dataTransfer.files ?? []))
      }}
      onClick={() => batchInputRef.current?.click()}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => { if (e.key === 'Enter') batchInputRef.current?.click() }}
    >
      <input
        ref={batchInputRef}
        type="file"
        multiple
        accept=".xml,.lbl,.img,.tif,.tiff,.h5,.hdf5,.dat,.bin,.tfw,.prj,.hdr"
        onChange={(e) => {
          processFiles(Array.from(e.target.files ?? []))
          if (batchInputRef.current) batchInputRef.current.value = ''
        }}
        hidden
      />
      <div className="flex items-center justify-between gap-3 flex-wrap">
        <div className="flex items-center gap-3">
          <div className="size-10 rounded-xl bg-sky-500/15 border border-sky-500/30 flex items-center justify-center text-sky-400 shrink-0 shadow-sm">
            <Mark name="layers" size={20} />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="text-xs font-bold uppercase tracking-wider text-foreground">
                Dual / Multi-Product Quick Drop
              </span>
              <Badge variant="outline" className="font-mono text-[10px] bg-sky-500/10 border-sky-500/30 text-sky-400 py-0">
                All-in-One
              </Badge>
            </div>
            <p className="text-xs text-muted-foreground m-0">
              Drop all your Source and Reference files together — ChandraShakti will auto-sort and distribute them.
            </p>
          </div>
        </div>
        <Button
          type="button"
          variant="outline"
          size="sm"
          className="h-8 px-3 text-xs font-semibold text-primary border-primary/30 hover:bg-primary/10 gap-1.5 shrink-0"
          onClick={(e) => {
            e.stopPropagation()
            batchInputRef.current?.click()
          }}
        >
          <Mark name="upload" size={14} />
          <span>Upload Multiple Files</span>
        </Button>
      </div>
    </div>
  )
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
      </div>}
      {event?.details && <StageDetails details={event.details} />}
      {previews.length > 0 && <div className="preview-grid">{previews.map((artifact) => <figure className="artifact-preview" key={artifact.artifact_id}>
        <div className="artifact-preview-wrap"><img loading="lazy" src={apiUrl(artifact.preview_url!)} alt={formatPreviewTitle(artifact.file_name)} /></div>
        <figcaption><span title={artifact.file_name}>{formatPreviewTitle(artifact.file_name)}</span></figcaption>
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

const CANONICAL_PREVIEWS: Record<string, string> = {
  warp_composite_overlay: 'Warp Composite Overlay',
  registered_preview: 'Registered Sub-Pixel Image',
  difference_heatmap: 'Residual Difference Heatmap',
  registration_verification: 'Multi-Pillar Verification Summary',
  overview_side_by_side: 'Side-by-Side Comparison',
  overview_false_color: 'False-Color Anaglyph Overlay',
  stage_01_reference_harmonized: 'Harmonized Reference Product (NAC)',
  stage_01_source_harmonized: 'Harmonized Source Product (OHRC)',
  harmonized_reference: 'Harmonized Reference Product (NAC)',
  harmonized_source: 'Harmonized Source Product (OHRC)',
  coarse_shift_preview: 'Coarse Shift & Polar Alignment',
  heatmap: 'Residual Difference Heatmap',
  diff_map: 'Pixel Error Difference Map',
  checkerboard: 'Checkerboard Composite Blend',
  edge_overlay: 'Canny Edge Boundary Alignment',
}

const KNOWN_ACRONYMS: Record<string, string> = {
  nac: 'NAC',
  ohrc: 'OHRC',
  crs: 'CRS',
  wkt: 'WKT',
  dem: 'DEM',
  rmse: 'RMSE',
  ssim: 'SSIM',
  ncc: 'NCC',
  gcp: 'GCP',
  tps: 'TPS',
  id: 'ID',
  rgb: 'RGB',
  psf: 'PSF',
  fft: 'FFT',
}

export function formatPreviewTitle(rawName: string): string {
  if (!rawName) return ''

  // Strip file extension (.png, .jpg, .jpeg, etc.)
  const stem = rawName.replace(/\.(png|jpe?g|webp|tiff?|bmp|svg)$/i, '').trim()

  // 1. Check exact canonical mapping
  if (CANONICAL_PREVIEWS[stem.toLowerCase()]) {
    return CANONICAL_PREVIEWS[stem.toLowerCase()]
  }

  // 2. Tile match visualizations: tile_0003_matches -> Dense Tie-Points (Tile #0003)
  const tileMatchRegex = /^tile_?(\d+)_matches$/i.exec(stem)
  if (tileMatchRegex) {
    return `Dense Tie-Points (Tile #${tileMatchRegex[1]})`
  }

  // 3. Tile source/ref patches
  const tileOhrcRegex = /^tile_?(\d+)_ohrc$/i.exec(stem)
  if (tileOhrcRegex) {
    return `OHRC Source Patch (Tile #${tileOhrcRegex[1]})`
  }
  const tileNacRegex = /^tile_?(\d+)_nac$/i.exec(stem)
  if (tileNacRegex) {
    return `NAC Reference Patch (Tile #${tileNacRegex[1]})`
  }

  // 4. Native registered resolutions: registered_native_25m_preview -> Native Registered Product (25m)
  const nativeRegex = /^registered_native_(\d+m?)(?:_preview)?$/i.exec(stem)
  if (nativeRegex) {
    return `Native Registered Product (${nativeRegex[1]})`
  }

  // 5. Clean Fallback for any other filename:
  // Remove leading stage prefixes: stage_01_, stage_1_, stage1_
  let cleaned = stem.replace(/^stage_0?(\d+)_/i, '')
  // Remove redundant _preview or preview_
  cleaned = cleaned.replace(/^preview_/i, '').replace(/_preview$/i, '')

  // Split into words, replace underscores and dashes with spaces
  const words = cleaned.split(/[_\-\s]+/).filter(Boolean)
  if (words.length === 0) return stem

  return words
    .map((w) => {
      const lower = w.toLowerCase()
      if (KNOWN_ACRONYMS[lower]) return KNOWN_ACRONYMS[lower]
      return lower.charAt(0).toUpperCase() + lower.slice(1)
    })
    .join(' ')
}

export function EmptyStageList() {
  return <div className="stage-empty-state">
    <div className="empty-orbit"><span /><i /><b><Mark name="moon" size={24} /></b></div>
    <h2>Ready for a run</h2>
    <p>Select both instruments, then add a source and reference product.</p>
  </div>
}

export function SpaceWaitingCard({
  stage,
  jobActive,
  prevStageName,
  stageIndex,
  onGoToInput,
}: {
  stage: { id: string; index: string; shortName: string; title: string; description: string }
  jobActive: boolean
  prevStageName?: string
  stageIndex: number
  onGoToInput?: () => void
}) {
  const STAGE_STEPS = [
    { index: '01', name: 'Harmonize' },
    { index: '02', name: 'Coarse Align' },
    { index: '03', name: 'Dense Match' },
    { index: '04', name: 'Hybrid Warp' },
    { index: '05', name: 'Verification' },
  ]

  return (
    <Card className="stage-waiting-card space-waiting-card">
      {/* Lunar & Space Orbital Viewport */}
      <div className="lunar-system" aria-hidden="true">
        {/* Star Field with Twinkling Stars */}
        <div className="star-field">
          <span className="space-star" style={{ top: '12%', left: '16%', width: '3px', height: '3px', animationDelay: '0.2s' }} />
          <span className="space-star" style={{ top: '22%', right: '14%', width: '4px', height: '4px', animationDelay: '1.4s' }} />
          <span className="space-star" style={{ bottom: '18%', left: '12%', width: '2.5px', height: '2.5px', animationDelay: '0.8s' }} />
          <span className="space-star" style={{ bottom: '26%', right: '18%', width: '3px', height: '3px', animationDelay: '2.1s' }} />
          <span className="space-star" style={{ top: '6%', right: '38%', width: '2px', height: '2px', animationDelay: '1.7s' }} />
          <span className="space-star" style={{ bottom: '8%', left: '42%', width: '3.5px', height: '3.5px', animationDelay: '0.5s' }} />
        </div>

        {/* Radar Horizon Sweep Beam */}
        <div className="radar-sweep-beam" />

        {/* Outer Orbital Ring */}
        <div className="orbit-ring-outer" />

        {/* 3D Tilted Orbital Track with Revolving Satellite */}
        <div className="orbit-track-tilted">
          <div className="satellite-orbiter">
            <div className="satellite-ping" />
            <div className="satellite-body" title="Orbital Sensor Probe">
              <Satellite size={13} className="text-sky-300" />
            </div>
          </div>
        </div>

        {/* The Cratered Moon Globe */}
        <div className="moon-body">
          <div className="moon-crater crater-1" />
          <div className="moon-crater crater-2" />
          <div className="moon-crater crater-3" />
          <div className="moon-crater crater-4" />
          <div className="moon-crater crater-5" />
        </div>
      </div>

      {/* Waiting Info & Mission Telemetry */}
      <div className="waiting-info">
        <div className="flex items-center gap-2">
          {jobActive ? (
            <Badge variant="outline" className="waiting-stage-badge bg-sky-500/15 text-sky-400 border-sky-500/35">
              <span className="size-2 rounded-full bg-sky-400 animate-ping" />
              <span>ORBITAL PIPELINE QUEUED</span>
            </Badge>
          ) : (
            <Badge variant="outline" className="waiting-stage-badge bg-secondary text-muted-foreground border-border/80">
              <span className="size-2 rounded-full bg-amber-400/80" />
              <span>PRE-FLIGHT STANDBY</span>
            </Badge>
          )}
        </div>

        <h3 className="waiting-title">
          Stage {stage.index}: {stage.title}
        </h3>

        <p className="waiting-desc">
          {jobActive
            ? `Awaiting telemetry and data artifacts from ${prevStageName ? `Stage (${prevStageName})` : 'preceding stage'} before initiation. The registration engine will automatically execute this phase upon receiving inputs.`
            : 'This computational stage is queued and waiting. Configure your products in Input Setup and click "Start registration run" to initiate orbital alignment.'}
        </p>

        {!jobActive && onGoToInput && (
          <Button
            type="button"
            variant="outline"
            size="sm"
            onClick={onGoToInput}
            className="mt-1 h-8 px-3.5 text-xs font-semibold text-primary border-primary/30 hover:bg-primary/10 gap-1.5"
          >
            <span>Configure &amp; Launch in Input Setup</span>
            <ArrowRight size={13} />
          </Button>
        )}

        {/* Mission Roadmap Progress Indicator */}
        <div className="pipeline-roadmap" aria-label="Pipeline sequence">
          {STAGE_STEPS.map((step, idx) => {
            const isCurrent = idx === stageIndex
            return (
              <div key={step.index} className="flex items-center gap-1">
                <div
                  className={`roadmap-step ${isCurrent ? 'active' : 'pending'}`}
                  title={`Stage ${step.index}: ${step.name}`}
                >
                  <span className="font-bold opacity-75">{step.index}</span>
                  <span className="truncate max-w-[70px] sm:max-w-none">{step.name}</span>
                </div>
                {idx < STAGE_STEPS.length - 1 && <span className="roadmap-arrow">›</span>}
              </div>
            )
          })}
        </div>
      </div>
    </Card>
  )
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

export function TransformStatsView({ stats }: { stats: Record<string, unknown> }) {
  const inlierCount = stats.inlier_count !== undefined ? Number(stats.inlier_count) : null
  const totalPoints = stats.total_points !== undefined ? Number(stats.total_points) : null
  const inlierRatio = stats.inlier_ratio !== undefined ? Number(stats.inlier_ratio) : null
  const coverage = stats.coverage_pct !== undefined ? Number(stats.coverage_pct) : null
  const rmseL1 = stats.rmse_layer1_affine !== undefined ? Number(stats.rmse_layer1_affine) : null
  const rmseL2 = stats.rmse_layer2_poly !== undefined ? Number(stats.rmse_layer2_poly) : null
  const rmseL3 = stats.rmse_layer3_tps !== undefined ? Number(stats.rmse_layer3_tps) : null
  const polyDegree = stats.poly_degree !== undefined ? Number(stats.poly_degree) : null
  const tpsSmoothing = stats.tps_smoothing !== undefined ? Number(stats.tps_smoothing) : null

  const tiles: Array<{ label: string; value: string; hint?: string; highlight?: boolean }> = []

  if (inlierCount !== null) {
    tiles.push({
      label: 'Inlier Matches',
      value: inlierCount.toLocaleString(),
      hint: totalPoints ? `of ${totalPoints.toLocaleString()} tie points` : undefined,
      highlight: true,
    })
  }

  if (totalPoints !== null && inlierCount === null) {
    tiles.push({
      label: 'Total Tie Points',
      value: totalPoints.toLocaleString(),
    })
  }

  if (inlierRatio !== null) {
    tiles.push({
      label: 'Inlier Ratio',
      value: `${(inlierRatio * 100).toFixed(1)}%`,
      hint: `Score: ${inlierRatio.toFixed(3)}`,
    })
  }

  if (coverage !== null) {
    tiles.push({
      label: 'Swath Coverage',
      value: `${coverage.toFixed(1)}%`,
      hint: 'Spatial distribution',
    })
  }

  if (rmseL1 !== null) {
    tiles.push({
      label: 'L1 Affine RMSE',
      value: `${rmseL1.toFixed(3)} px`,
      hint: 'Global linear model',
    })
  }

  if (rmseL2 !== null) {
    tiles.push({
      label: 'L2 Drift RMSE',
      value: `${rmseL2.toFixed(3)} px`,
      hint: polyDegree ? `Poly degree ${polyDegree}` : 'Along-track jitter',
    })
  }

  if (rmseL3 !== null && rmseL3 > 0) {
    tiles.push({
      label: 'L3 TPS RMSE',
      value: `${rmseL3.toFixed(3)} px`,
      hint: tpsSmoothing ? `TPS λ=${tpsSmoothing}` : 'Local topography',
    })
  } else if (stats.tps_smoothing !== undefined) {
    tiles.push({
      label: 'L3 TPS Layer',
      value: 'Bypassed (L2 Robust)',
      hint: 'Parallax minimal',
    })
  }

  const knownKeys = new Set([
    'inlier_count', 'total_points', 'inlier_ratio', 'coverage_pct',
    'rmse_layer1_affine', 'rmse_layer2_poly', 'rmse_layer3_tps',
    'poly_degree', 'tps_smoothing',
  ])
  for (const [k, v] of Object.entries(stats)) {
    if (!knownKeys.has(k) && v !== null && v !== undefined) {
      tiles.push({
        label: k.replaceAll('_', ' '),
        value: typeof v === 'number' ? (Number.isInteger(v) ? v.toLocaleString() : v.toFixed(3)) : String(v),
      })
    }
  }

  return (
    <div className="stage-detail span-full">
      <div className="flex flex-wrap items-center justify-between gap-2 pb-2 border-b border-border/70">
        <span className="text-xs font-bold uppercase tracking-wider text-foreground/90 flex items-center gap-1.5">
          <Layers3 size={15} className="text-primary" />
          Fitted 3-Layer Hybrid Transform Statistics
        </span>
        <div className="flex items-center gap-1.5 font-mono text-[11px] text-muted-foreground bg-secondary px-2 py-0.5 rounded-md">
          <span>{polyDegree ? `Poly Degree ${polyDegree}` : 'Affine + Poly'}</span>
          {tpsSmoothing ? <span>· TPS λ={tpsSmoothing}</span> : null}
        </div>
      </div>
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5 pt-2">
        {tiles.map((t, idx) => (
          <div key={idx} className="rounded-lg border border-border/80 bg-secondary/60 p-2.5 flex flex-col justify-between transition-colors hover:bg-secondary">
            <span className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">{t.label}</span>
            <div className="mt-1">
              <span className={`font-mono text-sm font-bold ${t.highlight ? 'text-primary' : 'text-foreground'}`}>{t.value}</span>
              {t.hint && <div className="text-[10px] text-muted-foreground font-mono mt-0.5">{t.hint}</div>}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

export function CrsView({ crs }: { crs: string }) {
  const [copied, setCopied] = useState(false)
  const [expanded, setExpanded] = useState(false)

  const projMatch = crs.match(/PROJCS\["([^"]+)"/i) || crs.match(/GEOGCS\["([^"]+)"/i)
  const projName = projMatch ? projMatch[1] : 'Lunar Planetary Coordinate System'

  const datumMatch = crs.match(/DATUM\["([^"]+)"/i)
  const datumName = datumMatch ? datumMatch[1] : null

  const spheroidMatch = crs.match(/SPHEROID\["([^"]+)",\s*([0-9.]+)/i)
  const spheroidRadius = spheroidMatch ? Number(spheroidMatch[2]).toLocaleString() : null

  const parallelMatch = crs.match(/standard_parallel_1"?,\s*([-\d.]+)/i)
  const standardParallel = parallelMatch ? `${parallelMatch[1]}°` : null

  const meridianMatch = crs.match(/central_meridian"?,\s*([-\d.]+)/i)
  const centralMeridian = meridianMatch ? `${meridianMatch[1]}°` : null

  function handleCopy() {
    void navigator.clipboard.writeText(crs)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <div className="stage-detail span-full">
      <div className="flex flex-wrap items-center justify-between gap-2 pb-2 border-b border-border/70">
        <div className="flex items-center gap-2">
          <span className="text-xs font-bold uppercase tracking-wider text-foreground/90 flex items-center gap-1.5">
            <Globe size={15} className="text-primary" />
            Coordinate Reference System (CRS)
          </span>
          <span className="inline-flex items-center gap-1 rounded-full bg-primary/10 px-2.5 py-0.5 text-xs font-semibold text-primary">
            {projName}
          </span>
        </div>
        <div className="flex items-center gap-1.5">
          <button
            type="button"
            onClick={handleCopy}
            className="inline-flex items-center gap-1 text-[11px] font-medium text-muted-foreground hover:text-primary px-2.5 py-1 rounded bg-secondary hover:bg-secondary/80 transition-colors"
            title="Copy raw WKT projection to clipboard"
          >
            {copied ? <Check size={12} className="text-emerald-500" /> : <Copy size={12} />}
            <span>{copied ? 'Copied WKT' : 'Copy WKT'}</span>
          </button>
          <button
            type="button"
            onClick={() => setExpanded(!expanded)}
            className="text-[11px] font-medium text-muted-foreground hover:text-foreground px-2 py-1 rounded hover:bg-secondary transition-colors"
          >
            {expanded ? 'Hide Raw WKT' : 'View Raw WKT'}
          </button>
        </div>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 pt-2">
        {datumName && (
          <div className="rounded-lg border border-border/80 bg-secondary/60 p-2">
            <span className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground block">Planetary Datum</span>
            <span className="font-mono text-xs font-semibold text-foreground">{datumName}</span>
          </div>
        )}
        {spheroidRadius && (
          <div className="rounded-lg border border-border/80 bg-secondary/60 p-2">
            <span className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground block">Lunar Radius</span>
            <span className="font-mono text-xs font-semibold text-foreground">{spheroidRadius} m</span>
          </div>
        )}
        {standardParallel && (
          <div className="rounded-lg border border-border/80 bg-secondary/60 p-2">
            <span className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground block">Standard Parallel</span>
            <span className="font-mono text-xs font-semibold text-foreground">{standardParallel}</span>
          </div>
        )}
        {centralMeridian && (
          <div className="rounded-lg border border-border/80 bg-secondary/60 p-2">
            <span className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground block">Central Meridian</span>
            <span className="font-mono text-xs font-semibold text-foreground">{centralMeridian}</span>
          </div>
        )}
      </div>

      {expanded && (
        <div className="mt-2.5 rounded-lg border border-border/80 bg-secondary/50 p-2.5 font-mono text-[11px] text-foreground/90 break-all leading-relaxed max-h-36 overflow-y-auto select-all shadow-inner">
          {crs}
        </div>
      )}
    </div>
  )
}

export function FilePairBanner({ source, reference }: { source?: string; reference?: string }) {
  if (!source && !reference) return null
  return (
    <div className="stage-detail span-full stat-card-hero !p-3 bg-card">
      <div className="flex flex-wrap items-center justify-between gap-2.5">
        <div className="flex items-center gap-2">
          <div className="size-7 rounded-lg bg-primary/15 border border-primary/30 grid place-items-center text-primary">
            <FileText size={15} />
          </div>
          <span className="text-[11px] font-bold uppercase tracking-wider text-foreground/90">Georeferenced Input Products</span>
        </div>
        <div className="flex flex-wrap items-center gap-2 text-xs font-mono">
          {source && (
            <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-secondary/60 border border-border/80 shadow-2xs">
              <span className="size-2 rounded-full bg-blue-500 animate-pulse" />
              <span className="text-[10px] font-bold uppercase text-muted-foreground">Src:</span>
              <span className="font-semibold text-foreground truncate max-w-[220px] sm:max-w-none">{source}</span>
            </div>
          )}
          {source && reference && <ArrowRight size={13} className="text-muted-foreground shrink-0" />}
          {reference && (
            <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-secondary/60 border border-border/80 shadow-2xs">
              <span className="size-2 rounded-full bg-purple-500" />
              <span className="text-[10px] font-bold uppercase text-muted-foreground">Ref:</span>
              <span className="font-semibold text-foreground truncate max-w-[220px] sm:max-w-none">{reference}</span>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export function GsdCard({ gsd }: { gsd: number }) {
  return (
    <div className="stage-detail stat-card-hero">
      <div className="flex items-center justify-between pb-2 border-b border-border/60">
        <div className="flex items-center gap-2">
          <div className="size-8 rounded-xl bg-cyan-500/10 border border-cyan-500/20 grid place-items-center text-cyan-400">
            <Ruler size={16} />
          </div>
          <div>
            <span className="text-xs font-bold uppercase tracking-wider text-foreground/90 block">Spatial Resolution (GSD)</span>
            <span className="text-[11px] text-muted-foreground">Ground sampling distance per pixel</span>
          </div>
        </div>
        <Badge variant="outline" className="font-mono text-[10px] bg-cyan-500/10 border-cyan-500/30 text-cyan-400 font-semibold">
          Equirectangular
        </Badge>
      </div>
      <div className="pt-3 flex items-baseline justify-between">
        <div className="flex items-baseline gap-2">
          <span className="font-mono text-3xl font-extrabold tracking-tight text-foreground">{gsd.toFixed(3)}</span>
          <span className="font-mono text-xs font-bold text-cyan-400 bg-cyan-500/10 px-2 py-0.5 rounded-md border border-cyan-500/20">
            m / px
          </span>
        </div>
        <span className="text-[11px] text-muted-foreground">Harmonized target scale</span>
      </div>
    </div>
  )
}

export function ScaleRatioCard({ ratio }: { ratio: number }) {
  return (
    <div className="stage-detail stat-card-hero">
      <div className="flex items-center justify-between pb-2 border-b border-border/60">
        <div className="flex items-center gap-2">
          <div className="size-8 rounded-xl bg-violet-500/10 border border-violet-500/20 grid place-items-center text-violet-400">
            <Maximize2 size={16} />
          </div>
          <div>
            <span className="text-xs font-bold uppercase tracking-wider text-foreground/90 block">Scale Disparity Ratio</span>
            <span className="text-[11px] text-muted-foreground">Cross-sensor spatial resolution factor</span>
          </div>
        </div>
        <Badge variant="outline" className="font-mono text-[10px] bg-violet-500/10 border-violet-500/30 text-violet-400 font-semibold">
          Cross-Sensor
        </Badge>
      </div>
      <div className="pt-3 flex items-baseline justify-between">
        <div className="flex items-baseline gap-1">
          <span className="font-mono text-3xl font-extrabold tracking-tight text-foreground">{ratio.toFixed(2)}</span>
          <span className="font-extrabold text-2xl text-violet-400 ml-0.5">×</span>
        </div>
        <span className="text-[11px] text-muted-foreground">Gaussian pyramid & MTF restored</span>
      </div>
    </div>
  )
}

export function OverlapCard({ sourcePct, refPct }: { sourcePct: number; refPct?: number }) {
  const hasRef = refPct !== undefined && typeof refPct === 'number'
  return (
    <div className="stage-detail stat-card-hero">
      <div className="flex items-center justify-between pb-2 border-b border-border/60">
        <div className="flex items-center gap-2">
          <div className="size-8 rounded-xl bg-emerald-500/10 border border-emerald-500/20 grid place-items-center text-emerald-400">
            <Percent size={16} />
          </div>
          <div>
            <span className="text-xs font-bold uppercase tracking-wider text-foreground/90 block">Surface Overlap Extent</span>
            <span className="text-[11px] text-muted-foreground">Georeferenced bounding box intersection</span>
          </div>
        </div>
        <Badge variant="outline" className="font-mono text-[10px] bg-emerald-500/10 border-emerald-500/30 text-emerald-400 font-semibold">
          Overlap BBox
        </Badge>
      </div>
      <div className="pt-3">
        <div className="flex items-baseline justify-between">
          <div className="flex items-baseline gap-2">
            <span className="font-mono text-3xl font-extrabold tracking-tight text-foreground">{sourcePct.toFixed(1)}%</span>
            <span className="text-xs font-semibold text-muted-foreground">Source Area</span>
          </div>
          {hasRef && (
            <div className="text-right">
              <span className="font-mono text-lg font-bold text-foreground/80">{refPct.toFixed(1)}%</span>
              <span className="text-[11px] text-muted-foreground block">Ref Area</span>
            </div>
          )}
        </div>
        <div className={`grid ${hasRef ? 'grid-cols-2' : 'grid-cols-1'} gap-2 mt-2.5`}>
          <div className="w-full bg-secondary rounded-full h-2 overflow-hidden border border-border/60">
            <div className="bg-emerald-500 h-full rounded-full transition-all duration-500" style={{ width: `${Math.min(100, Math.max(5, sourcePct))}%` }} />
          </div>
          {hasRef && (
            <div className="w-full bg-secondary rounded-full h-2 overflow-hidden border border-border/60">
              <div className="bg-blue-500 h-full rounded-full transition-all duration-500" style={{ width: `${Math.min(100, Math.max(5, refPct))}%` }} />
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export function CoarseShiftCard({ dx, dy }: { dx: number; dy: number }) {
  const dist = Math.hypot(dx, dy)
  return (
    <div className="stage-detail stat-card-hero">
      <div className="flex items-center justify-between pb-2 border-b border-border/60">
        <div className="flex items-center gap-2">
          <div className="size-8 rounded-xl bg-amber-500/10 border border-amber-500/20 grid place-items-center text-amber-400">
            <Move size={16} />
          </div>
          <div>
            <span className="text-xs font-bold uppercase tracking-wider text-foreground/90 block">Coarse Shift Vector</span>
            <span className="text-[11px] text-muted-foreground">Global translation solved via Fourier / LoFTR</span>
          </div>
        </div>
        <span className="font-mono text-xs font-bold text-amber-400 bg-amber-500/15 border border-amber-500/25 px-2 py-0.5 rounded-md">
          {dist.toFixed(1)} px Total
        </span>
      </div>
      <div className="grid grid-cols-2 gap-3 pt-3">
        <div className="rounded-xl border border-border/80 bg-secondary/60 p-2.5">
          <div className="flex items-center justify-between">
            <span className="text-[10px] font-bold uppercase tracking-wider text-muted-foreground">ΔX (Horizontal)</span>
            <span className="text-[10px] font-semibold text-amber-400 bg-amber-500/15 px-1.5 py-0.5 rounded font-mono">
              {dx < 0 ? '← West' : '→ East'}
            </span>
          </div>
          <div className="flex items-baseline gap-1 mt-1.5 font-mono">
            <span className="text-2xl font-extrabold text-foreground">{dx >= 0 ? `+${dx.toFixed(1)}` : dx.toFixed(1)}</span>
            <span className="text-xs font-bold text-muted-foreground">px</span>
          </div>
        </div>
        <div className="rounded-xl border border-border/80 bg-secondary/60 p-2.5">
          <div className="flex items-center justify-between">
            <span className="text-[10px] font-bold uppercase tracking-wider text-muted-foreground">ΔY (Vertical)</span>
            <span className="text-[10px] font-semibold text-amber-400 bg-amber-500/15 px-1.5 py-0.5 rounded font-mono">
              {dy < 0 ? '↑ North' : '↓ South'}
            </span>
          </div>
          <div className="flex items-baseline gap-1 mt-1.5 font-mono">
            <span className="text-2xl font-extrabold text-foreground">{dy >= 0 ? `+${dy.toFixed(1)}` : dy.toFixed(1)}</span>
            <span className="text-xs font-bold text-muted-foreground">px</span>
          </div>
        </div>
      </div>
    </div>
  )
}

export function ConfidenceCard({ confidence }: { confidence: number }) {
  const isHigh = confidence >= 0.75
  const isMed = confidence >= 0.5 && confidence < 0.75
  const badgeText = isHigh ? 'HIGH CONFIDENCE' : isMed ? 'MODERATE CERTAINTY' : 'LOW CERTAINTY'
  const badgeColor = isHigh ? 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30' : isMed ? 'bg-sky-500/15 text-sky-400 border-sky-500/30' : 'bg-amber-500/15 text-amber-400 border-amber-500/30'
  const barColor = isHigh ? 'from-emerald-500 to-teal-400' : isMed ? 'from-sky-500 to-cyan-400' : 'from-amber-500 to-orange-400'

  return (
    <div className="stage-detail stat-card-hero">
      <div className="flex items-center justify-between pb-2 border-b border-border/60">
        <div className="flex items-center gap-2">
          <div className="size-8 rounded-xl bg-emerald-500/10 border border-emerald-500/20 grid place-items-center text-emerald-400">
            <ShieldCheck size={16} />
          </div>
          <div>
            <span className="text-xs font-bold uppercase tracking-wider text-foreground/90 block">Alignment Confidence</span>
            <span className="text-[11px] text-muted-foreground">Cross-correlation peak certainty</span>
          </div>
        </div>
        <Badge variant="outline" className={`font-mono text-[10px] font-bold ${badgeColor}`}>
          {badgeText}
        </Badge>
      </div>
      <div className="pt-3">
        <div className="flex items-baseline justify-between">
          <span className="font-mono text-3xl font-extrabold tracking-tight text-foreground">{(confidence * 100).toFixed(1)}%</span>
          <span className="font-mono text-xs text-muted-foreground">Score: {confidence.toFixed(3)}</span>
        </div>
        <div className="w-full bg-secondary rounded-full h-2 mt-2.5 overflow-hidden border border-border/60">
          <div
            className={`bg-gradient-to-r ${barColor} h-full rounded-full transition-all duration-500`}
            style={{ width: `${Math.min(100, Math.max(5, confidence * 100))}%` }}
          />
        </div>
      </div>
    </div>
  )
}

export function MethodCard({ method }: { method: string }) {
  const friendlyName = method === 'global_thumbnail_loftr' ? 'LoFTR Cross-Attention (Global)' :
    method === 'loftr' ? 'LoFTR Detector-Free Transformer' :
    method === 'fft' ? 'Fourier-Mellin Phase Correlation' :
    method === 'crater' ? 'Crater Rim Constellation Matcher' :
    method.replaceAll('_', ' ').toUpperCase()

  return (
    <div className="stage-detail stat-card-hero">
      <div className="flex items-center justify-between pb-2 border-b border-border/60">
        <div className="flex items-center gap-2">
          <div className="size-8 rounded-xl bg-indigo-500/10 border border-indigo-500/20 grid place-items-center text-indigo-400">
            <Cpu size={16} />
          </div>
          <div>
            <span className="text-xs font-bold uppercase tracking-wider text-foreground/90 block">Matching Algorithm</span>
            <span className="text-[11px] text-muted-foreground">Active geometric correspondence solver</span>
          </div>
        </div>
        <Badge variant="outline" className="font-mono text-[10px] bg-indigo-500/10 border-indigo-500/30 text-indigo-400 font-semibold">
          Active Solver
        </Badge>
      </div>
      <div className="pt-3 flex items-baseline justify-between">
        <span className="font-mono text-xl font-bold text-foreground">{friendlyName}</span>
        <span className="text-[11px] text-muted-foreground">Detector-free</span>
      </div>
    </div>
  )
}

export function RasterShapeCard({ shape, width, height, orientation }: { shape?: [number, number] | string; width?: number; height?: number; orientation?: string }) {
  let w = width ?? 0
  let h = height ?? 0
  if (Array.isArray(shape) && shape.length >= 2) {
    h = Number(shape[0])
    w = Number(shape[1])
  }
  const mp = (w * h) / 1e6

  return (
    <div className="stage-detail stat-card-hero">
      <div className="flex items-center justify-between pb-2 border-b border-border/60">
        <div className="flex items-center gap-2">
          <div className="size-8 rounded-xl bg-sky-500/10 border border-sky-500/20 grid place-items-center text-sky-400">
            <Scan size={16} />
          </div>
          <div>
            <span className="text-xs font-bold uppercase tracking-wider text-foreground/90 block">Raster Matrix Dimensions</span>
            <span className="text-[11px] text-muted-foreground">Canonical pixel grid geometry</span>
          </div>
        </div>
        <Badge variant="outline" className="font-mono text-[10px] bg-sky-500/10 border-sky-500/30 text-sky-400 font-semibold">
          {mp > 0 ? `${mp.toFixed(2)} MP` : 'Grid Geometry'}
        </Badge>
      </div>
      <div className="pt-3 flex items-baseline justify-between">
        <div className="flex items-baseline gap-1.5 font-mono">
          <span className="text-2xl font-extrabold text-foreground">{h.toLocaleString()}</span>
          <span className="text-sm font-bold text-muted-foreground">×</span>
          <span className="text-2xl font-extrabold text-foreground">{w.toLocaleString()}</span>
          <span className="text-xs font-medium text-muted-foreground ml-1">px</span>
        </div>
        <span className="text-[11px] text-muted-foreground font-mono capitalize">
          {orientation ? `Orientation: ${orientation}` : '2D Matrix Grid'}
        </span>
      </div>
    </div>
  )
}

export function DriftModelCard({ model }: { model: Record<string, unknown> }) {
  const dySlope = typeof model.dy_slope === 'number' ? model.dy_slope : null
  const dyIntercept = typeof model.dy_intercept === 'number' ? model.dy_intercept : null
  const dxSlope = typeof model.dx_slope === 'number' ? model.dx_slope : null
  const dxIntercept = typeof model.dx_intercept === 'number' ? model.dx_intercept : null

  return (
    <div className="stage-detail span-full stat-card-hero">
      <div className="flex flex-wrap items-center justify-between gap-2 pb-2.5 border-b border-border/70">
        <div className="flex items-center gap-2">
          <div className="size-8 rounded-xl bg-amber-500/10 border border-amber-500/20 grid place-items-center text-amber-400">
            <Layers3 size={16} />
          </div>
          <div>
            <span className="text-xs font-bold uppercase tracking-wider text-foreground/90 block">
              Scanline Orbital Jitter & Drift Model
            </span>
            <span className="text-[11px] text-muted-foreground">
              Pushbroom line-scan dynamic sensor trajectory compensation
            </span>
          </div>
        </div>
        <Badge variant="outline" className="font-mono text-[10px] bg-amber-500/10 border-amber-500/30 text-amber-400 font-semibold">
          Polynomial Layer 2
        </Badge>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 pt-3">
        <div className="rounded-xl border border-border/80 bg-secondary/60 p-3 flex flex-col justify-between">
          <div className="flex items-center justify-between">
            <span className="text-[10px] font-bold uppercase tracking-wider text-muted-foreground">Along-Track Line Drift (ΔY)</span>
            <span className="text-[10px] font-mono text-amber-400 bg-amber-500/15 px-1.5 py-0.5 rounded font-semibold">Pitch & Velocity</span>
          </div>
          <div className="mt-2 grid grid-cols-2 gap-2">
            <div>
              <span className="text-[10px] text-muted-foreground block font-medium">Drift Rate:</span>
              <span className="font-mono text-sm font-bold text-foreground">{dySlope !== null ? `${dySlope >= 0 ? '+' : ''}${dySlope.toFixed(4)}` : '—'}</span>
              <span className="text-[10px] text-muted-foreground ml-1">px/line</span>
            </div>
            <div>
              <span className="text-[10px] text-muted-foreground block font-medium">Base Offset:</span>
              <span className="font-mono text-sm font-bold text-foreground">{dyIntercept !== null ? dyIntercept.toFixed(1) : '—'}</span>
              <span className="text-[10px] text-muted-foreground ml-1">px</span>
            </div>
          </div>
        </div>

        <div className="rounded-xl border border-border/80 bg-secondary/60 p-3 flex flex-col justify-between">
          <div className="flex items-center justify-between">
            <span className="text-[10px] font-bold uppercase tracking-wider text-muted-foreground">Cross-Track Scan Drift (ΔX)</span>
            <span className="text-[10px] font-mono text-cyan-400 bg-cyan-500/15 px-1.5 py-0.5 rounded font-semibold">Roll & Pointing</span>
          </div>
          <div className="mt-2 grid grid-cols-2 gap-2">
            <div>
              <span className="text-[10px] text-muted-foreground block font-medium">Drift Rate:</span>
              <span className="font-mono text-sm font-bold text-foreground">{dxSlope !== null ? `${dxSlope >= 0 ? '+' : ''}${dxSlope.toFixed(4)}` : '—'}</span>
              <span className="text-[10px] text-muted-foreground ml-1">px/line</span>
            </div>
            <div>
              <span className="text-[10px] text-muted-foreground block font-medium">Base Offset:</span>
              <span className="font-mono text-sm font-bold text-foreground">{dxIntercept !== null ? dxIntercept.toFixed(1) : '—'}</span>
              <span className="text-[10px] text-muted-foreground ml-1">px</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export function FeatureFunnelCard({
  candidateMatches,
  eccMatches,
  inlierMatches,
  cells,
  entropy,
}: {
  candidateMatches: number
  eccMatches?: number
  inlierMatches?: number
  cells?: number
  entropy?: number
}) {
  const hasEcc = eccMatches !== undefined
  const hasInliers = inlierMatches !== undefined
  const eccPct = hasEcc && candidateMatches > 0 ? (eccMatches / candidateMatches) * 100 : null
  const inlierPct = hasInliers && hasEcc && eccMatches > 0 ? (inlierMatches / eccMatches) * 100 : null

  return (
    <div className="stage-detail span-full stat-card-hero">
      <div className="flex flex-wrap items-center justify-between gap-2 pb-2.5 border-b border-border/70">
        <div className="flex items-center gap-2">
          <div className="size-8 rounded-xl bg-purple-500/10 border border-purple-500/20 grid place-items-center text-purple-400">
            <Sparkles size={16} />
          </div>
          <div>
            <span className="text-xs font-bold uppercase tracking-wider text-foreground/90 block">
              Feature Correspondence & Verification Pipeline
            </span>
            <span className="text-[11px] text-muted-foreground">
              Candidate attention pairs → continuous sub-pixel optimization → geometric RANSAC consensus
            </span>
          </div>
        </div>
        {entropy !== undefined && (
          <div className="flex items-center gap-2 font-mono text-[11px] text-muted-foreground bg-secondary px-2.5 py-1 rounded-md border border-border/60">
            <span>Entropy (H): <strong className="text-foreground">{entropy.toFixed(3)}</strong></span>
            {cells !== undefined && <span>· <strong className="text-foreground">{cells}</strong> Cells</span>}
          </div>
        )}
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 pt-3">
        <div className="rounded-xl border border-border/80 bg-secondary/60 p-3">
          <div className="flex items-center justify-between">
            <span className="text-[10px] font-bold uppercase tracking-wider text-muted-foreground">1. Candidate Matches</span>
            <span className="text-[10px] font-mono text-muted-foreground bg-secondary px-1.5 py-0.5 rounded border border-border/60">LoFTR</span>
          </div>
          <span className="font-mono text-2xl font-extrabold text-foreground mt-1 block">
            {candidateMatches.toLocaleString()}
          </span>
          <span className="text-[11px] text-muted-foreground">Cross-attention tie-point pairs</span>
        </div>

        {hasEcc && (
          <div className="rounded-xl border border-purple-500/25 bg-purple-500/10 p-3">
            <div className="flex items-center justify-between">
              <span className="text-[10px] font-bold uppercase tracking-wider text-purple-400">2. ECC Refined</span>
              {eccPct !== null && (
                <span className="text-[10px] font-bold text-purple-300 bg-purple-500/20 px-1.5 py-0.5 rounded font-mono">
                  {eccPct.toFixed(1)}% Refined
                </span>
              )}
            </div>
            <span className="font-mono text-2xl font-extrabold text-purple-300 mt-1 block">
              {eccMatches.toLocaleString()}
            </span>
            <span className="text-[11px] text-purple-400/80">&lt; 0.2 px Gauss-Newton precision</span>
          </div>
        )}

        {hasInliers && (
          <div className="rounded-xl border border-emerald-500/25 bg-emerald-500/10 p-3">
            <div className="flex items-center justify-between">
              <span className="text-[10px] font-bold uppercase tracking-wider text-emerald-400">3. Verified Inliers</span>
              {inlierPct !== null && (
                <span className="text-[10px] font-bold text-emerald-300 bg-emerald-500/20 px-1.5 py-0.5 rounded font-mono">
                  {inlierPct.toFixed(1)}% Inliers
                </span>
              )}
            </div>
            <span className="font-mono text-2xl font-extrabold text-emerald-300 mt-1 block">
              {inlierMatches.toLocaleString()}
            </span>
            <span className="text-[11px] text-emerald-400/80">RANSAC robust consensus set</span>
          </div>
        )}
      </div>
    </div>
  )
}

export function GenericSmartCard({ label, value }: { label: string; value: unknown }) {
  const formattedLabel = label.replaceAll('_', ' ')
  let unit = ''
  let displayVal = ''
  const isNumber = typeof value === 'number'

  if (isNumber) {
    if (label.toLowerCase().includes('px')) {
      unit = 'px'
      displayVal = Number.isInteger(value) ? value.toLocaleString() : value.toFixed(3)
    } else if (label.toLowerCase().includes('pct') || label.toLowerCase().includes('percent')) {
      unit = '%'
      displayVal = value.toFixed(1)
    } else if (label.toLowerCase().includes('sec') || label.toLowerCase().includes('time')) {
      unit = 's'
      displayVal = value.toFixed(2)
    } else {
      displayVal = Number.isInteger(value) ? value.toLocaleString() : value.toFixed(3)
    }
  } else if (typeof value === 'boolean') {
    displayVal = value ? 'Yes' : 'No'
  } else {
    displayVal = String(value)
  }

  return (
    <div className="stage-detail stat-card-hero">
      <div className="flex items-center justify-between pb-2 border-b border-border/60">
        <span className="text-xs font-bold uppercase tracking-wider text-foreground/90 truncate">{formattedLabel}</span>
        {unit && (
          <Badge variant="outline" className="font-mono text-[10px] bg-secondary border-border text-foreground/80">
            {unit}
          </Badge>
        )}
      </div>
      <div className="pt-3 flex items-baseline justify-between">
        <div className="flex items-baseline gap-1.5 font-mono">
          <span className="text-xl font-bold text-foreground break-all">{displayVal}</span>
          {unit && <span className="text-xs font-medium text-muted-foreground">{unit}</span>}
        </div>
      </div>
    </div>
  )
}

export function StageDetails({ details }: { details: Record<string, unknown> }) {
  const ignored = new Set(['previews', 'preview', 'preview_dimensions', 'files_created', 'metrics'])
  const activeKeys = new Set(Object.keys(details).filter(k => !ignored.has(k) && details[k] !== null && details[k] !== undefined))
  if (!activeKeys.size) return null

  // Flags for rich cards
  const hasSourceFile = activeKeys.has('source_file')
  const hasRefFile = activeKeys.has('reference_file')
  const showFileBanner = hasSourceFile || hasRefFile

  const gsdVal = typeof details.gsd_m === 'number' ? details.gsd_m : typeof details.gsd === 'number' ? details.gsd : null
  const scaleRatioVal = typeof details.scale_ratio === 'number' ? details.scale_ratio : null

  const sourceOverlap = typeof details.source_overlap_pct === 'number' ? details.source_overlap_pct : typeof details.overlap_pct === 'number' ? details.overlap_pct : null
  const refOverlap = typeof details.reference_overlap_pct === 'number' ? details.reference_overlap_pct : undefined

  const dxVal = typeof details.dx_px === 'number' ? details.dx_px : typeof details.coarse_dx === 'number' ? details.coarse_dx : null
  const dyVal = typeof details.dy_px === 'number' ? details.dy_px : typeof details.coarse_dy === 'number' ? details.coarse_dy : null
  const showCoarseShift = dxVal !== null && dyVal !== null

  const confVal = typeof details.confidence === 'number' ? details.confidence : typeof details.confidence_score === 'number' ? details.confidence_score : null
  const methodVal = typeof details.method_used === 'string' ? details.method_used : typeof details.method === 'string' ? details.method : typeof details.solver === 'string' ? details.solver : null

  const shapeVal = details.harmonized_shape_px
  const widthVal = typeof details.width_px === 'number' ? details.width_px : undefined
  const heightVal = typeof details.height_px === 'number' ? details.height_px : undefined
  const showShape = (shapeVal && Array.isArray(shapeVal)) || (widthVal !== undefined && heightVal !== undefined)

  const candidateMatches = typeof details.candidate_matches === 'number' ? details.candidate_matches : null
  const eccMatches = typeof details.ecc_refined_matches === 'number' ? details.ecc_refined_matches : undefined
  const inlierMatches = typeof details.inlier_matches === 'number' ? details.inlier_matches : undefined
  const spatialEntropy = typeof details.spatial_entropy === 'number' ? details.spatial_entropy : undefined
  const populatedCells = typeof details.populated_cells === 'number' ? details.populated_cells : undefined
  const showFeatureFunnel = candidateMatches !== null

  const driftModel = details.drift_model && typeof details.drift_model === 'object' ? details.drift_model as Record<string, unknown> : null
  const transformStats = details.transform_stats && typeof details.transform_stats === 'object' ? details.transform_stats as Record<string, unknown> : null
  const crsVal = typeof details.crs === 'string' ? details.crs : null

  // Mark all consumed keys so they don't get rendered redundantly
  const handled = new Set<string>()
  if (showFileBanner) { handled.add('source_file'); handled.add('reference_file') }
  if (gsdVal !== null) { handled.add('gsd_m'); handled.add('gsd') }
  if (scaleRatioVal !== null) { handled.add('scale_ratio') }
  if (sourceOverlap !== null) { handled.add('source_overlap_pct'); handled.add('reference_overlap_pct'); handled.add('overlap_pct') }
  if (showCoarseShift) { handled.add('dx_px'); handled.add('dy_px'); handled.add('coarse_dx'); handled.add('coarse_dy') }
  if (confVal !== null) { handled.add('confidence'); handled.add('confidence_score') }
  if (methodVal !== null) { handled.add('method_used'); handled.add('method'); handled.add('solver') }
  if (showShape) { handled.add('harmonized_shape_px'); handled.add('width_px'); handled.add('height_px'); handled.add('orientation') }
  if (showFeatureFunnel) {
    handled.add('candidate_matches'); handled.add('ecc_refined_matches'); handled.add('inlier_matches')
    handled.add('populated_cells'); handled.add('spatial_entropy')
  }
  if (driftModel) { handled.add('drift_model') }
  if (transformStats) { handled.add('transform_stats') }
  if (crsVal) { handled.add('crs') }

  const remaining = Object.entries(details).filter(([k, v]) => !ignored.has(k) && !handled.has(k) && v !== null && v !== undefined)

  return (
    <div className="stage-detail-grid">
      {/* 1. File Pipeline Strip */}
      {showFileBanner && (
        <FilePairBanner
          source={details.source_file ? String(details.source_file) : undefined}
          reference={details.reference_file ? String(details.reference_file) : undefined}
        />
      )}

      {/* 2. Primary Mathematical & Physical Metrics */}
      {gsdVal !== null && <GsdCard gsd={gsdVal} />}
      {scaleRatioVal !== null && <ScaleRatioCard ratio={scaleRatioVal} />}
      {sourceOverlap !== null && <OverlapCard sourcePct={sourceOverlap} refPct={refOverlap} />}
      {showShape && (
        <RasterShapeCard
          shape={shapeVal as [number, number]}
          width={widthVal}
          height={heightVal}
          orientation={details.orientation ? String(details.orientation) : undefined}
        />
      )}

      {/* 3. Coarse Shift & Confidence Metrics */}
      {showCoarseShift && <CoarseShiftCard dx={dxVal} dy={dyVal} />}
      {confVal !== null && <ConfidenceCard confidence={confVal} />}
      {methodVal !== null && <MethodCard method={methodVal} />}

      {/* 4. Orbital Drift Model */}
      {driftModel && <DriftModelCard model={driftModel} />}

      {/* 5. Feature Matching Pipeline Funnel */}
      {showFeatureFunnel && (
        <FeatureFunnelCard
          candidateMatches={candidateMatches}
          eccMatches={eccMatches}
          inlierMatches={inlierMatches}
          cells={populatedCells}
          entropy={spatialEntropy}
        />
      )}

      {/* 6. Fitted 3-Layer Hybrid Transform Stats */}
      {transformStats && <TransformStatsView stats={transformStats} />}

      {/* 7. Coordinate Reference System (CRS) */}
      {crsVal && <CrsView crs={crsVal} />}

      {/* 8. Fallback for any unmapped diagnostic metrics */}
      {remaining.map(([k, v]) => {
        if (v && typeof v === 'object' && !Array.isArray(v)) {
          const subEntries = Object.entries(v as Record<string, unknown>).filter(([_, val]) => val !== null && val !== undefined)
          if (!subEntries.length) return null
          return (
            <div className="stage-detail span-full stat-card-hero" key={k}>
              <span className="text-xs font-bold uppercase tracking-wider text-foreground/90 block pb-2 border-b border-border/60">
                {k.replaceAll('_', ' ')}
              </span>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 pt-2">
                {subEntries.map(([subK, subV]) => (
                  <div key={subK} className="rounded-lg border border-border/80 bg-secondary/60 p-2 flex flex-col">
                    <span className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">{subK.replaceAll('_', ' ')}</span>
                    <strong className="mt-0.5 break-all font-mono text-xs font-semibold text-foreground">{displayDetail(subV)}</strong>
                  </div>
                ))}
              </div>
            </div>
          )
        }
        return <GenericSmartCard key={k} label={k} value={v} />
      })}
    </div>
  )
}

function displayDetail(value: unknown): ReactNode {
  if (typeof value === 'number') return Number.isInteger(value) ? value.toLocaleString() : value.toFixed(3)
  if (typeof value === 'boolean') return value ? 'Yes' : 'No'
  if (typeof value === 'string') return value
  if (Array.isArray(value)) return value.map((item) => typeof item === 'number' ? item.toLocaleString() : String(item)).join(' × ')
  if (value && typeof value === 'object') {
    return Object.entries(value as Record<string, unknown>).map(([key, item]) => `${key.replaceAll('_', ' ')}: ${displayDetail(item)}`).join(' · ')
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
              {formatPreviewTitle(current.file_name)}
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
              title="Close full-screen preview"
              aria-label="Close full-screen preview"
            >
              <Mark name="x" size={18} />
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
              alt={formatPreviewTitle(current.file_name)}
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
      </div>
    </div>
  )
}
