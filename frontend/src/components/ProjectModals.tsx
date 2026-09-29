import { useEffect, useRef, useState } from 'react'
import {
  AlertCircle,
  ArrowRight,
  Calendar,
  Check,
  Clock3,
  Copy,
  Database,
  Folder,
  FolderOpen,
  FolderPlus,
  Loader2,
  RefreshCw,
  Search,
  Sparkles,
  X,
} from 'lucide-react'
import type { RegistrationJob } from '../types'
import { fetchJobs } from '../api'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'

interface NewProjectModalProps {
  isOpen: boolean
  onClose: () => void
  onConfirm: (projectName: string) => void
  currentJob: RegistrationJob | null
}

export function NewProjectModal({
  isOpen,
  onClose,
  onConfirm,
  currentJob,
}: NewProjectModalProps) {
  const [name, setName] = useState('')
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    if (isOpen) {
      setName('')
      setTimeout(() => inputRef.current?.focus(), 50)
    }
  }, [isOpen])

  if (!isOpen) return null

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    onConfirm(name.trim())
    onClose()
  }

  return (
    <div
      className="project-modal-backdrop"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose()
      }}
      role="dialog"
      aria-modal="true"
      aria-labelledby="new-project-title"
    >
      <div className="project-modal-card new-project-card">
        <div className="project-modal-header">
          <div className="project-modal-title-group">
            <div className="project-modal-icon-badge">
              <FolderPlus size={22} className="text-primary" />
            </div>
            <div>
              <h2 id="new-project-title" className="project-modal-title">
                Start New Project
              </h2>
              <p className="project-modal-subtitle">
                Initialize a fresh registration session with clean inputs.
              </p>
            </div>
          </div>
          <button
            type="button"
            className="project-modal-close"
            onClick={onClose}
            aria-label="Close dialog"
          >
            <X size={18} />
          </button>
        </div>

        <form onSubmit={handleSubmit} className="project-modal-body">
          {currentJob && (
            <div className="project-modal-notice">
              <AlertCircle size={16} className="text-amber-500 shrink-0 mt-0.5" />
              <div className="text-xs text-muted-foreground leading-relaxed">
                Active run{' '}
                <strong className="text-foreground">
                  {currentJob.project_name || currentJob.source_filename}
                </strong>{' '}
                is currently loaded. Starting a new project clears your active workspace so you can configure a new pair. Your past project remains safely stored in history.
              </div>
            </div>
          )}

          <div className="project-field-group">
            <label htmlFor="project-name-input" className="project-field-label">
              Project Name
            </label>
            <input
              id="project-name-input"
              ref={inputRef}
              type="text"
              className="project-text-input"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="e.g., Shackleton Crater High-Res Run 01"
              maxLength={128}
            />
            <p className="project-field-hint">
              Optional but recommended. If left empty, your project will be automatically named after the source raster filename.
            </p>
          </div>

          <div className="project-modal-footer">
            <Button type="button" variant="outline" onClick={onClose}>
              Cancel
            </Button>
            <Button type="submit" variant="default" className="gap-2">
              <FolderPlus size={16} />
              Create & Open Project
            </Button>
          </div>
        </form>
      </div>
    </div>
  )
}

interface ProjectsModalProps {
  isOpen: boolean
  onClose: () => void
  onSelectProject: (job: RegistrationJob) => void
  currentJobId?: string
}

export function ProjectsModal({
  isOpen,
  onClose,
  onSelectProject,
  currentJobId,
}: ProjectsModalProps) {
  const [jobs, setJobs] = useState<RegistrationJob[]>([])
  const [loading, setLoading] = useState(false)
  const [search, setSearch] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [copiedId, setCopiedId] = useState<string | null>(null)

  const loadProjects = async () => {
    setLoading(true)
    setError(null)
    try {
      const list = await fetchJobs(50)
      setJobs(list)
    } catch (err) {
      setError((err as Error).message || 'Failed to load project history')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    if (isOpen) {
      void loadProjects()
    }
  }, [isOpen])

  if (!isOpen) return null

  const filteredJobs = jobs.filter((j) => {
    const q = search.toLowerCase().trim()
    if (!q) return true
    const pName = (j.project_name || '').toLowerCase()
    const sName = (j.source_filename || '').toLowerCase()
    const rName = (j.reference_filename || '').toLowerCase()
    const sSrc = (j.sensor_src || '').toLowerCase()
    const sRef = (j.sensor_ref || '').toLowerCase()
    const status = (j.status || '').toLowerCase()
    return (
      pName.includes(q) ||
      sName.includes(q) ||
      rName.includes(q) ||
      sSrc.includes(q) ||
      sRef.includes(q) ||
      status.includes(q)
    )
  })

  function formatDate(isoDate: string): string {
    try {
      const d = new Date(isoDate)
      return d.toLocaleDateString(undefined, {
        month: 'short',
        day: 'numeric',
        year: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
      })
    } catch {
      return isoDate
    }
  }

  return (
    <div
      className="project-modal-backdrop"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose()
      }}
      role="dialog"
      aria-modal="true"
      aria-labelledby="projects-title"
    >
      <div className="project-modal-card projects-history-card">
        <div className="project-modal-header">
          <div className="project-modal-title-group">
            <div className="project-modal-icon-badge">
              <FolderOpen size={22} className="text-primary" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h2 id="projects-title" className="project-modal-title">
                  Saved Projects & History
                </h2>
                <Badge variant="secondary" className="text-xs">
                  {jobs.length} {jobs.length === 1 ? 'run' : 'runs'}
                </Badge>
              </div>
              <p className="project-modal-subtitle flex items-center gap-1.5 mt-0.5">
                <Database size={12} className="text-primary/70 shrink-0" />
                Stored persistently in local SQLite (<code>data/storage/lunar_reg.db</code>) & disk outputs.
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <button
              type="button"
              className="project-modal-icon-btn"
              onClick={loadProjects}
              title="Refresh project list"
              disabled={loading}
            >
              <RefreshCw size={16} className={loading ? 'animate-spin' : ''} />
            </button>
            <button
              type="button"
              className="project-modal-close"
              onClick={onClose}
              aria-label="Close dialog"
            >
              <X size={18} />
            </button>
          </div>
        </div>

        <div className="project-modal-search-row">
          <div className="project-search-wrap">
            <Search size={15} className="project-search-icon" />
            <input
              type="text"
              className="project-search-input"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search by project name, sensor pair, filename, or status..."
            />
            {search && (
              <button
                type="button"
                className="project-search-clear"
                onClick={() => setSearch('')}
                title="Clear search"
              >
                <X size={14} />
              </button>
            )}
          </div>
        </div>

        <div className="projects-list-body">
          {loading && jobs.length === 0 && (
            <div className="projects-loading-state">
              <Loader2 size={32} className="animate-spin text-primary" />
              <p className="text-sm text-muted-foreground mt-3">Loading project history...</p>
            </div>
          )}

          {error && (
            <div className="projects-error-state">
              <AlertCircle size={28} className="text-destructive mb-2" />
              <p className="text-sm font-semibold text-destructive">{error}</p>
              <Button size="sm" variant="outline" className="mt-3" onClick={loadProjects}>
                Retry
              </Button>
            </div>
          )}

          {!loading && !error && filteredJobs.length === 0 && (
            <div className="projects-empty-state">
              <FolderOpen size={36} className="text-muted-foreground/40 mb-2" />
              <p className="text-sm font-medium text-foreground">
                {search ? 'No projects match your search.' : 'No saved projects yet.'}
              </p>
              <p className="text-xs text-muted-foreground mt-1 max-w-sm">
                {search
                  ? 'Try searching for another keyword or clear the search input.'
                  : 'Start a new registration run and it will be saved here automatically with all artifacts and metrics.'}
              </p>
            </div>
          )}

          {!loading &&
            filteredJobs.map((item) => {
              const isActive = item.id === currentJobId
              const metrics = item.metrics as Record<string, unknown> | null | undefined
              const rmsePx = metrics?.rmse_px
              const inliers = metrics?.inlier_match_count

              let statusBadgeClass = 'bg-muted text-muted-foreground'
              let statusLabel: string = item.status
              if (item.status === 'SUCCESS') {
                statusBadgeClass = 'bg-emerald-500/15 text-emerald-600 dark:text-emerald-400 border-emerald-500/30'
              } else if (item.status === 'PROCESSING' || item.status === 'PENDING') {
                statusBadgeClass = 'bg-amber-500/15 text-amber-600 dark:text-amber-400 border-amber-500/30 animate-pulse'
                statusLabel = `${item.progress_pct}% · ${item.current_stage || 'running'}`
              } else if (item.status === 'FAILED') {
                statusBadgeClass = 'bg-destructive/15 text-destructive border-destructive/30'
              }

              return (
                <div
                  key={item.id}
                  className={`project-history-card ${isActive ? 'is-active-project' : ''}`}
                  onClick={() => {
                    onSelectProject(item)
                    onClose()
                  }}
                  role="button"
                  tabIndex={0}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                      e.preventDefault()
                      onSelectProject(item)
                      onClose()
                    }
                  }}
                >
                  <div className="project-history-card-header">
                    <div className="min-w-0 flex-1">
                      <div className="flex items-center gap-2 flex-wrap">
                        <strong className="project-history-name truncate">
                          {item.project_name || item.source_filename}
                        </strong>
                        {isActive && (
                          <Badge variant="outline" className="text-[10px] bg-primary/10 border-primary/40 text-primary py-0">
                            Current Active
                          </Badge>
                        )}
                        <span className={`text-[11px] font-semibold px-2 py-0.5 rounded-full border ${statusBadgeClass}`}>
                          {statusLabel}
                        </span>
                      </div>
                      <div className="project-history-files text-xs text-muted-foreground truncate mt-1">
                        <span className="font-medium text-foreground/80">{item.sensor_src}</span> ({item.source_filename})
                        {' → '}
                        <span className="font-medium text-foreground/80">{item.sensor_ref}</span> ({item.reference_filename})
                      </div>
                    </div>
                    <Button
                      type="button"
                      variant="ghost"
                      size="sm"
                      className="project-open-btn shrink-0 gap-1.5"
                      onClick={(e) => {
                        e.stopPropagation()
                        onSelectProject(item)
                        onClose()
                      }}
                    >
                      <span>Open</span>
                      <ArrowRight size={14} />
                    </Button>
                  </div>

                  <div className="project-history-card-footer">
                    <div className="flex items-center gap-2 flex-wrap">
                      <span className="font-mono text-[10px] text-foreground/80 bg-muted/80 px-1.5 py-0.5 rounded border border-border/50 select-all" title="Unique Project ID (UUID)">
                        ID: {item.id}
                      </span>
                      <button
                        type="button"
                        className="inline-flex items-center gap-1 text-[11px] text-primary hover:underline px-1 py-0.5 cursor-pointer"
                        onClick={(e) => {
                          e.stopPropagation()
                          void navigator.clipboard.writeText(item.id)
                          setCopiedId(item.id)
                          setTimeout(() => setCopiedId(null), 2000)
                        }}
                        title="Copy Project ID"
                      >
                        {copiedId === item.id ? <Check size={11} className="text-emerald-500" /> : <Copy size={11} />}
                        <span>{copiedId === item.id ? 'Copied ID!' : 'Copy ID'}</span>
                      </button>
                      <button
                        type="button"
                        className="inline-flex items-center gap-1 text-[11px] text-muted-foreground hover:text-foreground px-1 py-0.5 cursor-pointer"
                        onClick={(e) => {
                          e.stopPropagation()
                          const folderPath = `C:\\Padhai\\E\\SIH1\\data\\storage\\outputs\\${item.id}`
                          void navigator.clipboard.writeText(folderPath)
                          setCopiedId(`path-${item.id}`)
                          setTimeout(() => setCopiedId(null), 2000)
                        }}
                        title="Copy Windows Folder Path to this project's files"
                      >
                        {copiedId === `path-${item.id}` ? <Check size={11} className="text-emerald-500" /> : <Folder size={11} />}
                        <span>{copiedId === `path-${item.id}` ? 'Copied Path!' : 'Copy Folder Path'}</span>
                      </button>
                    </div>

                    <div className="flex items-center gap-3 text-[11px] text-muted-foreground flex-wrap">
                      <span className="flex items-center gap-1">
                        <Calendar size={12} className="opacity-70" />
                        {formatDate(item.created_at)}
                      </span>
                      {typeof item.wall_time_seconds === 'number' && (
                        <span className="flex items-center gap-1">
                          <Clock3 size={12} className="opacity-70" />
                          {Math.round(item.wall_time_seconds)}s wall time
                        </span>
                      )}
                    </div>

                    {item.status === 'SUCCESS' && (
                      <div className="flex items-center gap-2 flex-wrap">
                        {typeof rmsePx === 'number' && (
                          <span className="project-metric-pill" title="Sub-pixel RMSE">
                            <Sparkles size={11} className="text-primary shrink-0" />
                            RMSE: <strong>{rmsePx.toFixed(3)} px</strong>
                          </span>
                        )}
                        {typeof inliers === 'number' && (
                          <span className="project-metric-pill" title="Inlier tie points">
                            Inliers: <strong>{inliers}</strong>
                          </span>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              )
            })}
        </div>
      </div>
    </div>
  )
}
