import type {
  ArtifactList,
  HealthStatus,
  RegistrationConfig,
  RegistrationJob,
} from './types'

const configuredBase = import.meta.env.VITE_API_BASE_URL?.replace(/\/$/, '')
export const API_BASE = configuredBase || '/api/v1'

function errorMessage(payload: Record<string, unknown>, status: number): string {
  const detail = payload.detail
  if (typeof detail === 'string') return detail
  if (Array.isArray(detail)) {
    const messages = detail.map((item) => {
      if (!item || typeof item !== 'object') return ''
      const error = item as { loc?: unknown[]; msg?: unknown }
      const field = Array.isArray(error.loc) ? error.loc.slice(-1)[0] : undefined
      return [field ? String(field) : '', typeof error.msg === 'string' ? error.msg : ''].filter(Boolean).join(': ')
    }).filter(Boolean)
    if (messages.length) return messages.join(' · ')
  }
  return `Request failed (${status})`
}

async function readJson<T>(response: Response): Promise<T> {
  const payload = await response.json().catch(() => ({}))
  if (!response.ok) {
    throw new Error(errorMessage(payload as Record<string, unknown>, response.status))
  }
  return payload as T
}

export async function fetchHealth(): Promise<HealthStatus> {
  const healthUrl = configuredBase?.startsWith('http') ? `${new URL(configuredBase).origin}/health` : '/health'
  const response = await fetch(healthUrl, { cache: 'no-store' })
  return readJson<HealthStatus>(response)
}

export async function fetchJobs(): Promise<RegistrationJob[]> {
  const response = await fetch(`${API_BASE}/jobs?limit=8`, { cache: 'no-store' })
  const payload = await readJson<{ jobs: RegistrationJob[] }>(response)
  return payload.jobs
}

export async function fetchJob(jobId: string): Promise<RegistrationJob> {
  const response = await fetch(`${API_BASE}/jobs/${jobId}`, { cache: 'no-store' })
  return readJson<RegistrationJob>(response)
}

export async function fetchArtifacts(jobId: string): Promise<ArtifactList> {
  const response = await fetch(`${API_BASE}/jobs/${jobId}/artifacts`, { cache: 'no-store' })
  return readJson<ArtifactList>(response)
}

export function apiUrl(path: string): string {
  if (/^https?:\/\//.test(path)) return path
  if (path === API_BASE || path.startsWith(`${API_BASE}/`)) return path
  if (configuredBase?.startsWith('http') && path.startsWith('/api/v1/')) {
    return `${new URL(configuredBase).origin}${path}`
  }
  return `${API_BASE}${path.startsWith('/') ? path : `/${path}`}`
}

export function createJob(
  config: RegistrationConfig,
  onProgress: (progress: number) => void,
): Promise<RegistrationJob> {
  const body = new FormData()
  if (!config.source || !config.reference) return Promise.reject(new Error('Choose both a source and a reference product.'))

  body.append('source_file', config.source)
  body.append('reference_file', config.reference)
  config.sourceSidecars.forEach((file) => body.append('source_sidecars', file))
  config.referenceSidecars.forEach((file) => body.append('reference_sidecars', file))
  body.append('sensor_src', config.sensorSrc)
  body.append('sensor_ref', config.sensorRef)
  body.append('method', config.method)
  body.append('structural_method', config.structuralMethod)
  body.append('coarse_method', config.coarseMethod)
  body.append('poly_degree', String(config.polyDegree))
  body.append('tps_smoothing', String(config.tpsSmoothing))
  body.append('ransac_threshold', String(config.ransacThreshold))
  body.append('warp_order', String(config.warpOrder))
  body.append('wac_band', String(config.wacBand))
  body.append('grid_rows', String(config.gridRows))
  body.append('grid_cols', String(config.gridCols))
  body.append('export_native', String(config.exportNative))

  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest()
    request.open('POST', `${API_BASE}/jobs`)
    request.upload.onprogress = (event) => {
      if (event.lengthComputable) onProgress(Math.round((event.loaded / event.total) * 100))
    }
    request.onerror = () => reject(new Error('Could not reach the local registration API. Check that the backend is running.'))
    request.onabort = () => reject(new Error('Upload was cancelled.'))
    request.onload = () => {
      let payload: Record<string, unknown> = {}
      try { payload = JSON.parse(request.responseText) as Record<string, unknown> } catch {}
      if (request.status < 200 || request.status >= 300) {
        reject(new Error(errorMessage(payload, request.status).replace(/^Request failed/, 'Job could not be created')))
        return
      }
      resolve(payload as unknown as RegistrationJob)
    }
    request.send(body)
  })
}
