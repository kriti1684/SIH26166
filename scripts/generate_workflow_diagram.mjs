import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

// Standalone SVG renderer for the checked-out ChandaShakti workflow.
// Run from any directory with: node scripts/generate_workflow_diagram.mjs
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const output = path.join(root, 'chandashakti-workflow.svg')
const width = 2600
const height = 3120
const shapes = []

const colors = {
  bg: '#091322',
  panel: '#101f32',
  panel2: '#14283e',
  text: '#f3f8ff',
  muted: '#b5c7d9',
  line: '#738ba3',
  cyan: '#56d7d0',
  blue: '#77b9ff',
  violet: '#c0a1ff',
  orange: '#ffbf7a',
  green: '#8ee3aa',
}

function esc(value) {
  return String(value).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;')
}

function rect(x, y, w, h, stroke = colors.line, fill = colors.panel, radius = 22, sw = 2) {
  shapes.push(`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${radius}" fill="${fill}" stroke="${stroke}" stroke-width="${sw}"/>`)
}

function text(x, y, lines, { size = 27, color = colors.text, weight = 400, leading = 36, anchor = 'start' } = {}) {
  const tspans = lines.map((line, i) => `<tspan x="${x}" dy="${i ? leading : 0}">${esc(line)}</tspan>`).join('')
  shapes.push(`<text x="${x}" y="${y}" fill="${color}" font-family="Segoe UI, Arial, sans-serif" font-size="${size}" font-weight="${weight}" text-anchor="${anchor}">${tspans}</text>`)
}

function arrow(points, { color = colors.line, dashed = false, width: sw = 4 } = {}) {
  const d = points.map(([x, y], i) => `${i ? 'L' : 'M'} ${x} ${y}`).join(' ')
  shapes.push(`<path d="${d}" fill="none" stroke="${color}" stroke-width="${sw}" stroke-linecap="round" stroke-linejoin="round" ${dashed ? 'stroke-dasharray="10 10"' : ''} marker-end="url(#arrow)"/>`)
}

function card(x, y, w, h, heading, body, color = colors.blue) {
  rect(x, y, w, h, color, colors.panel2)
  rect(x + 22, y + 24, 8, h - 48, color, color, 4, 0)
  text(x + 54, y + 56, [heading], { size: 30, color, weight: 700 })
  text(x + 54, y + 101, body, { size: 26, color: colors.text, leading: 35 })
}

function stage(y, h, number, title, color) {
  rect(110, y, 2380, h, color, colors.panel, 28, 3)
  rect(135, y + 22, 100, 56, color, color, 15, 0)
  text(185, y + 61, [number], { size: 29, color: colors.bg, weight: 800, anchor: 'middle' })
  text(260, y + 61, [title], { size: 36, color, weight: 750 })
}

function mini(x, y, w, h, title, lines, color) {
  rect(x, y, w, h, color, colors.panel2, 16, 2)
  text(x + 23, y + 39, [title], { size: 27, color, weight: 700 })
  text(x + 23, y + 78, lines, { size: 23, color: colors.text, leading: 31 })
}

function outputLine(y, line, color = colors.muted) {
  text(170, y, [line], { size: 24, color, weight: 500 })
}

shapes.push(`<rect width="${width}" height="${height}" fill="${colors.bg}"/>`)
text(110, 77, ['CHANDASHAKTI — COMPLETE WORKFLOW'], { size: 48, color: colors.text, weight: 800 })
text(110, 122, ['Lunar image co-registration: browser, API, storage, five processing stages, and results'], { size: 25, color: colors.muted })

card(110, 175, 470, 175, '1  User input', ['Source + reference image', 'Sensors and run defaults', 'Optional label/payload files'], colors.cyan)
card(650, 175, 520, 175, '2  React workspace', ['Input / stages 1–5 / output', 'XHR upload with progress', 'Stage panels and downloads'], colors.cyan)
card(1240, 175, 370, 175, '3  Vite :5173', ['Dev proxy for /api', 'and /health'], colors.blue)
card(1680, 175, 810, 175, '4  FastAPI :8000', ['backend.main:app  •  /api/v1/jobs', 'Health, job, artifact, preview, download routes'], colors.blue)
arrow([[580, 262], [645, 262]], { color: colors.cyan })
arrow([[1170, 262], [1235, 262]], { color: colors.cyan })
arrow([[1610, 262], [1675, 262]], { color: colors.blue })
arrow([[2085, 350], [2085, 375], [480, 375], [480, 395]], { color: colors.blue })

card(110, 400, 740, 250, '5  Create job', ['POST multipart source + reference', 'Validate suffix, names, sidecars, size', 'Save uploads; insert PENDING job', 'Return job ID to browser'], colors.blue)
card(925, 400, 740, 250, '6  Local persistence', ['data/storage/uploads/{job_id}/', 'SQLite job row by default', 'Run config, status and stage events', 'data/storage/outputs + previews'], colors.violet)
card(1740, 400, 750, 250, '7  In-process runner', ['FastAPI BackgroundTasks starts worker', 'Worker sets PROCESSING and builds args', 'Calls run_pipeline(args, callback)', 'No Celery or Redis queue'], colors.orange)
arrow([[850, 525], [920, 525]], { color: colors.blue })
arrow([[1665, 525], [1735, 525]], { color: colors.violet })
arrow([[2115, 650], [2115, 698], [1300, 698], [1300, 735]], { color: colors.orange })

stage(750, 350, '01', 'Ingest, georeference, and harmonize', colors.cyan)
mini(160, 840, 640, 170, 'Optional IIRS branch', ['Score first 40 candidate bands;', 'extract selected band.'], colors.cyan)
mini(850, 840, 710, 170, 'Georeference', ['Projected raster: pass through.', 'Raw LRO: corners; WAC de-interleave.', 'Raw CH-2: SPICE GCPs or XML fallback.'], colors.cyan)
mini(1610, 840, 830, 170, 'Shared overlap grid', ['Find geographic intersection.', 'Crop reference and resample source', 'to the harmonized reference grid.'], colors.cyan)
arrow([[800, 925], [845, 925]], { color: colors.cyan })
arrow([[1560, 925], [1605, 925]], { color: colors.cyan })
outputLine(1055, 'Outputs → georeferenced/ and harmonized/ GeoTIFFs; overlap metadata; browser quicklooks', colors.cyan)
arrow([[1300, 1100], [1300, 1135]], { color: colors.cyan })

stage(1150, 250, '02', 'Structural features and coarse alignment', colors.blue)
mini(160, 1235, 720, 112, 'Auto method', ['Try thumbnail LoFTR; if weak, fall back.'], colors.blue)
mini(930, 1235, 720, 112, 'Fallback / explicit methods', ['Strip-wise structural FFT + crater voting.'], colors.blue)
mini(1700, 1235, 740, 112, 'Coarse result', ['Global dx/dy + along-track drift model.'], colors.blue)
arrow([[880, 1290], [925, 1290]], { color: colors.blue })
arrow([[1650, 1290], [1695, 1290]], { color: colors.blue })
outputLine(1380, 'Output → coarse_alignment_result.json; drift positions the next stage’s tile searches', colors.blue)
arrow([[1300, 1400], [1300, 1435]], { color: colors.blue })

stage(1450, 355, '03', 'Tiled matching, ECC refinement, and transform fit', colors.violet)
mini(160, 1540, 510, 170, 'Tiled search', ['Windowed reads on overlap;', 'offset windows using drift.'], colors.violet)
mini(720, 1540, 510, 170, 'Dense matching', ['LoFTR correspondence;', 'filter candidate matches.'], colors.violet)
mini(1280, 1540, 510, 170, 'Subpixel refinement', ['Patch-wise ECC;', 'phase-correlation fallback.'], colors.violet)
mini(1840, 1540, 600, 170, 'Hybrid transform', ['RANSAC affine + scanline drift;', 'TPS when support is sufficient.'], colors.violet)
arrow([[670, 1625], [715, 1625]], { color: colors.violet })
arrow([[1230, 1625], [1275, 1625]], { color: colors.violet })
arrow([[1790, 1625], [1835, 1625]], { color: colors.violet })
outputLine(1764, 'Outputs → candidate_matches.csv, subpixel_tie_points.csv, hybrid_transform_model.json, optional inliers CSV', colors.violet)
arrow([[1300, 1805], [1300, 1840]], { color: colors.violet })

stage(1855, 250, '04', 'Warp registered image', colors.orange)
mini(160, 1940, 1070, 110, 'Block-wise resampling', ['Invert transform; sample source onto reference pixel grid.'], colors.orange)
mini(1280, 1940, 1160, 110, 'Registered products', ['registered_subpixel.tif; optional native-GSD TIFF; quicklook + overlay.'], colors.orange)
arrow([[1230, 1995], [1275, 1995]], { color: colors.orange })
outputLine(2083, 'Output → registered lunar GeoTIFF aligned to the harmonized reference grid', colors.orange)
arrow([[1300, 2105], [1300, 2140]], { color: colors.orange })

stage(2155, 300, '05', 'Verification and scientific verdict', colors.green)
mini(160, 2240, 1070, 120, 'Measure evidence', ['Model residual RMSE, TPS cross-validation, inliers, entropy, and spatial coverage.'], colors.green)
mini(1280, 2240, 1160, 120, 'Summarize', ['Write verdict + metric provenance; generate diagnostic images.'], colors.green)
arrow([[1230, 2300], [1275, 2300]], { color: colors.green })
outputLine(2415, 'Outputs → verification_metrics.json, dashboard, heatmap, side-by-side, and false-color previews', colors.green)
arrow([[1300, 2455], [1300, 2490]], { color: colors.green })

card(110, 2505, 740, 290, '8  Files on disk', ['data/storage/outputs/{job_id}/', 'GeoTIFFs, CSVs, model, diagnostics', 'data/storage/previews/{job_id}/', 'Browser-sized PNG quicklooks'], colors.violet)
card(925, 2505, 740, 290, '9  Worker finalizes job', ['Persist progress + latest stage events', 'SUCCESS or FAILED execution status', 'Attach output links and metrics', 'Science verdict is a separate value'], colors.orange)
card(1740, 2505, 750, 290, '10  Browser receives results', ['GET /jobs/{id} + /artifacts', 'Artifact manifest scans output files', 'UI polls about every 2.2 seconds', 'Preview PNGs; download data products'], colors.cyan)
arrow([[850, 2650], [920, 2650]], { color: colors.violet })
arrow([[1665, 2650], [1735, 2650]], { color: colors.orange })
arrow([[2490, 2650], [2550, 2650], [2550, 145], [910, 145], [910, 170]], { color: colors.cyan, dashed: true, width: 3 })

rect(110, 2850, 2380, 175, colors.line, '#0c1b2e', 20, 2)
text(145, 2898, ['READING NOTES'], { size: 26, color: colors.muted, weight: 700 })
text(145, 2940, ['The browser stores the active job ID in localStorage. Pipeline progress is returned by polling, not push events.'], { size: 23, color: colors.text })
text(145, 2975, ['SUCCESS means the run finished; the verification verdict can still be UNCERTAIN or REJECTED.'], { size: 23, color: colors.text })
text(145, 3010, ['Current tile matcher uses LoFTR for loftr/ensemble; the API accepts crater but that tile mode is unsupported.'], { size: 23, color: colors.text })

const svg = `<?xml version="1.0" encoding="UTF-8"?>\n<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" role="img" aria-label="ChandaShakti end-to-end architecture and pipeline diagram">\n<defs><marker id="arrow" viewBox="0 0 12 12" refX="10" refY="6" markerWidth="12" markerHeight="12" orient="auto-start-reverse"><path d="M 1 1 L 11 6 L 1 11 z" fill="${colors.line}"/></marker></defs>\n${shapes.join('\n')}\n</svg>\n`
fs.writeFileSync(output, svg, 'utf8')
console.log(output)
