# ChandraShakti — Command-Line Interface (CLI) Usage Guide

ChandraShakti can be executed directly from the terminal via either `main.py` or `run_pipeline.py`. Both entrypoints accept identical arguments and invoke the complete 5-phase registration engine.

---

## 🚀 Quickstart Examples

### 1. Chandrayaan-2 OHRC vs. LRO NAC (High-Resolution Panchromatic)
```bash
python main.py \
  --source data/storage/uploads/OHRC_strip.tif \
  --reference data/storage/uploads/LRO_NAC_strip.tif \
  --sensor_src OHRC \
  --sensor_ref NAC \
  --out_dir projects/ohrc_nac_run \
  --grid_size 4 4 \
  --method loftr \
  --structural_method phase_congruency \
  --warp_order 3
```

### 2. Chandrayaan-2 TMC-2 vs. LRO WAC (Regional Multi-Scale Cross-Sensor)
```bash
python main.py \
  --source data/storage/uploads/TMC2_nadir.tif \
  --reference data/storage/uploads/LRO_WAC_mosaic.tif \
  --sensor_src TMC \
  --sensor_ref WAC \
  --out_dir projects/tmc_wac_run \
  --wac_band 7 \
  --grid_size 6 4 \
  --warp_order 3
```

### 3. Chandrayaan-2 IIRS vs. SELENE TC (SWIR Hyperspectral to Optical)
```bash
python main.py \
  --source data/storage/uploads/IIRS_cube.h5 \
  --reference data/storage/uploads/SELENE_TC.tif \
  --sensor_src IIRS \
  --sensor_ref TC \
  --out_dir projects/iirs_selene_run \
  --grid_size 4 4
```

---

## ⚙️ CLI Parameter Reference

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--source`, `-s` | `Path` | *Required* | Path to source observation raster (`.tif`, `.img`, `.h5`, PDS4 XML) |
| `--reference`, `-r` | `Path` | *Required* | Path to fixed reference raster (`.tif`, `.img`, PDS3/PDS4) |
| `--sensor_src` | `str` | `OHRC` | Source sensor type: `OHRC`, `TMC`, `TMC2`, `IIRS` |
| `--sensor_ref` | `str` | `NAC` | Reference sensor type: `NAC`, `WAC`, `SELENE`, `TC` |
| `--out_dir`, `-o` | `Path` | `projects/run_v2` | Target directory for all registration products and diagnostics |
| `--grid_size` | `int int` | `4 4` | Tiled feature matching grid dimensions (rows, cols) |
| `--method` | `str` | `loftr` | Matching algorithm: `loftr` (deep transformer), `ensemble`, `crater` |
| `--structural_method` | `str` | `phase_congruency` | Modality representation: `phase_congruency` (Log-Gabor) or `gradient` |
| `--coarse_method` | `str` | `auto` | Coarse alignment solver: `auto`, `fft`, `crater` |
| `--poly_degree` | `int` | `2` | Degree of along-track scanline drift polynomial (Layer 2) |
| `--tps_smoothing` | `float` | `0.05` | Thin Plate Spline regularization factor (Layer 3) |
| `--ransac_threshold` | `float` | `1.2` | RANSAC reprojection threshold in pixels (Layer 1 baseline) |
| `--warp_order` | `int` | `3` | Warping spline interpolation order: `3` (Bicubic), `1` (Bilinear) |
| `--wac_band` | `int` | `7` | WAC push-frame spectral band: `7` (689nm Red), `4` (566nm Green), `3` (415nm Blue) |
| `--force` | `flag` | `False` | Force recomputation of intermediate cached products |
| `--skip_verify` | `flag` | `False` | Skip Phase 5 Multi-Pillar verification engine |
| `--export_native` | `flag` | `False` | Export secondary full-size un-decimated native-GSD registered raster |

---

## 📂 Output Artifacts Created by Each Run

When execution finishes, the `--out_dir` contains:
```
<out_dir>/
├── registered_subpixel.tif          # Final sub-pixel registered GeoTIFF product
├── hybrid_transform_model.json      # Mathematical model parameters (Affine + Poly + TPS)
├── tie_points_inliers.csv           # Final verified sub-pixel ground control points
├── subpixel_tie_points.csv          # All ECC-refined sub-pixel tie-points
├── candidate_matches.csv            # Raw candidate matches from LoFTR/SIFT
├── coarse_alignment_result.json     # Initial coarse translation & along-track drift model
├── harmonized/
│   ├── bbox_overlap_source_cammap.tif  # Ingested & scale-harmonized source strip
│   └── bbox_overlap_ref_cropped.tif    # Pixel-aligned reference overlap
└── diagnostics/
    ├── verification_metrics.json       # Machine-readable JSON summary of all quality pillars
    ├── registration_verification.png   # 4-quadrant scientific dashboard with KPI card
    ├── difference_heatmap.png          # Absolute difference residual heatmap
    ├── overview_side_by_side.png       # Side-by-side full-swath comparison
    └── overview_false_color.png        # Dual-band anaglyph overlap composite
```
