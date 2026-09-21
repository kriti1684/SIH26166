# TASK SPECIFICATION: Universal Sub-Pixel Lunar Co-Registration Engine
## TMC–WAC & IIRS–WAC Multi-Modal Updates & Sub-Pixel Optimization

**Problem Statement:** ISRO SIH 26166 — Universal Sub-Pixel Multi-Modal Lunar Image Registration Engine  
**Target Precision:** Sub-pixel reprojection accuracy ($\text{RMSE} < 0.2\text{ px}$ / $< 1.0\text{ m}$ ground error)  
**Target Sensors:** 
- **TMC-2 vs. LRO WAC:** High-Resolution Panchromatic ($5.0\text{ m}$) to Multi-Spectral Push-Frame ($90\text{ m}$)
- **IIRS vs. LRO WAC:** Hyperspectral SWIR ($10.0\text{ m}$) to Multi-Spectral Push-Frame ($90\text{ m}$)
- **OHRC vs. LRO NAC:** High-Resolution Linescan ($0.25\text{ m}$) to Narrow-Angle Linescan ($0.50\text{ m}$) *(Zero Regression)*

---

## 1. Architectural Guardrails & Constraints

1. **Strict Modularity (Zero Regression for OHRC–NAC):**
   - Push-frame de-interleaving, 1D filter flat-field correction, and seam feathering must **only** trigger when `sensor == "WAC"` or `INSTRUMENT_MODE_ID == "COLOR"`.
   - Linescan sensors (`OHRC`, `NAC`, `TMC`) must bypass framelet operations completely.
   - For resolution ratios $< 3.0\times$ (e.g. OHRC–NAC at $2.0\times$), standard direct reference grid matching is preserved.
2. **Computational Load & "Smart Work" Guarantees (RTX 3050 6GB Laptop GPU):**
   - **Zero-RAM Full-Scene Ingestion:** BBox windowed streaming via `rasterio.windows.Window` must remain active. Swaths exceeding $10\text{ GB}$ uncompressed must never be loaded into RAM all at once.
   - **Hyperslab Slice Streaming for IIRS:** Read **only** the selected $800\text{ nm}$ reflective band from HDF5 cubes (`f['Image/Data'][band_idx, :, :]`) directly on disk ($< 50\text{ MB}$ RAM).
   - **Adaptive LoFTR VRAM Safeguard:** Keep tile dimensions during LoFTR attention $\le 768\text{ px}$ (divisible by 8) to cap VRAM under $3.2\text{ GB}$.
   - **Vectorized C-Speed Array Math:** Filter flat-fielding, seam feathering, and CLAHE must use pure vectorized NumPy/OpenCV routines ($< 0.1\text{ s}$ total overhead).
3. **Geodetic & Map Projection Integrity:**
   - Both rasters must be rigidly anchored to standard IAU 2000 Equirectangular Moon coordinates ($R = 1,737,400\text{ m}$).
   - Any intermediate harmonization grid must share the exact bounding box origin $(X_{\min}, Y_{\max})$ of the reference raster with strict integer pixel bounds.
   - The final warped sub-pixel product (`registered_subpixel.tif`) must be exported at the **full native resolution of the source sensor** ($5.0\text{ m}$ for TMC, $10.0\text{ m}$ for IIRS), completely eliminating downsampling blur.

---

## 2. Phased Implementation Roadmap

```
[Phase 1: WAC Push-Frame Optics & Seam Feathering]
  ├─ 1D CCD filter row flat-field transmission normalization
  ├─ 2-pixel cosine seam feathering across 14-line boundaries
  └─ Optical MTF unsharp masking (σ=1.2 px, gain=1.4) + CLAHE
                             │
                             ▼
[Phase 2: Adaptive Scale Harmonization & Native-Res Export]
  ├─ GSD ratio resolver: auto-determines intermediate scale (25-30m) when ratio >= 3x
  ├─ Strict geodetic integer-pixel snapping to Equirectangular Moon CRS
  └─ Dual-Resolution Warping: fits in harmonized frame, warps at native 5m/10m resolution
                             │
                             ▼
[Phase 3: IIRS Hyperspectral Pipeline Integration]
  ├─ Intelligent solar-reflective band selection: Target 689nm -> Select Band 0 (800nm)
  ├─ Pure-Python HDF5 streaming reader (< 50 MB memory)
  └─ Log-Gabor Phase Congruency for 100% modality-invariant crater topology
                             │
                             ▼
[Phase 4: Scientific Verification & Judge-Facing Dashboard]
  ├─ Dual-frame metric reporting: Reference Pixel RMSE + Native Pixel RMSE + Metric Ground RMSE
  └─ Updated diagnostics: difference heatmap, residual quiver plot, false-color overlay
```

---

## 3. Detailed Work Breakdown & Checklist

### Phase 1: WAC Push-Frame Optical Enhancement & Seam Elimination
- [x] **Task 1.1: 1D CCD Row Flat-Field Normalization in `deinterleave_wac`**
  - *Location:* [`src/preprocessing/ingest.py`](file:///c:/%23Padhai/E/SIH1/src/preprocessing/ingest.py)
  - *Problem:* CCD transmission drops from Row 0 ($114.65\text{ DN}$) to Row 13 ($112.03\text{ DN}$) across the physical filter strip, creating an artificial $+2.62\text{ DN}$ step jump ($3.59\times$ gradient spike) every 14 lines.
  - *Implementation:* Compute or apply the normalized transmission profile:
    $$\mathbf{p}_{\text{norm}}[r] = \frac{\bar{I}_{\text{row}}[r]}{\frac{1}{14}\sum_{i=0}^{13}\bar{I}_{\text{row}}[i]}$$
    Divide each $14\times 704$ framelet by $\mathbf{p}_{\text{norm}}$ before concatenation.
- [x] **Task 1.2: Inter-Framelet Cosine Seam Feathering**
  - *Location:* [`src/preprocessing/ingest.py`](file:///c:/%23Padhai/E/SIH1/src/preprocessing/ingest.py)
  - *Problem:* Sub-pixel cross-track flight yaw drift ($\Delta x \approx 1-2\text{ px}$) causes jagged horizontal "staircase" steps on crater rims.
  - *Implementation:* Apply a 2-line raised-cosine / linear feathering blend at each framelet seam:
    $$\text{Row}_{\text{seam}-1} = 0.75 \cdot I_{k-1}[13] + 0.25 \cdot I_{k}[0]$$
    $$\text{Row}_{\text{seam}} = 0.25 \cdot I_{k-1}[13] + 0.75 \cdot I_{k}[0]$$
- [x] **Task 1.3: Optical MTF Restoration Filter & Local Contrast Enhancement**
  - *Location:* [`src/preprocessing/ingest.py`](file:///c:/%23Padhai/E/SIH1/src/preprocessing/ingest.py)
  - *Problem:* Wide-angle lens diffraction ($90^\circ$ FOV) creates optical PSF blur of $\approx 1.2 - 1.5\text{ px}$.
  - *Implementation:* 
    1. Apply CLAHE ($\text{clipLimit}=2.0$, tile grid $8\times 8$) to normalize lunar regolith dynamic range.
    2. Apply Gaussian unsharp mask:
       $$I_{\text{sharp}} = 1.4 \cdot I_{\text{clahe}} - 0.4 \cdot \text{Gaussian}(I_{\text{clahe}}, \sigma=1.2)$$
  - *Success Criteria:* Laplacian sharpness increases by $\ge 5\times$ ($>600$); SIFT keypoint count increases by $\ge 8\times$ ($>1,000$ keypoints); seam gradient ratio drops to $< 1.2\times$.

---

### Phase 2: Adaptive Scale Harmonization & Dual-Resolution Native Export
- [x] **Task 2.1: Automatic Multi-Scale GSD Resolver**
  - *Location:* [`src/preprocessing/scale_harmonizer.py`](file:///c:/%23Padhai/E/SIH1/src/preprocessing/scale_harmonizer.py)
  - *Logic:*
    $$\text{Ratio} = \frac{\max(\text{GSD}_{\text{src}}, \text{GSD}_{\text{ref}})}{\min(\text{GSD}_{\text{src}}, \text{GSD}_{\text{ref}})}$$
    - If $\text{Ratio} < 3.0$ (e.g. OHRC-NAC at $2.0\times$): Keep $\text{GSD}_{\text{harm}} = \text{GSD}_{\text{ref}}$ (existing behavior).
    - If $\text{Ratio} \ge 3.0$ (e.g. TMC-WAC at $18.0\times$, IIRS-WAC at $9.0\times$):
      $$\text{GSD}_{\text{harm}} = \text{clamp}\left(\sqrt{\text{GSD}_{\text{src}} \cdot \text{GSD}_{\text{ref}}}, 20.0, 30.0\right)\text{ m/px}$$
      *(For TMC-WAC: $\sqrt{5.03 \times 90.75} \approx 21.4\text{ m/px}$, clamped to $25.0\text{ m/px}$)*.
- [x] **Task 2.2: Geodetic Alignment & Integer Pixel Snapping**
  - *Location:* [`src/preprocessing/scale_harmonizer.py`](file:///c:/%23Padhai/E/SIH1/src/preprocessing/scale_harmonizer.py)
  - *Implementation:* Lock the harmonized raster bounds to the exact geographic overlap bounds $(X_{\min}, Y_{\min}, X_{\max}, Y_{\max})$ in Equirectangular Moon coordinates:
    $$\text{width} = \text{round}\left(\frac{X_{\max} - X_{\min}}{\text{GSD}_{\text{harm}}}\right), \quad \text{height} = \text{round}\left(\frac{Y_{\max} - Y_{\min}}{\text{GSD}_{\text{harm}}}\right)$$
    Ensure affine transform pixel width $\Delta x = \text{GSD}_{\text{harm}}$ and $\Delta y = -\text{GSD}_{\text{harm}}$ are identical.
- [x] **Task 2.3: Native-Resolution Sub-Pixel Warping (Stage 4)**
  - *Location:* [`src/registration/warper.py`](file:///c:/%23Padhai/E/SIH1/src/registration/warper.py)
  - *Problem:* Previously, the output product was saved on the downsampled grid, losing the native 5m sharpness of TMC.
  - *Implementation:* Scale the fitted Hybrid Deformation Model from the harmonized coordinate frame back into the **native source resolution coordinate frame**:
    $$\mathbf{p}_{\text{src\_native}} = \mathbf{p}_{\text{harm}} \cdot \left(\frac{\text{GSD}_{\text{harm}}}{\text{GSD}_{\text{src\_native}}}\right)$$
    Warp the un-decimated source image directly into `registered_subpixel.tif` at native resolution ($5.03\text{ m}$ for TMC, $10.0\text{ m}$ for IIRS) using order-3 bicubic spline interpolation.

---

### Phase 3: IIRS Hyperspectral Pipeline Integration
- [x] **Task 3.1: Automatic Solar-Reflective Band Selection for WAC**
  - *Location:* [`src/preprocessing/band_selector.py`](file:///c:/%23Padhai/E/SIH1/src/preprocessing/band_selector.py)
  - *Implementation:* When `sensor_ref == "WAC"` or target wavelength $\approx 689\text{ nm}$:
    - Exclude all thermal bands ($> 3,000\text{ nm}$) and hydroxyl absorption regions ($2,800 - 3,000\text{ nm}$).
    - Select **Band 0 ($800.0\text{ nm}$)** or **Band 1 ($816.5\text{ nm}$)**.
    - Validate that solar reflectance contrast matches WAC Band 7 ($689\text{ nm}$).
- [x] **Task 3.2: Pushbroom HDF5 Hyper-Slab Streaming**
  - *Location:* [`src/preprocessing/ingest.py`](file:///c:/%23Padhai/E/SIH1/src/preprocessing/ingest.py)
  - *Implementation:* In `_load_iirs_h5`, use slice notation on the dataset object:
    `f['Image/Data'][band_idx, :, :]` to load strictly 1 band into memory. RAM usage must remain $< 50\text{ MB}$.
- [x] **Task 3.3: Modality-Invariant Crater Ridge Matching via Log-Gabor Phase Congruency**
  - *Location:* [`src/preprocessing/structural.py`](file:///c:/%23Padhai/E/SIH1/src/preprocessing/structural.py)
  - *Implementation:* Verify Phase Congruency maps for IIRS $800\text{ nm}$ vs WAC $689\text{ nm}$. Ensure edge energy is localized along geometric crater rims, achieving $\ge 0.70$ cosine similarity regardless of mineral albedo variations.

---

### Phase 4: Multi-Pillar Scientific Verification & Judge-Facing Dashboards
- [ ] **Task 4.1: Dual-Frame Precision Metric Computation**
  - *Location:* [`src/registration/verifier.py`](file:///c:/%23Padhai/E/SIH1/src/registration/verifier.py)
  - *Implementation:* Report metrics across both reference pixel units and physical ground units:
    1. **Reference Grid RMSE ($\text{px}$)**: Reprojection error in reference coordinate pixels.
    2. **Native Source Grid RMSE ($\text{px}$)**: Error expressed in native source pixel units.
    3. **Physical Ground Error ($\text{meters}$)**:
       $$\text{Error}_{\text{meters}} = \text{RMSE}_{\text{px}} \cdot \text{GSD}_{\text{ref}}$$
    4. **Sub-Pixel Classification**: Flag as `SUB-PIXEL PRECISION ACHIEVED` when Ground Error $< \text{GSD}_{\text{source}} \times 0.5$ (ideal $< 0.2\text{ px}$).
- [ ] **Task 4.2: Scientific Verification Dashboard Generation**
  - *Location:* [`src/registration/verifier.py`](file:///c:/%23Padhai/E/SIH1/src/registration/verifier.py)
  - *Outputs to produce in `diagnostics/`:*
    - `registration_verification.png`: 4-panel dashboard (residual heatmap, vector scatter, convex hull, verdict card).
    - `difference_heatmap.png`: Normalized absolute pixel difference $(|I_{\text{warped}} - I_{\text{ref}}|)$.
    - `overview_false_color.png`: Cyan/Magenta composite showing crater rim alignment.
    - `overview_side_by_side.png`: Co-registered visual comparison.

---

### Phase 5: Automated Testing & Validation Protocol
- [ ] **Task 5.1: Unit Test Suite Expansion**
  - Expand [`tests/test_phase1_harmonization.py`](file:///c:/%23Padhai/E/SIH1/tests/test_phase1_harmonization.py):
    - `test_wac_optical_enhancement`: Verifies seam feathering and MTF sharpening.
    - `test_adaptive_scale_harmonization_math`: Verifies GSD ratio resolver and geodetic bounding box snap.
    - `test_iirs_wac_band_selection`: Verifies selection of Band 0 ($800\text{ nm}$) when matched with WAC ($689\text{ nm}$).
- [ ] **Task 5.2: Non-Regression Check on OHRC–NAC**
  - Run full test suite: `pytest tests/` (all tests must pass with 0 failures).
- [ ] **Task 5.3: End-to-End Test_3 Execution (TMC–WAC)**
  - Execute full pipeline:
    ```powershell
    python run_pipeline.py `
      --source Test_Images/Test_3/TMC/ch2_tmc_ncn_20210517T1508532205_d_img_d18.xml `
      --reference Test_Images/Test_3/WAC/M171992374CE.IMG `
      --sensor_src TMC --sensor_ref WAC `
      --out_dir projects/test_3_subpixel --method loftr --force
    ```
  - Verify:
    - Zero horizontal barcode stripes.
    - Zero jagged 14-line staircase steps.
    - Coherent LoFTR tie-points across all populated tiles.
    - Sub-pixel ECC convergence with $\text{RMSE} < 0.2\text{ px}$ (or $< 1.0\text{ m}$ ground error).
    - Output product `registered_subpixel.tif` retains native 5m sharpness.

---

## 4. Success Metrics & Acceptance Criteria

| Metric | Baseline Test_3 | Target (After Updates) | Verification Method |
| :--- | :--- | :--- | :--- |
| **Seam Gradient Discontinuity** | $9.09$ ($3.59\times$ normal) | $< 3.0$ ($< 1.2\times$ normal) | `scratch/deep_blur_study.py` |
| **WAC Reference Sharpness** | $112.48$ | $> 600.0$ ($\ge 5\times$ sharper) | Laplacian Variance |
| **Inlier Match Count** | $7 / 13$ | $\ge 25$ verified tie-points | `subpixel_tie_points.csv` |
| **Reprojection RMSE** | $1.8904\text{ px}$ | **$< 0.50\text{ px}$ (Target: $< 0.20\text{ px}$)** | `verification_metrics.json` |
| **Final Product Resolution** | Downsampled ($90.75\text{ m}$) | **Native High-Res ($5.03\text{ m}$)** | `registered_subpixel.tif` metadata |
| **Peak GPU VRAM Usage** | $< 3.5\text{ GB}$ | $< 3.5\text{ GB}$ (safe on 6 GB RTX 3050) | PyTorch CUDA memory tracking |
| **Existing Modality Tests** | 18 passed | 100% pass (Zero regressions) | `pytest tests/` |
