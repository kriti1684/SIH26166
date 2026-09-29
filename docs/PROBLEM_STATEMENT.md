# ISRO Smart India Hackathon (SIH 2024) — Problem Statement 26166

## Multi-modal, Sun Angle and Scale Invariant Image Correspondence using Chandrayaan-2 Optical Images (OHRC, TMC-2, and IIRS)

| Field | Details |
| :--- | :--- |
| **Organization** | Indian Space Research Organisation (ISRO) |
| **Department** | Department of Space / ISRO |
| **Category** | Software |
| **Theme** | Space Technology |
| **Problem ID** | 26166 |

---

## 1. Background

**Image Registration** is the foundational photogrammetric process of geometrically transforming two or more images of the same planetary scene taken:
- At different orbital epochs / dates,
- Under varying solar illumination angles,
- From divergent viewing geometries and altitudes, or
- Across different optical sensor modalities,

into a single, highly accurate, unified pixel-aligned coordinate system.

### Key Components
- **Source Image (Moving / Warped):** The primary observation image that is geometrically deformed to match the target reference.
- **Reference Image (Fixed / Base):** The ground-truth or base reference image onto which the source image is registered.

---

## 2. Core Problem & Physical Challenges

Planetary image correspondence on the Moon presents extreme physical hurdles not encountered in terrestrial satellite registration:

### 1. Illumination Disparity & Shadow Reversals
Variations in Sun azimuth and elevation drastically alter the photometric response of lunar craters, boulders, and ridges. Handcrafted feature detectors (SIFT, SURF, ORB, AKAZE) fail completely when shadow directions are inverted or perpendicular.

### 2. Viewpoint & Along-Track Pushbroom Geometry
Line-scan pushbroom cameras (OHRC, TMC-2) accumulate unmodeled along-track jitter, roll/pitch attitude drift, and sensor thermal flexure over orbits exceeding 20,000 pixels. A simple global affine or homography transformation is mathematically incapable of modeling these deformations.

### 3. Extreme Scale & Ground Sampling Distance (GSD) Disparities ($> 18\times$)
Matching high-resolution payloads (e.g. OHRC at $0.25\text{ m/px}$ or TMC-2 at $5\text{ m/px}$) against regional reference products (e.g. LRO WAC at $90.75\text{ m/px}$) requires scale harmonization without introducing aliasing artifacts or detail loss.

---

## 3. Expected Deliverables & Evaluation Criteria

The competition mandates a generic, production-ready software engine delivering:

1. **Sub-Pixel Registration Accuracy**:
   - Target: $\le 0.50\text{ px}$ reprojection RMSE across independent holdout cross-validation.
   - High-precision tier: $< 0.20 - 0.25\text{ px}$ RMSE.
2. **Uniform Spatial Distribution of Inliers**:
   - Tie-points must span the entire image swath ($> 50\% - 70\%$ of along-track height), avoiding single-crater cluster collapse.
   - High spatial entropy ($H(S) > 2.0$) and convex hull area ($> 15\% - 20\%$).
3. **Multi-Pillar Verification & Provenance**:
   - Transparent evaluation metrics (RMSE, MAD error, inlier count, inlier ratio, composite scientific confidence).
   - Automated diagnostic visual dashboards, difference heatmaps, and side-by-side verification previews.
4. **100% ISIS-Free Cross-Platform Deployment**:
   - Operates in pure Python/C++ without requiring USGS ISIS3 binary installations, WSL containers, or Linux virtualization.

---

## 4. Optical Payloads & Planetary Datasets

### Chandrayaan-2 Optical Payloads
- **OHRC (Orbiter High Resolution Camera)**: $0.25\text{ m/px}$ high-resolution panchromatic imaging.
- **TMC-2 (Terrain Mapping Camera-2)**: $5.0\text{ m/px}$ stereo triplets (Fore, Nadir, Aft).
- **IIRS (Imaging Infrared Spectrometer)**: $250$-band hyperspectral cube ($0.8 - 5.0\ \mu\text{m}$).
- *Data Portal*: [ISSDC ChMapBrowse](https://chmapbrowse.issdc.gov.in/)

### Reference Datasets
- **LRO NAC (Narrow Angle Camera)**: $0.5 - 2.5\text{ m/px}$ lunar reconnaissance imagery.
- **LRO WAC (Wide Angle Camera)**: $90.75\text{ m/px}$ global regional mosaic.
- **SELENE (Kaguya) Terrain Camera (TC)**: $10\text{ m/px}$ ortho-mosaic coverage.
