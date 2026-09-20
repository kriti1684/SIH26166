# Adversarial Red Team Benchmark Report

### Evaluation of Lunar Registration Robustness against Deceptive Traps

| Challenge ID | Attack Vector | Raw Matcher Verdict | Multi-Pillar Verdict | Defense Mechanism |
| :--- | :--- | :--- | :--- | :--- |
| **RED-01**: Deceptive Twin Crater Trap | Visually identical impact craters at disparate lunar locatio... | `ACCEPTED (24 circular rim feat...` | **REJECTED** | Standard visual descriptors match symmetrical rim texture, but geometr... |
| **RED-02**: Opposite-Sun Shadow Inversion Illusion | 180° Solar Azimuth shift inverts lunar crater shadows, rever... | `ACCEPTED / AMBIGUOUS (Spurious...` | **REJECTED** | Solar ephemeris verification detects that the light source has flipped... |
| **RED-03**: Degenerate Clustered Boulder Trap | 25 tightly packed keypoints on a single 15x15 px boulder pat... | `ACCEPTED (25/25 inliers, 100% ...` | **REJECTED** | Visual matchers report 100% inlier ratio on a single cluster. Spatial ... |
| **RED-04**: Featureless Mare Noise Correlation | Smooth basaltic mare regions with low contrast and sensor sh... | `UNCERTAIN (Weak feature respon...` | **REJECTED** | The pipeline rejects low-contrast spurious noise matches before any tr... |

---

## RED-01: Deceptive Twin Crater Trap

- **Attack Vector**: Visually identical impact craters at disparate lunar locations (-65°S vs -78°S) with matching rim textures.
- **Raw Visual Matcher**: ACCEPTED (24 circular rim feature matches, raw confidence: 0.84)
- **Physics Verification Verdict**: **REJECTED** (CORRECTLY_REJECTED)
- **Scientific Metrics**: Spatial Entropy $H(S)=0.497$, Convex Hull Coverage = 2.73%, Illumination Score = 0.85
- **Triggered Rejection Guards**:
  - `Spatial Entropy Warning: Matches concentrated on single circular rim (H=0.50)`
  - `Geometric Reprojection Residual: Exceeded strict planar tolerance (residual > 6.2 px)`
  - `Non-Unique Projective Mapping: Circular crater symmetry creates 360° rotational ambiguity`
- **Scientific Justification**: Standard visual descriptors match symmetrical rim texture, but geometric residual gating and spatial distribution analysis successfully detect the false correlation.

## RED-02: Opposite-Sun Shadow Inversion Illusion

- **Attack Vector**: 180° Solar Azimuth shift inverts lunar crater shadows, reversing bright and dark facets.
- **Raw Visual Matcher**: ACCEPTED / AMBIGUOUS (Spurious edge gradient alignment on opposing rim contours)
- **Physics Verification Verdict**: **REJECTED** (CORRECTLY_REJECTED)
- **Scientific Metrics**: Spatial Entropy $H(S)=0.620$, Convex Hull Coverage = 38.5%, Illumination Score = 0.20
- **Triggered Rejection Guards**:
  - `Solar Azimuth Conflict: Delta Azimuth dAz=180.0 deg indicates direct shadow polarity reversal`
  - `Illumination Divergence: Sunlight vector opposition causes illuminated crater rims to match shadowed inner walls`
- **Scientific Justification**: Solar ephemeris verification detects that the light source has flipped 180°, vetoing false edge alignments before warping.

## RED-03: Degenerate Clustered Boulder Trap

- **Attack Vector**: 25 tightly packed keypoints on a single 15x15 px boulder patch with zero frame-wide distribution.
- **Raw Visual Matcher**: ACCEPTED (25/25 inliers, 100% inlier ratio, raw confidence: 0.96)
- **Physics Verification Verdict**: **REJECTED** (CORRECTLY_REJECTED)
- **Scientific Metrics**: Spatial Entropy $H(S)=-0.000$, Convex Hull Coverage = 0.006%, Illumination Score = 0.88
- **Triggered Rejection Guards**:
  - `Spatial Entropy Collapse: All inliers concentrated in a single grid cell (H=-0.000)`
  - `Convex Hull Area Anomaly: Match points cover only 0.006% of frame (< 4.0% minimum threshold)`
  - `Degenerate Support: Insufficient baseline for full-image projective transformation`
- **Scientific Justification**: Visual matchers report 100% inlier ratio on a single cluster. Spatial distribution entropy gating successfully flags and rejects the degenerate cluster.

## RED-04: Featureless Mare Noise Correlation

- **Attack Vector**: Smooth basaltic mare regions with low contrast and sensor shot noise producing spurious candidate matches.
- **Raw Visual Matcher**: UNCERTAIN (Weak feature response, low candidate match density)
- **Physics Verification Verdict**: **REJECTED** (CORRECTLY_REJECTED)
- **Scientific Metrics**: Spatial Entropy $H(S)=0.396$, Convex Hull Coverage = 0.88%, Illumination Score = 0.55
- **Triggered Rejection Guards**:
  - `Insufficient Inliers: Less than 4 stable correspondences in low-contrast regolith (N=3)`
  - `Matrix Condition Number Instability: Transformation solution degenerate`
- **Scientific Justification**: The pipeline rejects low-contrast spurious noise matches before any transformation can corrupt the registered product.

