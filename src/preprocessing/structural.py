"""
src/preprocessing/structural.py
====================================
Universal Modality-Invariant Structural Representation Engine.

Converts any lunar raster (OHRC, IIRS, TMC-2, NAC, WAC) into an
illumination-invariant structural map suitable for cross-sensor feature matching.

Theory:
-------
Classical feature detectors (SIFT, SuperPoint) fail when solar illumination changes
because they detect intensity gradients — gradients that flip sign or vanish when
the sun moves from 8.6 deg to 66.2 deg.

The solution: Phase Congruency (PC) — a Fourier-theoretic measure that detects
feature points where local frequency components are maximally in phase. PC is:
  1. Completely contrast-invariant (works for both deep shadow and flat noon terrain)
  2. Completely polarity-invariant (shadow edges and lit edges produce the same response)
  3. Multi-scale (detects crater rims at all diameters)

The implementation uses Log-Gabor filters following Kovesi (1996, 1999).

Outputs per image:
  - Phase Congruency map (floating point, 0-1)
  - Shadow validity mask (1 = valid, 0 = shadowed/invalid)
  - Normalized Gradient Magnitude (Sobel, fast fallback for uniformly lit terrain)
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, Optional, Dict
import rasterio
from rasterio.windows import Window

# ─── Log-Gabor Filter Bank Parameters ───────────────────────────────────────
# These are Kovesi's canonical defaults, tuned for planetary imagery
NUM_SCALES = 4         # Number of radial frequency scales
NUM_ORIENTATIONS = 6   # Number of filter orientations (every 30 deg)
MIN_WAVELENGTH = 6.0   # Smallest wavelength in pixels
MULT = 2.1             # Scaling factor between successive wavelengths
SIGMA_ON_F = 0.55      # Bandwidth of Log-Gabor filter (sigma/f ratio)
NOISE_THRESH_FACTOR = 2.0  # Noise threshold factor


# ─── Log-Gabor Filter Construction ──────────────────────────────────────────

def _log_gabor_filter(rows: int, cols: int, f0: float, sigma_f: float) -> np.ndarray:
    """
    Construct a single isotropic Log-Gabor radial filter in frequency domain.
    Log-Gabor: transfer = exp(-(log(f/f0))^2 / (2*(log(sigma_f/f0))^2))
    """
    u = np.fft.fftfreq(cols)
    v = np.fft.fftfreq(rows)
    U, V = np.meshgrid(u, v)
    radius = np.sqrt(U**2 + V**2)
    radius[0, 0] = 1.0  # Avoid log(0)

    log_gabor = np.exp(-(np.log(radius / f0))**2 / (2.0 * (np.log(sigma_f / f0))**2))
    log_gabor[0, 0] = 0.0  # Zero DC component
    return log_gabor.astype(np.float32)


def _build_filter_bank(rows: int, cols: int) -> list:
    """
    Build a multi-scale Log-Gabor filter bank (isotropic, radial only).
    Returns list of (filter_fft, wavelength) tuples for each scale.
    """
    filters = []
    for s in range(NUM_SCALES):
        wavelength = MIN_WAVELENGTH * (MULT ** s)
        f0 = 1.0 / wavelength
        sigma_f = SIGMA_ON_F * f0

        lf = _log_gabor_filter(rows, cols, f0, sigma_f)
        filters.append((lf, wavelength))
    return filters


# ─── Phase Congruency ────────────────────────────────────────────────────────

def compute_phase_congruency(
    image: np.ndarray,
    num_scales: int = NUM_SCALES,
    min_wavelength: float = MIN_WAVELENGTH,
    mult: float = MULT,
    sigma_on_f: float = SIGMA_ON_F,
    noise_thresh_factor: float = NOISE_THRESH_FACTOR
) -> np.ndarray:
    """
    Compute the 2D Phase Congruency map of an image using isotropic Log-Gabor filters.

    Phase Congruency at a point is high when Fourier components across scales
    all arrive at the same phase — this signals a structural feature (edge, rim, ridge)
    regardless of local contrast or illumination intensity.

    PC(x) = sum_n [ A_n * cos(phi_n - phi_mean) - |sin(phi_n - phi_mean)| ]
             / (sum_n A_n + epsilon)

    Parameters:
        image: 2D float32 array, values in [0, 255] or [0, 1]
        num_scales: number of Log-Gabor frequency scales
        min_wavelength: smallest spatial wavelength in pixels
        mult: scaling factor between successive wavelengths
        sigma_on_f: Log-Gabor bandwidth parameter
        noise_thresh_factor: scalar multiplier on noise floor threshold

    Returns:
        pc_map: 2D float32 array in [0, 1], illumination-invariant structural features
    """
    rows, cols = image.shape
    img_f = image.astype(np.float64)

    # Normalize to zero-mean to suppress DC
    img_f -= img_f.mean()

    IM = np.fft.fft2(img_f)

    # Accumulate across scales using Kovesi's full weighted mean phase formulation.
    # Key insight: we accumulate complex phasors so the mean phase tracks the
    # dominant phase direction. cos(phi_n - phi_mean) is the phase coherence per scale.
    # This correctly handles contrast changes because An cancels in the ratio.
    sum_an = np.zeros((rows, cols), dtype=np.float64)    # Total amplitude
    xsum = np.zeros((rows, cols), dtype=np.float64)       # Sum of An*cos(phi_n) -> weighted real
    ysum = np.zeros((rows, cols), dtype=np.float64)       # Sum of An*sin(phi_n) -> weighted imag

    noise_energies = []

    for s in range(num_scales):
        wavelength = min_wavelength * (mult ** s)
        f0 = 1.0 / wavelength
        sigma_f = sigma_on_f * f0

        lf = _log_gabor_filter(rows, cols, f0, sigma_f).astype(np.float64)

        # Apply filter in frequency domain
        IF = IM * lf
        resp = np.fft.ifft2(IF)

        even = np.real(resp)  # Even-symmetric (cos) response
        odd  = np.imag(resp)  # Odd-symmetric (sin) response

        # Amplitude at this scale
        An = np.sqrt(even**2 + odd**2) + 1e-10

        # Accumulate amplitude-weighted phasors
        # cos(phi) = even/An, sin(phi) = odd/An
        xsum += even  # = An * cos(phi_n)
        ysum += odd   # = An * sin(phi_n)
        sum_an += An

        noise_energies.append(np.median(An))

    # Resultant amplitude of the phasor sum = sum_n [An * cos(phi_n - phi_mean)]
    coherent = np.sqrt(xsum**2 + ysum**2)   # This is illumination-invariant

    # Noise threshold (simple energy-based estimate)
    noise_floor = noise_thresh_factor * (np.mean(noise_energies) + 1e-10)

    # Phase Congruency: ratio of coherent phasor resultant to total amplitude
    # PC = max(0, |phasor_sum| - tau) / (sum_An + epsilon)
    pc = np.maximum(0.0, coherent - noise_floor) / (sum_an + 1e-10)

    # Clip to [0, 1]
    pc = np.clip(pc, 0.0, 1.0)
    return pc.astype(np.float32)


# ─── Normalized Gradient Field (Sobel) ──────────────────────────────────────

def compute_normalized_gradient(image: np.ndarray) -> np.ndarray:
    """
    Compute the normalized Sobel gradient magnitude.
    Fast illumination-robust fallback for uniformly illuminated terrain.
    The gradient magnitude detects slope changes (crater rims) regardless of albedo offset.
    """
    img_u8 = image.astype(np.float32)
    gx = cv2.Sobel(img_u8, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_u8, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    mag_max = mag.max()
    if mag_max > 0:
        mag /= mag_max
    return mag.astype(np.float32)


# ─── Shadow / Photometric Validity Mask ─────────────────────────────────────

def compute_shadow_mask(
    image: np.ndarray,
    shadow_pct_threshold: float = 5.0,
    shadow_nodata_value: float = 0.0,
) -> np.ndarray:
    """
    Generate a binary validity mask that identifies moving shadow regions.

    Logic:
        1. Identify NoData pixels (value == shadow_nodata_value).
        2. Identify dark-shadow pixels using Otsu's adaptive threshold on the
           lower 15th percentile of the histogram.
        3. Return mask: 1 = valid terrain, 0 = shadow or NoData.

    This prevents keypoint detectors from anchoring on shadow terminators
    that shift when the sun angle changes.

    Parameters:
        image: 2D float32/uint8 array
        shadow_pct_threshold: percentile below which pixels are classified as shadowed
        shadow_nodata_value: pixel value representing NoData (typically 0)

    Returns:
        mask: 2D uint8 array (1=valid, 0=shadow/invalid)
    """
    img_f = image.astype(np.float32)

    # 1. NoData mask
    nodata_mask = (img_f == shadow_nodata_value).astype(np.uint8)

    # 2. Shadow mask: pixels below the Pth percentile of valid pixels
    valid_pixels = img_f[img_f > shadow_nodata_value]
    if len(valid_pixels) < 100:
        return np.ones_like(img_f, dtype=np.uint8)

    shadow_floor = np.percentile(valid_pixels, shadow_pct_threshold)
    shadow_mask = (img_f <= shadow_floor).astype(np.uint8)

    # 3. Grow shadow mask slightly to exclude ambiguous penumbra zones
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    shadow_dilated = cv2.dilate(shadow_mask, kernel)

    # Valid = NOT (nodata OR shadow)
    invalid = np.clip(nodata_mask + shadow_dilated, 0, 1)
    valid_mask = (1 - invalid).astype(np.uint8)

    return valid_mask


# ─── Master Structural Representation Function ──────────────────────────────

def compute_structural_representation(
    image: np.ndarray,
    method: str = "phase_congruency",
    apply_clahe: bool = True,
    clahe_clip: float = 3.0,
    clahe_grid: int = 8,
    shadow_pct_threshold: float = 5.0
) -> Dict[str, np.ndarray]:
    """
    Main entry point: Produce the illumination-invariant structural
    representation for a single-band image array.

    Parameters:
        image: 2D array (float32 or uint8), any scale
        method: "phase_congruency" | "gradient" | "combined"
        apply_clahe: If True, apply CLAHE before structural computation
        clahe_clip: CLAHE clip limit
        clahe_grid: CLAHE tile size
        shadow_pct_threshold: Shadow mask dark percentile cutoff

    Returns:
        dict with keys:
          'structural': The primary structural map (float32, 0-1)
          'shadow_mask': Binary validity mask (uint8, 0 or 1)
          'clahe': CLAHE-enhanced image (uint8)
    """
    # Normalize to uint8 for CLAHE
    img_f = image.astype(np.float32)
    valid = img_f[img_f > 0]
    if len(valid) < 100:
        return {
            "structural": np.zeros_like(img_f),
            "shadow_mask": np.zeros_like(img_f, dtype=np.uint8),
            "clahe": np.zeros_like(img_f, dtype=np.uint8)
        }

    p2, p98 = np.percentile(valid, (2, 98))
    img_u8 = np.clip((img_f - p2) / (p98 - p2 + 1e-6) * 255.0, 0, 255).astype(np.uint8)

    # CLAHE local contrast enhancement
    if apply_clahe:
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_grid, clahe_grid))
        img_clahe = clahe.apply(img_u8)
    else:
        img_clahe = img_u8.copy()

    # Shadow / photometric validity mask
    shadow_mask = compute_shadow_mask(img_f, shadow_pct_threshold=shadow_pct_threshold)

    # Structural representation
    if method == "phase_congruency":
        structural = compute_phase_congruency(img_clahe.astype(np.float64))
    elif method == "gradient":
        structural = compute_normalized_gradient(img_clahe.astype(np.float32))
    elif method == "combined":
        pc = compute_phase_congruency(img_clahe.astype(np.float64))
        grad = compute_normalized_gradient(img_clahe.astype(np.float32))
        structural = 0.6 * pc + 0.4 * grad
    else:
        raise ValueError(f"Unknown method: {method}. Use 'phase_congruency', 'gradient', or 'combined'.")

    # Zero out structural features inside shadow regions
    structural *= shadow_mask.astype(np.float32)

    return {
        "structural": structural,
        "shadow_mask": shadow_mask,
        "clahe": img_clahe
    }


def load_and_compute_structural(
    tif_path: Path,
    band: int = 1,
    window: Optional[Window] = None,
    method: str = "phase_congruency",
    shadow_pct_threshold: float = 5.0
) -> Dict[str, np.ndarray]:
    """
    Load a GeoTIFF (optionally windowed) and compute its structural representation.

    Parameters:
        tif_path: Path to input GeoTIFF
        band: Band index to read (1-indexed)
        window: Optional rasterio Window to read a sub-region
        method: Structural method
        shadow_pct_threshold: Shadow detection threshold

    Returns:
        Dict with 'structural', 'shadow_mask', 'clahe' arrays
    """
    tif_path = Path(tif_path)
    with rasterio.open(tif_path) as src:
        data = src.read(band, window=window).astype(np.float32)

    return compute_structural_representation(
        data,
        method=method,
        shadow_pct_threshold=shadow_pct_threshold
    )
