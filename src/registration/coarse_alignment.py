"""
src/registration/coarse_alignment.py
========================================
High-Reliability Global Coarse Alignment Engine (Task 2.2).

Provides two independent methods for estimating the global (dx, dy) offset between
the source and reference BBox-harmonized rasters, without requiring ANY feature descriptors.
Both are illumination-invariant.

Method A: Bandpass FFT Phase Correlation
-----------------------------------------
Operates on structural maps (Phase Congruency or gradient), not raw pixels.
Uses Hanning windowing to suppress edge-ringing artifacts.
Accurate to approximately +/- 2 pixels.

Method B: Crater Rim Consensus Voting
---------------------------------------
Physically robust: extracts circular/elliptical crater rims (stable 3D features
whose geometry is sun-angle invariant), then votes for the most consistent
translation vector across all matched crater pairs.
Proven in our research tests to find the correct offset (dx=2, dy=-54) with
8 consensus votes where no other method succeeded.

The module runs BOTH methods and cross-validates the results.
If they agree within a tolerance, the higher-confidence estimate is used.
If they disagree, a fallback hierarchy is invoked.
"""

import json
import math
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import cv2
import numpy as np
import rasterio
from rasterio.windows import Window

try:
    from src.preprocessing.structural import compute_structural_representation
except ImportError:
    from preprocessing.structural import compute_structural_representation


# ─── Constants ───────────────────────────────────────────────────────────────

# Hanning window size for FFT phase correlation (must be power of 2 for speed)
FFT_PATCH_SIZE = 2048

# Minimum Hough circle confidence for rim detection
MIN_HOUGH_ACCUM = 40

# Minimum radius and max radius for crater rim detection (in pixels)
CRATER_MIN_RADIUS_PX = 15
CRATER_MAX_RADIUS_PX = 200

# Histogram bin width (pixels) for crater center voting accumulator
VOTING_BIN_SIZE = 4

# Max plausible offset range to search (+/- pixels, both axes)
MAX_OFFSET_SEARCH_PX = 300

# Agreement threshold: methods agree if their estimates are within this many pixels
METHOD_AGREEMENT_THRESH_PX = 30


# ─── Method A: FFT Phase Correlation on Structural Maps ─────────────────────

def phase_correlation_coarse(
    src_arr: np.ndarray,
    ref_arr: np.ndarray,
    patch_size: int = FFT_PATCH_SIZE
) -> Tuple[float, float, float]:
    """
    Estimate (dx, dy) global offset via Hanning-windowed FFT phase correlation
    on a central patch of the structural maps.

    Steps:
      1. Extract a central patch_size x patch_size crop from both structural maps.
      2. Apply a Hanning window to suppress spectral leakage at boundaries.
      3. Compute normalized cross-power spectrum in frequency domain.
      4. Find the peak of the inverse FFT (peak = translation vector).

    Returns:
        (dx, dy, peak_response): translation in pixels and peak strength (0-1)
    """
    h, w = src_arr.shape

    # Extract central patch
    r0 = max(0, (h - patch_size) // 2)
    c0 = max(0, (w - patch_size) // 2)
    r1 = min(h, r0 + patch_size)
    c1 = min(w, c0 + patch_size)

    ph = r1 - r0
    pw = c1 - c0

    src_patch = src_arr[r0:r1, c0:c1].astype(np.float32)
    ref_patch = ref_arr[r0:r1, c0:c1].astype(np.float32)

    # Hanning window to suppress edge discontinuities
    hann = cv2.createHanningWindow((pw, ph), cv2.CV_32F)
    src_w = src_patch * hann
    ref_w = ref_patch * hann

    # Phase correlation
    shift, response = cv2.phaseCorrelate(src_w, ref_w)
    dx, dy = float(shift[0]), float(shift[1])

    return dx, dy, float(response)


# ─── Method B: Crater Rim Consensus Voting ───────────────────────────────────

def crater_rim_consensus_voting(
    src_clahe: np.ndarray,
    ref_clahe: np.ndarray,
    min_radius: int = CRATER_MIN_RADIUS_PX,
    max_radius: int = CRATER_MAX_RADIUS_PX,
    bin_size: int = VOTING_BIN_SIZE,
    max_offset: int = MAX_OFFSET_SEARCH_PX
) -> Tuple[Optional[float], Optional[float], int, np.ndarray]:
    """
    Estimate (dx, dy) by:
      1. Detecting crater circular rims in both images via Hough circle transform.
      2. For each pair (src_crater_i, ref_crater_j) with similar radius,
         compute the candidate translation: (ref_j.cx - src_i.cx, ref_j.cy - src_i.cy).
      3. Build a 2D vote histogram over the translation space.
      4. The peak bin is the consensus offset.

    Returns:
        (dx, dy, peak_votes, vote_histogram)
        If no consensus found, returns (None, None, 0, histogram)
    """
    # Detect craters via Hough Circles on CLAHE-enhanced images
    def detect_craters(img: np.ndarray) -> Optional[np.ndarray]:
        circles = cv2.HoughCircles(
            img,
            cv2.HOUGH_GRADIENT,
            dp=1.5,
            minDist=min_radius * 2,
            param1=100,
            param2=MIN_HOUGH_ACCUM,
            minRadius=min_radius,
            maxRadius=max_radius
        )
        return circles[0] if circles is not None else None

    src_circles = detect_craters(src_clahe)
    ref_circles = detect_craters(ref_clahe)

    n_src = len(src_circles) if src_circles is not None else 0
    n_ref = len(ref_circles) if ref_circles is not None else 0

    print(f"  [CRATER-VOTE] Detected {n_src} source rims, {n_ref} reference rims.")

    if src_circles is None or ref_circles is None or n_src == 0 or n_ref == 0:
        return None, None, 0, np.array([])

    # Build translation vote accumulator
    bins = np.arange(-max_offset, max_offset + bin_size, bin_size)
    n_bins = len(bins) - 1
    H = np.zeros((n_bins, n_bins), dtype=np.int32)

    vote_count = 0
    for sc in src_circles:
        sx, sy, sr = sc[0], sc[1], sc[2]
        for rc in ref_circles:
            rx, ry, rr = rc[0], rc[1], rc[2]
            # Only match craters with similar radii (within 25%)
            if abs(sr - rr) / max(sr, rr, 1) > 0.25:
                continue
            tdx = rx - sx
            tdy = ry - sy
            # Only consider offsets within our search range
            if abs(tdx) >= max_offset or abs(tdy) >= max_offset:
                continue

            bx = int((tdx + max_offset) / bin_size)
            by = int((tdy + max_offset) / bin_size)
            bx = min(bx, n_bins - 1)
            by = min(by, n_bins - 1)
            H[by, bx] += 1
            vote_count += 1

    if vote_count == 0:
        print("  [CRATER-VOTE] No radius-compatible pairs found in offset range.")
        return None, None, 0, H

    # Find peak
    peak_idx = np.unravel_index(np.argmax(H), H.shape)
    peak_votes = int(H[peak_idx])
    peak_dx = float(bins[peak_idx[1]] + bin_size / 2.0)
    peak_dy = float(bins[peak_idx[0]] + bin_size / 2.0)

    print(f"  [CRATER-VOTE] Consensus peak: dx={peak_dx:.1f}, dy={peak_dy:.1f} with {peak_votes} votes.")
    return peak_dx, peak_dy, peak_votes, H


# ─── Master Coarse Alignment ─────────────────────────────────────────────────

def run_coarse_alignment(
    source_harmonized_path: Path,
    ref_cropped_path: Path,
    output_dir: Path,
    patch_size: int = FFT_PATCH_SIZE,
    num_strips: int = 10,
    structural_method: str = "phase_congruency",
    min_crater_votes: int = 4,
    agreement_thresh: float = float(METHOD_AGREEMENT_THRESH_PX)
) -> Dict[str, Any]:
    """
    Estimate the global (dx, dy) offset and along-track drift between the BBox-harmonized 
    source and reference rasters using multi-strip vertical sampling.

    Workflow:
        1. Sample N=8-10 horizontal strips along the full image height.
        2. For each strip, compute Phase Correlation on structural representations
           and Crater Rim Consensus Voting (illumination-invariant, zero classical descriptors).
        3. Fit a 1D linear drift model: dy(row) = a*row + b and dx(row) = c*row + d.
        4. Write the strip profile and drift model to coarse_alignment_result.json.
    """
    source_harmonized_path = Path(source_harmonized_path)
    ref_cropped_path = Path(ref_cropped_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "coarse_alignment_result.json"

    print("[COARSE-ALIGN] Running multi-strip Decimated Coarse Alignment (Structural FFT + Crater Voting)...")
    print(f"  Source: {source_harmonized_path.name}")
    print(f"  Ref:    {ref_cropped_path.name}")

    with rasterio.open(source_harmonized_path) as src, rasterio.open(ref_cropped_path) as ref:
        strip_profiles = []
        h = min(src.height, ref.height)
        w = min(src.width, ref.width)

        strip_step = h / num_strips
        
        for i in range(num_strips):
            r_center = int((i + 0.5) * strip_step)
            c_center = w // 2

            r0 = max(0, r_center - patch_size // 2)
            c0 = max(0, c_center - patch_size // 2)
            
            # Read window
            win = Window(col_off=c0, row_off=r0, width=min(patch_size, w - c0), height=min(patch_size, h - r0))
            src_patch = src.read(1, window=win).astype(np.float32)
            ref_patch = ref.read(1, window=win).astype(np.float32)
            
            # Check for sufficient valid data
            if (src_patch > 0).mean() < 0.2 or (ref_patch > 0).mean() < 0.2:
                print(f"  Strip {i+1}/{num_strips} at row {r_center}: Too much nodata, skipping.")
                continue

            # Compute structural maps
            src_struct = compute_structural_representation(src_patch, method=structural_method)
            ref_struct = compute_structural_representation(ref_patch, method=structural_method)
            
            src_structural = src_struct["structural"]
            ref_structural = ref_struct["structural"]
            src_clahe = src_struct["clahe"]
            ref_clahe = ref_struct["clahe"]
            
            # Method A: FFT Phase Correlation
            dx_fft, dy_fft, fft_resp = phase_correlation_coarse(src_structural, ref_structural, patch_size)
            
            # Method B: Crater Rim Consensus
            dx_c, dy_c, c_votes, _ = crater_rim_consensus_voting(src_clahe, ref_clahe)
            
            # Selection
            final_dx, final_dy = dx_fft, dy_fft
            method_used = "fft"
            if dx_c is not None and c_votes >= min_crater_votes:
                dist = math.hypot(dx_fft - dx_c, dy_fft - dy_c)
                if dist <= agreement_thresh:
                    final_dx, final_dy = dx_c, dy_c
                    method_used = "crater_voting (agreed)"
                elif c_votes >= min_crater_votes * 2:
                    final_dx, final_dy = dx_c, dy_c
                    method_used = "crater_voting (dominant)"

            print(f"  Strip {i+1}/{num_strips} at row {r_center}: dx={final_dx:.1f}, dy={final_dy:.1f} (fft_resp={fft_resp:.3f}, c_votes={c_votes}, used={method_used})")
            
            strip_profiles.append({
                "strip_idx": i,
                "row": r_center,
                "dx": final_dx,
                "dy": final_dy,
                "fft_response": fft_resp,
                "crater_votes": c_votes,
                "method_used": method_used
            })
            
    if not strip_profiles:
        raise ValueError("Failed to compute coarse alignment for any strip.")

    # Fit 1D drift model with robust MAD outlier filtering
    rows = np.array([sp["row"] for sp in strip_profiles], dtype=np.float64)
    dxs = np.array([sp["dx"] for sp in strip_profiles], dtype=np.float64)
    dys = np.array([sp["dy"] for sp in strip_profiles], dtype=np.float64)

    med_dx = float(np.median(dxs))
    med_dy = float(np.median(dys))
    
    mad_dx = float(np.median(np.abs(dxs - med_dx)))
    mad_dy = float(np.median(np.abs(dys - med_dy)))
    
    # Tolerant inlier mask: points within 3 * max(MAD, 15px) of the median
    inliers = (np.abs(dxs - med_dx) <= max(3.0 * mad_dx, 45.0)) & \
              (np.abs(dys - med_dy) <= max(3.0 * mad_dy, 45.0))
              
    from scipy.stats import linregress
    
    if np.sum(inliers) >= 3:
        rows_inl = rows[inliers]
        dxs_inl = dxs[inliers]
        dys_inl = dys[inliers]
        dy_slope, dy_intercept, _, _, _ = linregress(rows_inl, dys_inl)
        dx_slope, dx_intercept, _, _, _ = linregress(rows_inl, dxs_inl)
        # Sanity check: slope should not be physically absurd for along-track satellite motion (|slope| < 0.2)
        if abs(dy_slope) > 0.15:
            dy_slope, dy_intercept = 0.0, med_dy
        if abs(dx_slope) > 0.15:
            dx_slope, dx_intercept = 0.0, med_dx
    else:
        dy_slope, dx_slope = 0.0, 0.0
        dy_intercept, dx_intercept = med_dy, med_dx

    print("  [COARSE-ALIGN] Drift Model fitted:")
    print(f"    dy(row) = {dy_slope:.6f} * row + {dy_intercept:.2f}")
    print(f"    dx(row) = {dx_slope:.6f} * row + {dx_intercept:.2f}")
    print(f"    Median dx = {med_dx:.2f}, Median dy = {med_dy:.2f}")

    result = {
        "dx": med_dx,  # Backwards compatibility
        "dy": med_dy,  # Backwards compatibility
        "confidence": float(np.mean([sp["fft_response"] for sp in strip_profiles])),
        "drift_model": {
            "dy_slope": float(dy_slope),
            "dy_intercept": float(dy_intercept),
            "dx_slope": float(dx_slope),
            "dx_intercept": float(dx_intercept)
        },
        "strip_profiles": strip_profiles,
        "source": str(source_harmonized_path),
        "reference": str(ref_cropped_path)
    }

    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    return result
