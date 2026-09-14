"""
src/preprocessing/scale_harmonizer.py
======================================
Scale-Space Harmonizer — fulfills SIH26166 Scale Invariance requirement.

Problem
-------
Different lunar sensors have wildly different Ground Sampling Distances (GSD):
  - CH-2 OHRC  :   0.25 m/px  (very high resolution)
  - LRO NAC    :   0.5–1.2 m/px
  - CH-2 TMC-2 :   5.0 m/px
  - LRO WAC    : 100.0 m/px

Matching OHRC (0.25 m) directly against WAC (100 m) means a single WAC pixel
covers a 400×400 OHRC patch — SuperPoint & SuperGlue will fail entirely.

Solution
--------
1. Compute GSD ratio R = GSD_ref / GSD_source.
2. If R > THRESHOLD (default 1.5):
     - Apply Gaussian blur (sigma ~ R/2) to the moving image to remove aliasing.
     - Downsample moving image to approximately match reference GSD.
     - Keep the full-resolution image saved separately for final fine warping.
3. Return both the coarse-scale and the full-resolution arrays.

Usage
-----
    from src.preprocessing.scale_harmonizer import harmonize_scale, get_gsd

    coarse_arr, full_arr, scale_factor = harmonize_scale(
        source_path="./normalized/OHRC_normalized.tif",
        ref_path="./normalized/NAC_normalized.tif",
    )
"""

import math
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine
import scipy.ndimage as ndimage

# ── Tuneable constants ────────────────────────────────────────────────────────
GSD_RATIO_THRESHOLD   = 1.5   # If ref_GSD / source_GSD > this, harmonize
MAX_DOWNSCALE_FACTOR  = 32.0  # Never downsample beyond 1:32 in one pass
GAUSSIAN_SIGMA_FACTOR = 0.5   # sigma = factor × downsample_ratio (anti-alias)


# ── Public API ────────────────────────────────────────────────────────────────

def get_gsd(file_path: Path) -> Optional[float]:
    """
    Read the Ground Sampling Distance (metres/pixel) from a GeoTIFF.

    Returns None if the file has no spatial reference or a non-metric CRS.
    """
    file_path = Path(file_path)
    try:
        with rasterio.open(file_path) as ds:
            t = ds.transform
            # pixel width in map units
            gsd = abs(t.a)
            # If CRS is geographic (degrees), convert roughly to metres
            if ds.crs and ds.crs.is_geographic:
                gsd = gsd * 111_320.0  # 1 degree ≈ 111.32 km
            return gsd
    except Exception as e:
        print(f"[SCALE] Could not read GSD from {file_path}: {e}")
        return None


def harmonize_scale(
    source_path: Path,
    ref_path: Path,
    out_coarse_path: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Harmonize the resolution of `source_path` to approximately match `ref_path`.

    If no harmonization is needed (GSD ratio ≤ threshold), returns the
    original source array as both the coarse and full-resolution outputs.

    Parameters
    ----------
    source_path    : Path to the moving (source) normalized GeoTIFF.
    ref_path       : Path to the reference normalized GeoTIFF.
    out_coarse_path: Optional path to write the coarse-scale TIF to disk.

    Returns
    -------
    coarse_arr   : np.ndarray (H', W') — downsampled to match ref GSD
    full_arr     : np.ndarray (H, W)  — original full-resolution source
    scale_factor : float — downscale factor applied (1.0 = no change)
    """
    source_path = Path(source_path)
    ref_path    = Path(ref_path)

    gsd_source = get_gsd(source_path)
    gsd_ref    = get_gsd(ref_path)

    # Load full-resolution source
    with rasterio.open(source_path) as ds:
        full_arr  = ds.read(1).astype(np.float32)
        src_meta  = ds.meta.copy()
        src_transform = ds.transform

    if gsd_source is None or gsd_ref is None:
        print("[SCALE] Could not determine GSD for one or both images — "
              "skipping harmonization.")
        return full_arr, full_arr, 1.0

    ratio = gsd_ref / gsd_source  # > 1 means ref is coarser than source
    print(f"[SCALE] GSD source={gsd_source:.3f} m, ref={gsd_ref:.3f} m, "
          f"ratio={ratio:.2f}×")

    if ratio <= GSD_RATIO_THRESHOLD:
        print("[SCALE] GSD ratio within threshold — no harmonization needed.")
        return full_arr, full_arr, 1.0

    # Clamp downscale to MAX_DOWNSCALE_FACTOR
    scale_factor = min(ratio, MAX_DOWNSCALE_FACTOR)
    print(f"[SCALE] Downscaling source by {scale_factor:.2f}× for coarse matching...")

    coarse_arr = _gaussian_downsample(full_arr, scale_factor)

    # ── Optionally write to disk ─────────────────────────────────────────────
    if out_coarse_path is not None:
        out_coarse_path = Path(out_coarse_path)
        out_coarse_path.parent.mkdir(parents=True, exist_ok=True)

        new_h, new_w = coarse_arr.shape
        new_transform = Affine(
            src_transform.a * scale_factor,
            src_transform.b,
            src_transform.c,
            src_transform.d,
            src_transform.e * scale_factor,
            src_transform.f,
        )
        out_meta = src_meta.copy()
        out_meta.update({
            'driver': 'GTiff',
            'height': new_h,
            'width':  new_w,
            'dtype':  'uint8',
            'transform': new_transform,
            'compress': 'lzw',
        })
        with rasterio.open(out_coarse_path, 'w', **out_meta) as dst:
            dst.write(coarse_arr.astype(np.uint8), 1)
        print(f"[SCALE] Wrote coarse-scale image → {out_coarse_path} "
              f"(shape: {coarse_arr.shape})")

    return coarse_arr, full_arr, scale_factor


def upsample_offsets(
    coarse_offsets: np.ndarray,
    scale_factor: float,
) -> np.ndarray:
    """
    Scale tie-point pixel coordinates that were computed on the coarse image
    back to the full-resolution coordinate system.

    Parameters
    ----------
    coarse_offsets : (N, 4) array of (x_src, y_src, x_ref, y_ref) tie-points
    scale_factor   : the scale_factor returned by harmonize_scale()

    Returns
    -------
    (N, 4) array with source coordinates multiplied by scale_factor
    """
    if scale_factor == 1.0:
        return coarse_offsets

    full_res = coarse_offsets.copy().astype(np.float64)
    # Source coords (cols 0-1) are in the coarse image; scale them up
    full_res[:, 0] *= scale_factor  # x_src
    full_res[:, 1] *= scale_factor  # y_src
    return full_res


# ── Internal helpers ──────────────────────────────────────────────────────────

def _gaussian_downsample(arr: np.ndarray, factor: float) -> np.ndarray:
    """
    Anti-aliased downsample using Gaussian pre-filter.

    Steps:
      1. Gaussian blur with sigma = SIGMA_FACTOR × factor (prevents aliasing).
      2. Slice every `factor` pixels.
      3. Normalize to uint8 range [0, 255].
    """
    sigma = GAUSSIAN_SIGMA_FACTOR * factor
    blurred = ndimage.gaussian_filter(arr.astype(np.float32), sigma=sigma)

    # Integer step for slicing
    step = max(1, int(round(factor)))
    coarse = blurred[::step, ::step]

    # Normalize to uint8
    lo, hi = coarse.min(), coarse.max()
    if hi > lo:
        coarse = (coarse - lo) / (hi - lo) * 255.0
    else:
        coarse = np.zeros_like(coarse)

    return coarse.astype(np.float32)


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Scale-Space Harmonizer — downsample a source image to "
                    "match a reference image GSD for coarse matching."
    )
    parser.add_argument('--source', required=True,
                        help='Path to source (moving) normalized GeoTIFF')
    parser.add_argument('--ref',    required=True,
                        help='Path to reference normalized GeoTIFF')
    parser.add_argument('--output', required=True,
                        help='Path to write coarse-scale output GeoTIFF')
    args = parser.parse_args()

    coarse, full, sf = harmonize_scale(
        source_path=Path(args.source),
        ref_path=Path(args.ref),
        out_coarse_path=Path(args.output),
    )
    print(f"\nDone. Scale factor applied: {sf:.2f}×")
    print(f"  Full resolution : {full.shape}")
    print(f"  Coarse output   : {coarse.shape}")
