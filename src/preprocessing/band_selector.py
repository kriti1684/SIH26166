"""
src/preprocessing/band_selector.py
=====================================
Intelligent Hyperspectral Band Selector for Chandrayaan-2 IIRS (800nm - 5000nm).
Fulfills ISRO SIH Problem Statement 26166 for multi-modal IIRS co-registration
against LRO WAC (321nm - 689nm) and SELENE TC/MI.

Physics & Spectral Theory:
--------------------------
1. Solar Reflected SWIR Window:
   Reflected solar radiation dominates in Channels 0 to ~40 (800nm - 1250nm).
   Beyond 2500nm, thermal emission from the hot daytime lunar surface dominates,
   inverting contrast and corrupting visual feature correlation against optical sensors.
2. Spectral Proximity:
   LRO WAC Band 7 is at 689nm. The lowest IIRS bands (~800nm-850nm) have the minimal
   spectral separation (~110nm - 160nm), maximizing albedo correlation.
3. Information Entropy & SNR:
   The algorithm scans candidate SWIR bands, eliminates dead detector lines,
   and selects the band maximizing spatial entropy and standard deviation.
4. Pseudo-Panchromatic Synthesis:
   Optionally computes a radiance-weighted mean of the top K solar bands to boost
   Signal-to-Noise Ratio (SNR by factor of sqrt(K)).
"""

from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import numpy as np
import rasterio
from rasterio.windows import Window


# Nominal wavelength calibration for Chandrayaan-2 IIRS (256 bands, ~800nm to 5000nm)
# Band width approx 16.5 nm
IIRS_START_WAVELENGTH_NM = 800.0
IIRS_BAND_STEP_NM = 16.5

# Reference sensor nominal target bands (nm)
REFERENCE_TARGET_WAVELENGTHS = {
    "WAC": 689.0,      # LRO WAC 689 nm (Red/NIR edge)
    "SELENE_TC": 750.0, # SELENE Terrain Camera Panchromatic center
    "SELENE_MI": 750.0, # SELENE Multi-band Imager VIS/NIR
    "NAC": 700.0        # LRO NAC broadband panchromatic center
}


def get_iirs_band_wavelength(band_idx: int) -> float:
    """Compute nominal center wavelength (nm) for an IIRS band (0-indexed)."""
    return IIRS_START_WAVELENGTH_NM + band_idx * IIRS_BAND_STEP_NM


def compute_band_entropy(patch: np.ndarray, bins: int = 64) -> float:
    """Compute discrete Shannon information entropy of an image patch."""
    valid = patch[np.isfinite(patch) & (patch > 0)]
    if len(valid) < 100:
        return 0.0
    hist, _ = np.histogram(valid, bins=bins)
    p = hist[hist > 0] / float(len(valid))
    return float(-np.sum(p * np.log2(p)))


def select_best_band_for_reference(
    iirs_path: Path,
    reference_sensor: str = "WAC",
    max_swir_band: int = 40,
    sample_window_size: int = 512
) -> Dict[str, Any]:
    """
    Scans candidate bands in an IIRS hyperspectral raster and selects the optimal
    band for registration against the specified reference sensor.
    
    Returns:
        Dict with keys: 'selected_band' (1-indexed for rasterio), 'wavelength_nm',
                        'entropy', 'std_dev', 'target_sensor', 'candidate_scores'
    """
    iirs_path = Path(iirs_path)
    target_wavelength = REFERENCE_TARGET_WAVELENGTHS.get(reference_sensor.upper(), 689.0)

    print(f"[IIRS-BAND-SELECTOR] Analyzing hyperspectral cube: {iirs_path.name}")
    print(f"  Target Reference: {reference_sensor} (Wavelength ~{target_wavelength:.1f} nm)")

    with rasterio.open(iirs_path) as src:
        total_bands = src.count
        h, w = src.height, src.width

        # Read a central sample window to evaluate spatial quality fast
        r0 = max(0, (h - sample_window_size) // 2)
        c0 = max(0, (w - sample_window_size) // 2)
        sample_win = Window(col_off=c0, row_off=r0,
                            width=min(sample_window_size, w),
                            height=min(sample_window_size, h))

        candidate_limit = min(total_bands, max_swir_band)
        scores = []

        for b_idx in range(candidate_limit):
            band_num = b_idx + 1 # 1-indexed
            try:
                patch = src.read(band_num, window=sample_win).astype(np.float32)
                valid = patch[np.isfinite(patch) & (patch > 0)]

                if len(valid) < 100:
                    continue

                # Compute quality metrics
                std_val = float(np.std(valid))
                entropy_val = compute_band_entropy(valid)
                wavelength = get_iirs_band_wavelength(b_idx)
                
                # Spectral distance penalty: prefer bands closer to reference wavelength
                spectral_dist = abs(wavelength - target_wavelength)
                spectral_score = 1.0 / (1.0 + (spectral_dist / 500.0))

                # Composite score: balance contrast, entropy, and spectral proximity
                # Higher is better
                composite_score = (std_val * 0.4) + (entropy_val * 10.0 * 0.3) + (spectral_score * 100.0 * 0.3)

                scores.append({
                    "band_1idx": band_num,
                    "band_0idx": b_idx,
                    "wavelength_nm": wavelength,
                    "std": std_val,
                    "entropy": entropy_val,
                    "spectral_dist_nm": spectral_dist,
                    "composite_score": composite_score
                })
            except Exception as e:
                continue

        if not scores:
            print("[IIRS-BAND-SELECTOR] Warning: Could not evaluate candidate bands. Defaulting to Band 1.")
            return {
                "selected_band": 1,
                "wavelength_nm": IIRS_START_WAVELENGTH_NM,
                "target_sensor": reference_sensor
            }

        # Sort by composite score descending
        scores.sort(key=lambda x: x["composite_score"], reverse=True)
        best = scores[0]

        print(f"[IIRS-BAND-SELECTOR] [OK] Optimal band selected:")
        print(f"  Selected Band: {best['band_1idx']} (Center wavelength = {best['wavelength_nm']:.1f} nm)")
        print(f"  Entropy: {best['entropy']:.2f}, Std: {best['std']:.2f}, Score: {best['composite_score']:.2f}")

        return {
            "selected_band": best["band_1idx"],
            "wavelength_nm": best["wavelength_nm"],
            "entropy": best["entropy"],
            "std_dev": best["std"],
            "target_sensor": reference_sensor,
            "all_candidates_evaluated": len(scores)
        }


def extract_or_synthesize_band(
    iirs_path: Path,
    output_tif: Path,
    selected_band: Optional[int] = None,
    synthesize_pan: bool = False,
    num_synthesis_bands: int = 10
) -> Path:
    """
    Extracts the selected optimal band (or synthesizes a pseudo-panchromatic band)
    from an IIRS hyperspectral image and writes a clean 1-band GeoTIFF.
    """
    iirs_path = Path(iirs_path)
    output_tif = Path(output_tif)
    output_tif.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(iirs_path) as src:
        profile = src.profile.copy()
        profile.update({
            "count": 1,
            "dtype": "float32",
            "nodata": 0.0
        })

        if synthesize_pan:
            print(f"[IIRS-BAND-SELECTOR] Synthesizing Pseudo-Panchromatic band from first {num_synthesis_bands} bands...")
            bands_to_read = min(src.count, num_synthesis_bands)
            accum = np.zeros((src.height, src.width), dtype=np.float32)
            valid_counts = np.zeros((src.height, src.width), dtype=np.float32)

            for b in range(1, bands_to_read + 1):
                data = src.read(b).astype(np.float32)
                mask = np.isfinite(data) & (data > 0)
                accum[mask] += data[mask]
                valid_counts[mask] += 1.0

            valid_mask = valid_counts > 0
            accum[valid_mask] /= valid_counts[valid_mask]
            out_data = accum
        else:
            band_num = selected_band if selected_band is not None else 1
            print(f"[IIRS-BAND-SELECTOR] Extracting Band {band_num}...")
            out_data = src.read(band_num).astype(np.float32)

        with rasterio.open(output_tif, "w", **profile) as dst:
            dst.write(out_data, 1)

    print(f"[IIRS-BAND-SELECTOR] Saved processed band GeoTIFF: {output_tif.name}")
    return output_tif
