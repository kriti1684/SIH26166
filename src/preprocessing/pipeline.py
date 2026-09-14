"""
src/preprocessing/pipeline.py
================================
Multi-Sensor Ingestion & Preprocessing Pipeline — 100% ISIS-free, WSL-free.

Previously this file contained:
  - WSL subprocess bridging (wsl -d Ubuntu-24.04 ... mamba run ... isisimport)
  - lronac2isis / lrowac2isis / spiceinit / cam2map ISIS3 commands
  - to_wsl_path() helpers and DEFAULT_WSL_* environment variables

All of that is now replaced by:
  - ingest.py         → Pure-Python rasterio/GDAL raster reader
  - spice_georeference.py → Native SpiceyPy ray-tracer (no WSL, no ISIS)
  - normalizer.py     → uint8 contrast normalizer

CLI Usage:
    python -m src.preprocessing.pipeline \\
        --input ./data/raw_data/ch2_ohrc_20211014T072133.xml \\
        --sensor OHRC \\
        --output_dir ./normalized

    python -m src.preprocessing.pipeline \\
        --input ./data/raw_data/M162680801LE.IMG \\
        --sensor NAC \\
        --output_dir ./normalized
"""

import argparse
import sys
from pathlib import Path

from .ingest import (
    load_raster,
    inspect_projection,
    extract_pds4_bounds,
    extract_pds4_metadata,
    write_raw_tif,
    MOON_CRS,
)
from .normalizer import normalize_tiff
from .spice_georeference import fallback_4_corner, compute_gcps, apply_gcps_gdal

# ── Kernel search paths (Windows & Linux, no WSL needed) ─────────────────────
import os
_DEFAULT_KERNEL_DIR = Path(
    os.environ.get(
        "SPICE_KERNEL_DIR",
        str(Path(__file__).parents[3] / "data" / "spice_kernels"),
    )
)

_SENSOR_KERNEL_MAP = {
    "OHRC":  "ch2_ohr_v01.ti",
    "TMC":   "ch2_tmc_v01.ti",
    "TMC2":  "ch2_tmc_v01.ti",
    "IIRS":  "ch2_iir_v01.ti",
}

_COMMON_KERNELS = [
    "naif0012.tls",
    "pck00010.tpc",
    "ch2_sclk_v1.tsc",
    "ch2_v01.tf",
    "de430s.bsp",
]


def _find_local_kernels(sensor: str) -> list[Path]:
    """
    Discover SPICE kernels stored locally in data/spice_kernels/.

    Returns a list of existing kernel paths. If none found, returns [].
    This triggers the automatic fallback to the 4-corner GCP method in
    spice_georeference.py.
    """
    found = []
    base = _DEFAULT_KERNEL_DIR
    if not base.exists():
        print(f"[PIPELINE] Kernel directory not found: {base}  → will use 4-corner fallback.")
        return []

    # Common kernels (LSK, PCK, SCLK, FK, SPK)
    for kname in _COMMON_KERNELS:
        for candidate in base.rglob(kname):
            found.append(candidate)
            break  # take first match

    # Instrument-specific kernel (IK)
    ik_name = _SENSOR_KERNEL_MAP.get(sensor.upper())
    if ik_name:
        for candidate in base.rglob(ik_name):
            found.append(candidate)
            break

    # Also pick up any .bc (CK attitude) or .bsp (SPK trajectory) kernels
    for ext in ("*.bc", "*.bsp"):
        for candidate in base.rglob(ext):
            if candidate not in found:
                found.append(candidate)

    if found:
        print(f"[PIPELINE] Found {len(found)} SPICE kernels in {base}")
    else:
        print(f"[PIPELINE] No SPICE kernels found in {base}  → will use 4-corner fallback.")

    return found


# ── Sensor processing functions ───────────────────────────────────────────────

def process_lro(input_file: Path, sensor: str, output_dir: Path) -> Path:
    """
    Ingest and normalize a LRO NAC or WAC image — no ISIS required.

    Strategy
    --------
    1. Check if the file is already a map-projected GeoTIFF (QuickMap downloads).
       → If yes, pass directly to normalizer.

    2. If raw PDS3 (.IMG) → GDAL's native PDS driver reads image + metadata
       without lronac2isis. The raster is unprojected (Level-1).
       → Normalizer rescales to uint8. No georeferencing is applied at this
         stage; coarse matching in SuperPoint handles the alignment.

    Note: Full SPICE georeferencing for LRO requires lronac2isis / lrowac2isis
    kernels. Those are available if you download the LRO data as GeoTIFF from
    https://quickmap.lroc.asu.edu/ (recommended for SIH demos).
    """
    stem = input_file.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    norm_tif = output_dir / f"{stem}_normalized.tif"

    proj_info = inspect_projection(input_file)

    if proj_info['is_projected']:
        print(f"[PIPELINE] {input_file.name} is already map-projected "
              f"(GSD ≈ {proj_info['gsd_m']:.2f} m) — normalizing directly.")
        normalize_tiff(input_file, norm_tif, is_ohrc=False)
    else:
        print(f"[PIPELINE] {input_file.name} appears unprojected — "
              f"reading via rasterio and normalizing without georeference.")
        # Still useful for feature matching (SuperPoint is position-agnostic)
        normalize_tiff(input_file, norm_tif, is_ohrc=False)

    print(f"[PIPELINE] LRO {sensor} → {norm_tif}")
    return norm_tif


def process_ch2(input_file: Path, sensor: str, output_dir: Path) -> Path:
    """
    Ingest and georeference a Chandrayaan-2 OHRC / TMC-2 / IIRS image.

    Strategy (dual-engine, fully native):
    ──────────────────────────────────────
    Engine 1 — SPICE (High Precision):
        If SPICE kernels exist in data/spice_kernels/, run sincpt() ray-tracing
        to produce dense GCPs → GDAL TPS warp → Equirectangular Moon GeoTIFF.

    Engine 2 — 4-Corner Fallback (Zero-SPICE):
        If kernels are absent, extract lat/lon corners from the PDS4 XML
        <isda:System_Level_Coordinates> block and apply a simple affine warp.

    Both engines produce a valid GeoTIFF with Moon Equirectangular CRS that
    can be matched against LRO data.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_file.stem

    # ── Determine the linked .img raster path ──────────────────────────────
    xml_file = input_file if input_file.suffix.lower() == '.xml' else None
    if xml_file is None:
        # If user passed the .img directly, look for sibling .xml
        sibling_xml = input_file.with_suffix('.xml')
        xml_file = sibling_xml if sibling_xml.exists() else None

    raw_img = input_file.with_suffix('.img')
    if not raw_img.exists():
        raw_img = input_file.with_suffix('.IMG')

    raw_tif_path  = output_dir / f"{stem}_raw.tif"
    norm_tif_path = output_dir / f"{stem}_normalized.tif"

    # ── Check if already projected ─────────────────────────────────────────
    proj_info = inspect_projection(input_file)
    if proj_info['is_projected']:
        print(f"[PIPELINE] {input_file.name} is already map-projected — normalizing.")
        normalize_tiff(input_file, norm_tif_path, is_ohrc=(sensor == "OHRC"))
        return norm_tif_path

    # ── Load raw raster into a plain TIF (needed for gdal_translate GCPs) ──
    print(f"[PIPELINE] Loading raw raster: {raw_img if raw_img.exists() else input_file}")
    src_path = raw_img if raw_img.exists() else input_file
    arr, ds = load_raster(src_path)
    if ds is not None:
        ds.close()
    write_raw_tif(arr, raw_tif_path)

    # ── Try SPICE engine first ─────────────────────────────────────────────
    georef_ok = False
    if xml_file is not None:
        kernels = _find_local_kernels(sensor)
        if kernels:
            print(f"[PIPELINE] Running SPICE engine for {sensor}...")
            try:
                gcps = compute_gcps(
                    xml_path=xml_file,
                    width=arr.shape[1],
                    height=arr.shape[0],
                    kernel_paths=[str(k) for k in kernels],
                    sensor=sensor if sensor in ('OHRC', 'TMC', 'IIRS') else 'OHRC',
                    step=100,
                )
                if gcps:
                    apply_gcps_gdal(xml_file, raw_tif_path, output_dir / f"{stem}_georef.tif", gcps)
                    normalize_tiff(
                        output_dir / f"{stem}_georef.tif",
                        norm_tif_path,
                        is_ohrc=(sensor == "OHRC"),
                    )
                    georef_ok = True
            except Exception as e:
                print(f"[PIPELINE] SPICE engine failed: {e}. Falling back to 4-corner method.")

        # ── Fallback: 4-corner affine georeferencing from XML ────────────
        if not georef_ok:
            print(f"[PIPELINE] Running 4-corner fallback for {sensor}...")
            success = fallback_4_corner(xml_file, raw_tif_path, output_dir / f"{stem}_georef.tif")
            if success:
                normalize_tiff(
                    output_dir / f"{stem}_georef.tif",
                    norm_tif_path,
                    is_ohrc=(sensor == "OHRC"),
                )
                georef_ok = True

    # ── Last resort: normalize raw without georeference ────────────────────
    if not georef_ok:
        print("[PIPELINE] WARNING: Could not georeference. "
              "Normalizing raw image without projection — "
              "coarse SuperPoint matching will still work.")
        normalize_tiff(raw_tif_path, norm_tif_path, is_ohrc=(sensor == "OHRC"))

    # ── Cleanup intermediates ──────────────────────────────────────────────
    for tmp in [raw_tif_path]:
        if tmp.exists():
            tmp.unlink()

    print(f"[PIPELINE] CH-2 {sensor} → {norm_tif_path}")
    return norm_tif_path


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Multi-Sensor Ingestion & Preprocessing Pipeline\n"
            "100%% Python — no ISIS3, no WSL, no external dependencies.\n\n"
            "Supported sensors:\n"
            "  NAC   — LRO Narrow Angle Camera       (0.5–1.2 m)\n"
            "  WAC   — LRO Wide Angle Camera          (100 m)\n"
            "  OHRC  — Chandrayaan-2 High Res Camera  (0.25 m)\n"
            "  TMC2  — Chandrayaan-2 Terrain Mapping  (5.0 m)\n"
            "  IIRS  — Chandrayaan-2 IR Imaging Spec  (8–20 m)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input", required=True, type=Path,
        help="Path to raw image file (.IMG, .xml, .h5, .tif)",
    )
    parser.add_argument(
        "--sensor", required=True,
        choices=["NAC", "WAC", "OHRC", "TMC2", "IIRS"],
        help="Instrument identifier",
    )
    parser.add_argument(
        "--output_dir", default=Path("./normalized"), type=Path,
        help="Directory to write normalized GeoTIFF (default: ./normalized)",
    )
    args = parser.parse_args()

    if args.sensor in ("NAC", "WAC"):
        out = process_lro(args.input, args.sensor, args.output_dir)
    else:
        out = process_ch2(args.input, args.sensor, args.output_dir)

    print(f"\n[PIPELINE] ✓ Done: {out}")


if __name__ == "__main__":
    main()
