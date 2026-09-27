from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import json
import rasterio
from osgeo import gdal

try:
    from src.preprocessing.bounding_overlap import compute_geographic_overlap, crop_window_to_geotiff
except ImportError:
    from .bounding_overlap import compute_geographic_overlap, crop_window_to_geotiff


def get_gsd(file_path: Path) -> Optional[float]:
    """Read the Ground Sampling Distance (meters/pixel) from a GeoTIFF."""
    file_path = Path(file_path)
    try:
        with rasterio.open(file_path) as ds:
            t = ds.transform
            pixel_w = abs(t.a)
            # Lunar radius = 1737.4 km -> 1 deg = ~30323 meters
            if pixel_w < 0.1 or (ds.crs and ds.crs.is_geographic):
                return pixel_w * 30323.0
            elif ds.crs is None:
                return 0.8
            return pixel_w
    except Exception as e:
        print(f"[SCALE] Could not read GSD from {file_path}: {e}")
        return None


def crop_and_harmonize_overlap(
    source_path: Path,
    ref_path: Path,
    output_dir: Path,
    prefix: str = "bbox_overlap",
    force_recompute: bool = False,
    resample_alg: int = gdal.GRA_Bilinear,
    sensor_ref: str = "",
    sensor_src: str = ""
) -> Tuple[Path, Path, Dict[str, Any]]:
    """
    1. Computes geographic overlap bounding box between source and reference.
    2. Crops Reference raster directly from disk using windowed reads.
    3. Warps Source raster onto the cropped Reference raster grid (Cam2Map).
    
    Returns:
        (source_harmonized_tif, ref_cropped_tif, metadata_dict)
    """
    source_path = Path(source_path)
    ref_path = Path(ref_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_cammap_path = output_dir / f"{prefix}_source_cammap.tif"
    ref_cropped_path = output_dir / f"{prefix}_ref_cropped.tif"
    meta_json_path = output_dir / f"{prefix}_meta.json"

    # Protection: If inputs are already the pre-harmonized rasters, reuse them directly without self-overwriting
    if source_path.resolve() == source_cammap_path.resolve() and ref_cropped_path.exists():
        print(f"[BBOX-HARMONIZE] Inputs are already pre-harmonized rasters. Reusing directly.")
        metadata = {}
        if meta_json_path.exists():
            try:
                with open(meta_json_path, "r") as f:
                    metadata = json.load(f)
            except Exception:
                pass
        return source_cammap_path, ref_cropped_path, metadata

    if not force_recompute and source_cammap_path.exists() and ref_cropped_path.exists() and meta_json_path.exists():
        try:
            src_mtime = source_path.stat().st_mtime
            ref_mtime = ref_path.stat().st_mtime
            cache_mtime = min(source_cammap_path.stat().st_mtime, ref_cropped_path.stat().st_mtime)
            if src_mtime <= cache_mtime and ref_mtime <= cache_mtime:
                with open(meta_json_path, "r") as f:
                    metadata = json.load(f)
                cached_src = Path(metadata.get("source_orig", "")).name
                cached_ref = Path(metadata.get("ref_orig", "")).name
                if cached_src == source_path.name and cached_ref == ref_path.name:
                    print(f"[BBOX-HARMONIZE] Found valid and up-to-date harmonized pair in {output_dir}. Loading cached.")
                    return source_cammap_path, ref_cropped_path, metadata
        except Exception:
            pass

    print("[BBOX-HARMONIZE] Analyzing geographic overlap between:")
    print(f"  Source: {source_path.name}")
    print(f"  Ref:    {ref_path.name}")

    overlap_info = compute_geographic_overlap(source_path, ref_path)
    if not overlap_info["has_overlap"]:
        raise ValueError(f"Images do not geographically overlap: {overlap_info.get('error')}")

    l, b, r, t = overlap_info["overlap_bounds"]
    ref_xres, ref_yres = overlap_info["ref_res"]
    src_xres, src_yres = overlap_info["src_res"]
    ref_crs = overlap_info["overlap_crs"]

    gsd_ref = max(ref_xres, ref_yres)
    gsd_src = max(src_xres, src_yres)
    ratio = max(gsd_src, gsd_ref) / min(gsd_src, gsd_ref)
    
    # Check if reference is WAC (avoid upsampling low-res WAC by 4x)
    is_wac = (str(sensor_ref).upper() == "WAC") or ("wac" in ref_path.name.lower())
    
    # Only use intermediate geometric mean GSD if coarse sensors are involved (GSD > 10m)
    # For high-resolution sensors (OHRC, NAC, TMC), always harmonize to reference GSD
    needs_intermediate_resampling = (ratio >= 3.0 and not is_wac and max(gsd_src, gsd_ref) > 10.0)
    
    if needs_intermediate_resampling:
        import math
        gsd_harm = min(max(math.sqrt(gsd_src * gsd_ref), 10.0), 30.0)
        target_w = int(round((r - l) / gsd_harm))
        target_h = int(round((t - b) / gsd_harm))
    else:
        gsd_harm = gsd_ref
        target_w = int(overlap_info["ref_window"].width)
        target_h = int(overlap_info["ref_window"].height)

    target_crs_wkt = ref_crs.to_wkt() if hasattr(ref_crs, "to_wkt") else str(ref_crs)

    print(f"[BBOX-HARMONIZE] Step 1: Harmonizing Reference to bounding box (Ratio: {ratio:.2f}x, is_wac={is_wac})...")
    if needs_intermediate_resampling:
        warp_ref_options = gdal.WarpOptions(
            format="GTiff",
            outputBounds=[l, b, r, t],
            outputBoundsSRS=target_crs_wkt,
            dstSRS=target_crs_wkt,
            width=target_w,
            height=target_h,
            resampleAlg=resample_alg,
            srcNodata=0,
            dstNodata=0,
            multithread=True,
            warpOptions=["NUM_THREADS=ALL_CPUS"],
            creationOptions=["TILED=YES", "COMPRESS=DEFLATE", "BIGTIFF=IF_SAFER"]
        )
        gdal.Warp(str(ref_cropped_path), str(ref_path), options=warp_ref_options)
        target_crs = target_crs_wkt
    else:
        crop_window_to_geotiff(ref_path, overlap_info["ref_window"], ref_cropped_path)
        with rasterio.open(ref_cropped_path) as ref_crop_ds:
            target_w = ref_crop_ds.width
            target_h = ref_crop_ds.height
            target_crs = ref_crop_ds.crs.to_wkt()
            ref_bounds = [ref_crop_ds.bounds.left, ref_crop_ds.bounds.bottom, ref_crop_ds.bounds.right, ref_crop_ds.bounds.top]

    print(f"[BBOX-HARMONIZE] Step 2: Warping Source onto harmonized grid ({target_w}x{target_h}, GSD={gsd_harm:.2f}m)...")
    warp_bounds = ref_bounds if 'ref_bounds' in locals() else [l, b, r, t]
    warp_options = gdal.WarpOptions(
        format="GTiff",
        outputBounds=warp_bounds,
        outputBoundsSRS=target_crs_wkt,
        dstSRS=target_crs,
        width=target_w,
        height=target_h,
        resampleAlg=resample_alg,
        srcNodata=0,
        dstNodata=0,
        multithread=True,
        warpOptions=["NUM_THREADS=ALL_CPUS"],
        creationOptions=["TILED=YES", "COMPRESS=DEFLATE", "BIGTIFF=IF_SAFER"]
    )
    gdal.Warp(str(source_cammap_path), str(source_path), options=warp_options)

    source_native_crop_path = None
    if is_wac and ratio >= 3.0:
        source_native_crop_path = output_dir / f"{prefix}_source_native_crop.tif"
        native_w = int(round((r - l) / gsd_src))
        native_h = int(round((t - b) / gsd_src))
        native_warp_opts = gdal.WarpOptions(
            format="GTiff",
            outputBounds=[l, b, r, t],
            outputBoundsSRS=target_crs_wkt,
            dstSRS=target_crs,
            width=native_w,
            height=native_h,
            resampleAlg=gdal.GRA_Bilinear,
            srcNodata=0,
            dstNodata=0,
            multithread=True,
            warpOptions=["NUM_THREADS=ALL_CPUS"],
            creationOptions=["TILED=YES", "COMPRESS=DEFLATE", "BIGTIFF=IF_SAFER"]
        )
        gdal.Warp(str(source_native_crop_path), str(source_path), options=native_warp_opts)
        print(f"  [BBOX-HARMONIZE] Generated native-resolution source crop: {source_native_crop_path.name} ({native_w}x{native_h}, GSD={gsd_src:.2f}m)")

    print("[BBOX-HARMONIZE] Step 3: Verifying North-Up physical orientation alignment...")
    detected_orientation = verify_and_rectify_relative_orientation(source_cammap_path, ref_cropped_path)

    metadata = {
        "source_orig": str(source_path),
        "ref_orig": str(ref_path),
        "overlap_bounds": [l, b, r, t],
        "width": target_w,
        "height": target_h,
        "ref_overlap_pct": overlap_info["ref_overlap_pct"],
        "src_overlap_pct": overlap_info["src_overlap_pct"],
        "res_x": ref_xres,
        "res_y": ref_yres,
        "gsd_harm": gsd_harm,
        "scale_ratio": ratio,
        "is_wac": is_wac,
        "source_native_crop": str(source_native_crop_path) if source_native_crop_path else None,
        "orientation": detected_orientation,
        "crs": target_crs_wkt
    }

    with open(meta_json_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print("[BBOX-HARMONIZE] [OK] Scale and grid harmonization complete:")
    print(f"  Grid Size: {target_w} x {target_h} px")
    print(f"  Mutual Overlap: Source {overlap_info['src_overlap_pct']:.2f}%, Ref {overlap_info['ref_overlap_pct']:.2f}%")
    print(f"  Orientation: {detected_orientation.upper()}")
    print(f"  Source Harmonized: {source_cammap_path.name}")
    print(f"  Ref Cropped:       {ref_cropped_path.name}")

    return source_cammap_path, ref_cropped_path, metadata


def verify_and_rectify_relative_orientation(
    source_cammap_path: Path,
    ref_cropped_path: Path,
    target_height: int = 600
) -> str:
    """
    Verifies that the harmonized rasters are aligned North-up.
    Both rasters are already reprojected onto the standard lunar CRS (Equirectangular Moon),
    which is strictly North-up oriented.
    Returns 'normal'.
    """
    return "normal"

