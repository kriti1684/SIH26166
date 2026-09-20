"""
src/preprocessing/scale_harmonizer.py
========================================
Multi-Modal Scale & Grid Harmonizer.
Computes the minimal geographic bounding box overlap between any pair of lunar sensors
(OHRC, TMC-2, IIRS, LRO NAC, WAC, SELENE TC), crops to that bounding box, and reprojects
the source image onto the exact reference pixel grid and Ground Sampling Distance (GSD).

Result:
    - Both output rasters have identical (H, W) dimensions.
    - GSD ratio is exactly 1.0 (sub-pixel aligned grid).
    - 0% memory or compute wasted on non-overlapping territory.
"""

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
    resample_alg: int = gdal.GRA_Bilinear
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

    print(f"[BBOX-HARMONIZE] Analyzing geographic overlap between:")
    print(f"  Source: {source_path.name}")
    print(f"  Ref:    {ref_path.name}")

    overlap_info = compute_geographic_overlap(source_path, ref_path)
    if not overlap_info["has_overlap"]:
        raise ValueError(f"Images do not geographically overlap: {overlap_info.get('error')}")

    l, b, r, t = overlap_info["overlap_bounds"]
    ref_xres, ref_yres = overlap_info["ref_res"]
    ref_crs = overlap_info["overlap_crs"]

    # 1. Crop Reference raster to the exact overlap window
    print(f"[BBOX-HARMONIZE] Step 1: Cropping Reference to bounding box...")
    crop_window_to_geotiff(ref_path, overlap_info["ref_window"], ref_cropped_path)

    with rasterio.open(ref_cropped_path) as ref_crop_ds:
        target_w = ref_crop_ds.width
        target_h = ref_crop_ds.height
        target_crs = ref_crop_ds.crs.to_wkt()
        target_transform = ref_crop_ds.transform

    # 2. Warp Source raster directly onto the cropped Reference grid
    print(f"[BBOX-HARMONIZE] Step 2: Warping Source onto cropped Reference grid ({target_w}x{target_h})...")
    warp_options = gdal.WarpOptions(
        format="GTiff",
        outputBounds=[l, b, r, t],
        outputBoundsSRS=ref_crs.to_wkt() if hasattr(ref_crs, "to_wkt") else str(ref_crs),
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

    # 3. Step 3: Automatic Physical Orientation Verification & Rectification
    print(f"[BBOX-HARMONIZE] Step 3: Verifying North-Up physical orientation alignment...")
    detected_orientation = verify_and_rectify_relative_orientation(source_cammap_path, ref_cropped_path)

    # 4. Save comprehensive harmonization metadata
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
        "scale_ratio": 1.0,
        "orientation": detected_orientation,
        "crs": ref_crs.to_string() if hasattr(ref_crs, "to_string") else str(ref_crs)
    }

    with open(meta_json_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[BBOX-HARMONIZE] [OK] Scale and grid harmonization complete:")
    print(f"  Grid Size: {target_w} x {target_h} px")
    print(f"  Mutual Overlap: Source {overlap_info['src_overlap_pct']:.2f}%, Ref {overlap_info['ref_overlap_pct']:.2f}%")
    print(f"  Orientation: {detected_orientation.upper()}")
    print(f"  Source Harmonized: {source_cammap_path.name}")
    print(f"  Ref Cropped:       {ref_cropped_path.name}")

    return source_cammap_path, ref_cropped_path, metadata


def verify_and_rectify_relative_orientation(
    source_cammap_path: Path,
    ref_cropped_path: Path,
    target_height: int = 1000
) -> str:
    """
    Overview feature matching to detect if ref_cropped is upside down or mirrored.
    Checks 4 orientations ('normal', 'rot180', 'flip_v', 'flip_h') and verifies
    that the affine rotation angle aligns with North-up (|angle| < 20 deg).
    If an inversion ('rot180', 'flip_v', 'flip_h') is detected with high confidence
    while 'normal' fails, it PHYSICALLY flushes the rectified pixels to disk
    so that QGIS, diagnostic PNGs, and downstream matching engines always operate North-up!
    """
    import cv2
    import numpy as np

    try:
        with rasterio.open(source_cammap_path) as ds_s, rasterio.open(ref_cropped_path) as ds_r:
            h_s, w_s = ds_s.height, ds_s.width
            h_r, w_r = ds_r.height, ds_r.width
            th_h = min(target_height, h_s, h_r)
            ws = max(8, int(th_h * w_s / h_s))
            wr = max(8, int(th_h * w_r / h_r))
            s_arr = ds_s.read(1, out_shape=(th_h, ws), resampling=rasterio.enums.Resampling.bilinear).astype(np.float32)
            r_arr = ds_r.read(1, out_shape=(th_h, wr), resampling=rasterio.enums.Resampling.bilinear).astype(np.float32)

        def prep(a):
            v = a[a > 0]
            if len(v) < 100:
                return np.zeros(a.shape, dtype=np.uint8)
            p2, p98 = np.percentile(v, (2, 98))
            u = np.clip((a - p2) / (p98 - p2 + 1e-6) * 255.0, 0, 255).astype(np.uint8)
            return cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(u)

        s_u8 = prep(s_arr)
        r_u8 = prep(r_arr)

        sift = cv2.SIFT_create(nfeatures=2500)
        kp1, des1 = sift.detectAndCompute(s_u8, None)
        if des1 is None or len(kp1) < 15:
            return "normal"

        modes = [
            ("normal", r_u8),
            ("rot180", cv2.flip(r_u8, -1)),
            ("flip_v", cv2.flip(r_u8, 0)),
            ("flip_h", cv2.flip(r_u8, 1))
        ]

        bf = cv2.BFMatcher(cv2.NORM_L2)
        scores = {}

        for name, img in modes:
            kp2, des2 = sift.detectAndCompute(img, None)
            if des2 is None or len(kp2) < 15:
                scores[name] = 0
                continue
            matches = bf.knnMatch(des1, des2, k=2)
            good = [m for m, n in matches if len((m, n)) == 2 and m.distance < 0.75 * n.distance]
            if len(good) < 10:
                scores[name] = 0
                continue
            p1 = np.float32([kp1[m.queryIdx].pt for m in good])
            p2 = np.float32([kp2[m.trainIdx].pt for m in good])
            M, inliers = cv2.estimateAffinePartial2D(p1, p2, method=cv2.RANSAC, ransacReprojThreshold=10.0)
            if M is None or inliers is None:
                scores[name] = 0
                continue
            angle = np.rad2deg(np.arctan2(M[1, 0], M[0, 0]))
            # In North-up GIS space, the true physical orientation must be near 0 deg
            if abs(angle) < 20.0:
                scores[name] = int(np.sum(inliers))
            else:
                scores[name] = 0

        best_mode = max(scores, key=scores.get)
        best_score = scores[best_mode]
        normal_score = scores.get("normal", 0)

        if best_mode != "normal" and best_score >= 15 and best_score > 2 * normal_score:
            print(f"  [AUTO-ORIENTATION] CRITICAL: Reference raster is physically inverted/mirrored ({best_mode.upper()})! Rectifying on disk...")
            with rasterio.open(ref_cropped_path) as ds:
                profile = ds.profile.copy()
                full_data = ds.read(1)

            if best_mode == "rot180":
                full_data = np.flipud(np.fliplr(full_data))
            elif best_mode == "flip_v":
                full_data = np.flipud(full_data)
            elif best_mode == "flip_h":
                full_data = np.fliplr(full_data)

            with rasterio.open(ref_cropped_path, "w", **profile) as ds:
                ds.write(full_data, 1)

            print(f"  [AUTO-ORIENTATION] [OK] Successfully physically rectified {ref_cropped_path.name} to North-up orientation.")
            return best_mode

        return "normal"
    except Exception as e:
        print(f"  [AUTO-ORIENTATION] Warning: Orientation verification encountered error: {e}")
        return "normal"

