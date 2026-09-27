from pathlib import Path
from typing import Dict, Any, Optional
import rasterio
from rasterio.windows import Window
from rasterio.warp import transform_bounds


def compute_geographic_overlap(
    source_path: Path,
    ref_path: Path,
    min_overlap_pct: float = 0.001,
    padding_pixels: int = 50
) -> Dict[str, Any]:
    """
    Compute geographic intersection bounds in the reference image's CRS.
    
    Returns a dictionary containing:
        - has_overlap: bool
        - overlap_bounds: (left, bottom, right, top) in ref_crs
        - overlap_crs: CRS of reference raster
        - src_window: Window in source pixel coordinates covering overlap
        - ref_window: Window in reference pixel coordinates covering overlap
        - src_overlap_pct: Percentage of source covered by overlap
        - ref_overlap_pct: Percentage of ref covered by overlap
        - ref_res: (x_res, y_res) of reference raster in CRS units
    """
    source_path = Path(source_path)
    ref_path = Path(ref_path)

    with rasterio.open(ref_path) as ref_ds, rasterio.open(source_path) as src_ds:
        ref_crs = ref_ds.crs
        src_crs = src_ds.crs

        ref_bounds = ref_ds.bounds
        src_bounds = src_ds.bounds

        if src_crs != ref_crs and src_crs is not None and ref_crs is not None:
            src_b_in_ref = transform_bounds(src_crs, ref_crs, *src_bounds)
        else:
            src_b_in_ref = (src_bounds.left, src_bounds.bottom, src_bounds.right, src_bounds.top)

        inter_left = max(ref_bounds.left, src_b_in_ref[0])
        inter_bottom = max(ref_bounds.bottom, src_b_in_ref[1])
        inter_right = min(ref_bounds.right, src_b_in_ref[2])
        inter_top = min(ref_bounds.top, src_b_in_ref[3])

        if inter_left >= inter_right or inter_bottom >= inter_top:
            return {
                "has_overlap": False,
                "error": "No geographic overlap found between source and reference bounds."
            }

        ref_area = (ref_bounds.right - ref_bounds.left) * (ref_bounds.top - ref_bounds.bottom)
        src_area = (src_b_in_ref[2] - src_b_in_ref[0]) * (src_b_in_ref[3] - src_b_in_ref[1])
        inter_area = (inter_right - inter_left) * (inter_top - inter_bottom)

        ref_overlap_pct = (inter_area / ref_area) * 100.0 if ref_area > 0 else 0.0
        src_overlap_pct = (inter_area / src_area) * 100.0 if src_area > 0 else 0.0

        if ref_overlap_pct < min_overlap_pct and src_overlap_pct < min_overlap_pct:
            return {
                "has_overlap": False,
                "error": f"Overlap area too small (ref: {ref_overlap_pct:.3f}%, src: {src_overlap_pct:.3f}%)."
            }

        def bounds_to_window(ds, left, bottom, right, top, pad=0):
            row_top, col_left = ds.index(left, top)
            row_bottom, col_right = ds.index(right, bottom)
            
            row_start = max(0, min(row_top, row_bottom) - pad)
            row_end = min(ds.height, max(row_top, row_bottom) + pad)
            col_start = max(0, min(col_left, col_right) - pad)
            col_end = min(ds.width, max(col_left, col_right) + pad)

            return Window(
                col_off=col_start,
                row_off=row_start,
                width=max(1, col_end - col_start),
                height=max(1, row_end - row_start)
            )

        ref_win = bounds_to_window(ref_ds, inter_left, inter_bottom, inter_right, inter_top, pad=padding_pixels)

        if src_crs != ref_crs and src_crs is not None and ref_crs is not None:
            inter_b_in_src = transform_bounds(ref_crs, src_crs, inter_left, inter_bottom, inter_right, inter_top)
            src_win = bounds_to_window(src_ds, inter_b_in_src[0], inter_b_in_src[1], inter_b_in_src[2], inter_b_in_src[3], pad=padding_pixels)
        else:
            src_win = bounds_to_window(src_ds, inter_left, inter_bottom, inter_right, inter_top, pad=padding_pixels)

        ref_xres = abs(ref_ds.transform.a)
        ref_yres = abs(ref_ds.transform.e)

        return {
            "has_overlap": True,
            "overlap_bounds": (inter_left, inter_bottom, inter_right, inter_top),
            "overlap_crs": ref_crs,
            "src_window": src_win,
            "ref_window": ref_win,
            "src_overlap_pct": src_overlap_pct,
            "ref_overlap_pct": ref_overlap_pct,
            "ref_res": (ref_xres, ref_yres),
            "src_res": (abs(src_ds.transform.a), abs(src_ds.transform.e))
        }


def crop_window_to_geotiff(
    input_tif: Path,
    window: Window,
    output_tif: Path,
    indexes: Optional[int] = None
) -> Path:
    """
    Directly extracts a windowed region from input_tif and writes it to output_tif
    with preserved geospatial coordinates and updated Affine transform.
    Does NOT read or load non-overlapping portions of the raster.
    """
    input_tif = Path(input_tif)
    output_tif = Path(output_tif)
    output_tif.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(input_tif) as src:
        if indexes is not None:
            data = src.read(indexes, window=window)
            count = 1
        else:
            data = src.read(window=window)
            count = src.count

        win_transform = rasterio.windows.transform(window, src.transform)
        profile = src.profile.copy()
        profile.update({
            "height": int(window.height),
            "width": int(window.width),
            "transform": win_transform,
            "count": count
        })

        try:
            with rasterio.open(output_tif, "w", **profile) as dst:
                if count == 1 and data.ndim == 2:
                    dst.write(data, 1)
                else:
                    dst.write(data)
        except Exception:
            if output_tif.exists():
                try:
                    with rasterio.open(output_tif, "r+") as dst:
                        if dst.shape == (int(window.height), int(window.width)):
                            dst.transform = win_transform
                            if count == 1 and data.ndim == 2:
                                dst.write(data, 1)
                            else:
                                dst.write(data)
                        else:
                            raise
                except Exception:
                    raise RuntimeError(
                        f"Cannot overwrite {output_tif.name} because it is currently opened by another application (e.g. QGIS). "
                        f"Please remove or close {output_tif.name} in QGIS and re-run."
                    )
            else:
                raise

    return output_tif
