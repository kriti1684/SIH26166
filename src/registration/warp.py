"""
src/registration/warp.py
===========================
High-Precision Sub-Pixel Image Warper (Task 4.1).

Warps a source bounding-box raster onto the exact reference raster pixel grid 
using bicubic/spline interpolation (order=3) guided by the 3-Layer Hybrid 
Transformation (Affine + 1D Drift + Thin Plate Spline).

Preserves full georeferencing metadata (Moon CRS, affine geotransform) to produce 
a sub-pixel registered GeoTIFF.
"""

import argparse
from pathlib import Path
from typing import Union

import cv2
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling
from scipy.ndimage import map_coordinates

try:
    from src.registration.hybrid_transform import HybridTransform
except ImportError:
    from .hybrid_transform import HybridTransform


def invert_coordinates_fixedpoint(
    model: HybridTransform,
    ref_coords: np.ndarray,
    max_iters: int = 4
) -> np.ndarray:
    """
    Fixed-point Newton-Raphson inversion of the forward hybrid transformation.
    Given (x_ref, y_ref), finds (x_src, y_src) such that model.predict(x_src, y_src) == (x_ref, y_ref).
    """
    # Linear affine inverse as initial approximation
    A = model.affine_matrix  # 2x3: [ [a00, a01, tx], [a10, a11, ty] ]
    M = A[:, :2]
    t = A[:, 2]
    M_inv = np.linalg.inv(M)

    # Initial guess: x_src0 = M_inv @ (ref_coords - t)
    shifted = ref_coords - t.reshape(1, 2)
    src_guess = (M_inv @ shifted.T).T

    for _ in range(max_iters):
        pred_ref = model.predict(src_guess)
        delta_ref = ref_coords - pred_ref
        # Correction using affine Jacobian
        src_guess += (M_inv @ delta_ref.T).T

    return src_guess


def generate_composite_overlay(
    registered_path: Path,
    ref_path: Path,
    output_png: Path,
    max_dim: int = 1400
) -> Path:
    """
    Generates a false-color composite image overlay (Red = Warped Source, Green = Ref, Blue = Ref).
    Perfect alignment produces neutral monochrome tones; residual misregistration appears as color fringes.
    """
    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(registered_path) as s, rasterio.open(ref_path) as r:
        scale = min(1.0, max_dim / max(s.width, s.height))
        out_shape = (1, max(1, round(s.height * scale)), max(1, round(s.width * scale)))
        src_dec = s.read(1, out_shape=out_shape, resampling=Resampling.average)
        ref_dec = r.read(1, out_shape=out_shape, resampling=Resampling.average)

    # Normalize to 0-255 uint8 for visualization
    def to_u8(arr):
        v_min, v_max = np.percentile(arr[arr > 0], 2) if np.any(arr > 0) else 0, np.percentile(arr, 98)
        if v_max > v_min:
            scaled = np.clip((arr - v_min) / (v_max - v_min) * 255.0, 0, 255)
            return scaled.astype(np.uint8)
        return arr.astype(np.uint8)

    src_u8 = to_u8(src_dec)
    ref_u8 = to_u8(ref_dec)

    min_h = min(src_u8.shape[0], ref_u8.shape[0])
    min_w = min(src_u8.shape[1], ref_u8.shape[1])

    rgb = np.zeros((min_h, min_w, 3), dtype=np.uint8)
    rgb[..., 0] = src_u8[:min_h, :min_w]  # Red = Registered Source
    rgb[..., 1] = ref_u8[:min_h, :min_w]  # Green = Reference
    rgb[..., 2] = ref_u8[:min_h, :min_w]  # Blue = Reference

    # Convert to BGR for OpenCV saving
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_png), bgr)
    print(f"  [WARP] Generated diagnostic composite overlay: {output_png}")
    return output_png


def warp_image_subpixel(
    source_path: Union[str, Path],
    ref_path: Union[str, Path],
    transform: Union[HybridTransform, str, Path],
    output_path: Union[str, Path],
    order: int = 3,
    block_rows: int = 1024,
    nodata_val: int = 0,
    target_gsd: float = None
) -> Path:
    """
    Warps the source raster onto the reference raster grid using high-precision bicubic/spline 
    interpolation guided by the 3-layer hybrid deformation model.

    Parameters:
        source_path: Path to source GeoTIFF (e.g. harmonized bounding-box source)
        ref_path: Path to reference GeoTIFF
        transform: HybridTransform object or path to hybrid_transform_model.json
        output_path: Path to save registered GeoTIFF
        order: Spline interpolation order (3 = bicubic, 1 = bilinear)
        block_rows: Number of rows per streaming tile to minimize RAM usage
        nodata_val: Nodata fill value
        target_gsd: If provided, scales the output grid and transform to this native GSD.

    Returns:
        Path to output registered GeoTIFF
    """
    source_path = Path(source_path)
    ref_path = Path(ref_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load transform model if provided as path
    if isinstance(transform, (str, Path)):
        model = HybridTransform.load(Path(transform))
    else:
        model = transform

    if target_gsd is not None:
        with rasterio.open(ref_path) as r:
            gsd_harm = abs(r.transform.a)
        scale_factor = gsd_harm / target_gsd
        if hasattr(model, 'scale'):
            model = model.scale(scale_factor)

    # Attempt to derive exact inverse transform
    inv_model = None
    try:
        inv_model = model.get_inverse_transform()
        print("  [WARP] Inverted HybridTransform constructed (ref -> src).")
    except Exception as e:
        print(f"  [WARP] Note: Using fixed-point inversion solver ({e}).")

    print(f"[WARP] Warping {source_path.name} -> {output_path.name} (order={order})...")

    with rasterio.open(ref_path) as ref_ds, rasterio.open(source_path) as src_ds:
        ref_w = ref_ds.width
        ref_h = ref_ds.height
        src_w = src_ds.width
        src_h = src_ds.height
        ref_transform = ref_ds.transform

        if target_gsd is not None:
            scale_factor = abs(ref_transform.a) / target_gsd
            ref_w = int(np.round(ref_w * scale_factor))
            ref_h = int(np.round(ref_h * scale_factor))
            from rasterio.transform import Affine
            ref_transform = ref_transform * Affine.scale(1.0 / scale_factor, 1.0 / scale_factor)

        # Setup output profile matching reference georeferencing
        profile = ref_ds.profile.copy()
        profile.update({
            "driver": "GTiff",
            "count": 1,
            "width": ref_w,
            "height": ref_h,
            "transform": ref_transform,
            "nodata": nodata_val,
            "compress": "DEFLATE",
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
            "bigtiff": "IF_SAFER"
        })

        try:
            writer_ctx = rasterio.open(output_path, "w", **profile)
        except Exception:
            if output_path.exists():
                try:
                    writer_ctx = rasterio.open(output_path, "r+")
                    if writer_ctx.shape != (profile["height"], profile["width"]):
                        writer_ctx.close()
                        raise
                    writer_ctx.transform = profile["transform"]
                except Exception:
                    raise RuntimeError(
                        f"Cannot overwrite {output_path.name} because it is locked by another program (e.g. QGIS). "
                        f"Please close {output_path.name} in QGIS and re-run."
                    )
            else:
                raise

        with writer_ctx as dst_ds:
            for row_start in range(0, ref_h, block_rows):
                row_end = min(row_start + block_rows, ref_h)
                block_h = row_end - row_start

                # Adaptive deformation grid evaluation:
                # Direct exact evaluation for small images; uniform sub-sampled grid + bicubic for large 100+ MP rasters
                if ref_w * block_h <= 500_000:
                    cols = np.arange(ref_w, dtype=np.float64)
                    rows = np.arange(row_start, row_end, dtype=np.float64)
                    grid_x, grid_y = np.meshgrid(cols, rows)
                    pts_ref = np.column_stack([grid_x.ravel(), grid_y.ravel()])

                    if inv_model is not None:
                        pts_src = inv_model.predict(pts_ref)
                    else:
                        pts_src = invert_coordinates_fixedpoint(model, pts_ref)

                    src_x = pts_src[:, 0].reshape(block_h, ref_w)
                    src_y = pts_src[:, 1].reshape(block_h, ref_w)
                else:
                    num_cols = max(32, ref_w // 16)
                    num_rows = max(32, block_h // 16)
                    sub_cols = np.linspace(0, ref_w - 1, num_cols)
                    sub_rows = np.linspace(row_start, row_end - 1, num_rows)
                    grid_sub_x, grid_sub_y = np.meshgrid(sub_cols, sub_rows)
                    pts_sub_ref = np.column_stack([grid_sub_x.ravel(), grid_sub_y.ravel()])

                    if inv_model is not None:
                        pts_sub_src = inv_model.predict(pts_sub_ref)
                    else:
                        pts_sub_src = invert_coordinates_fixedpoint(model, pts_sub_ref)

                    src_sub_x = pts_sub_src[:, 0].reshape(num_rows, num_cols).astype(np.float32)
                    src_sub_y = pts_sub_src[:, 1].reshape(num_rows, num_cols).astype(np.float32)

                    src_x = cv2.resize(src_sub_x, (ref_w, block_h), interpolation=cv2.INTER_CUBIC)
                    src_y = cv2.resize(src_sub_y, (ref_w, block_h), interpolation=cv2.INTER_CUBIC)

                # Determine minimum required window to read from source raster
                valid_coords = (
                    np.isfinite(src_x) & np.isfinite(src_y) &
                    (src_x >= -10) & (src_x <= src_w + 10) &
                    (src_y >= -10) & (src_y <= src_h + 10)
                )

                if not np.any(valid_coords):
                    # Entire block maps outside source raster
                    dst_ds.write(np.full((block_h, ref_w), nodata_val, dtype=profile["dtype"]), 1,
                                 window=Window(0, row_start, ref_w, block_h))
                    continue

                min_x = int(np.floor(np.min(src_x[valid_coords]))) - 4
                max_x = int(np.ceil(np.max(src_x[valid_coords]))) + 4
                min_y = int(np.floor(np.min(src_y[valid_coords]))) - 4
                max_y = int(np.ceil(np.max(src_y[valid_coords]))) + 4

                # Clamp read window to source boundaries
                win_col_off = max(0, min_x)
                win_row_off = max(0, min_y)
                win_col_end = min(src_w, max_x)
                win_row_end = min(src_h, max_y)
                win_w = win_col_end - win_col_off
                win_h = win_row_end - win_row_off

                if win_w <= 0 or win_h <= 0:
                    dst_ds.write(np.full((block_h, ref_w), nodata_val, dtype=profile["dtype"]), 1,
                                 window=Window(0, row_start, ref_w, block_h))
                    continue

                read_win = Window(col_off=win_col_off, row_off=win_row_off, width=win_w, height=win_h)
                src_block_data = src_ds.read(1, window=read_win).astype(np.float32)

                # Convert global source coordinates to local window coordinates
                local_x = src_x - win_col_off
                local_y = src_y - win_row_off

                # High-precision spline interpolation
                out_block = map_coordinates(
                    src_block_data,
                    [local_y, local_x],
                    order=order,
                    mode="constant",
                    cval=nodata_val,
                    prefilter=True
                )

                # Mask out pixels that mapped beyond the actual valid window
                oob = (
                    (src_x < 0) | (src_x >= src_w) |
                    (src_y < 0) | (src_y >= src_h) |
                    ~np.isfinite(src_x) | ~np.isfinite(src_y)
                )
                out_block[oob] = nodata_val

                # Cast to profile data type
                if profile["dtype"] == "uint8":
                    out_block = np.clip(np.round(out_block), 0, 255).astype(np.uint8)
                else:
                    out_block = out_block.astype(profile["dtype"])

                dst_ds.write(out_block, 1, window=Window(0, row_start, ref_w, block_h))

    print(f"[WARP] [OK] Sub-pixel registered GeoTIFF saved: {output_path}")

    # Generate composite overlay
    diag_dir = output_path.parent / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    overlay_path = diag_dir / "warp_composite_overlay.png"
    try:
        generate_composite_overlay(output_path, ref_path, overlay_path)
    except Exception as e:
        print(f"  [WARP] Warning: Could not generate composite overlay: {e}")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Sub-Pixel GeoTIFF Warper (Phase 4.1)")
    parser.add_argument("--source", type=Path, required=True, help="Path to source GeoTIFF")
    parser.add_argument("--reference", type=Path, required=True, help="Path to reference GeoTIFF")
    parser.add_argument("--model", type=Path, required=True, help="Path to hybrid_transform_model.json")
    parser.add_argument("--output", type=Path, required=True, help="Path for output registered GeoTIFF")
    parser.add_argument("--order", type=int, default=3, help="Interpolation order (3=bicubic, 1=bilinear)")
    parser.add_argument("--block_rows", type=int, default=1024, help="Block rows for streaming I/O")
    args = parser.parse_args()

    warp_image_subpixel(
        source_path=args.source,
        ref_path=args.reference,
        transform=args.model,
        output_path=args.output,
        order=args.order,
        block_rows=args.block_rows
    )


if __name__ == "__main__":
    main()
