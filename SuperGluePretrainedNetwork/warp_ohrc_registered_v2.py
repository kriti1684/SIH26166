print("Starting warp script...", flush=True)
from pathlib import Path
import json
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

print("Importing numpy...", flush=True)
import numpy as np
print("Importing rasterio...", flush=True)
import rasterio
print("Importing pandas...", flush=True)
import pandas as pd
print("Importing rasterio windows...", flush=True)
from rasterio.windows import Window
print("Importing rasterio enums...", flush=True)
from rasterio.enums import Resampling
print("Importing scipy...", flush=True)
from scipy.ndimage import map_coordinates
print("Imports done.", flush=True)

import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_img", type=Path, required=True, help="Path to source image")
    parser.add_argument("--ref_img", type=Path, required=True, help="Path to reference image")
    parser.add_argument("--output_dir", type=Path, required=True, help="Run output directory")
    return parser.parse_args()

print("Parsing args...", flush=True)
args = parse_args()

print("Setting up paths...", flush=True)
OHRC_PATH = args.source_img
NAC_PATH = args.ref_img
MATCH_DIR = args.output_dir
MODEL_JSON = MATCH_DIR / "hybrid_transform_model.json"
DRIFT_PROFILE_CSV = MATCH_DIR / "along_track_drift_profile.csv"
OUTPUT_PATH = MATCH_DIR / f"{OHRC_PATH.stem}_registered.tif"

BLOCK_ROWS = 4000
FIXEDPOINT_ITERS = 4

print("Loading model from JSON...", flush=True)
with open(MODEL_JSON, "r", encoding="utf-8") as f:
    model = json.load(f)
print("Model loaded.", flush=True)

theta_rad = np.radians(model["rotation_deg"])
scale = model["scale"]
coeffs_x = np.array(model["tx_poly_coeffs"])
coeffs_y = np.array(model["ty_poly_coeffs"])

R = scale * np.array([
    [np.cos(theta_rad), -np.sin(theta_rad)],
    [np.sin(theta_rad), np.cos(theta_rad)],
])
# Avoid np.linalg.inv due to DLL conflicts with rasterio on Windows
R_inv = (1.0 / scale) * np.array([
    [np.cos(theta_rad), np.sin(theta_rad)],
    [-np.sin(theta_rad), np.cos(theta_rad)],
])

# Determine the actual fitted row domain from the drift profile CSV if
# available; otherwise fall back to the known match range.
if DRIFT_PROFILE_CSV.exists():
    seg_df = pd.read_csv(DRIFT_PROFILE_CSV)
    ROW_DOMAIN_MIN = float(seg_df["row_start"].min())
    ROW_DOMAIN_MAX = float(seg_df["row_end"].max())
else:
    ROW_DOMAIN_MIN, ROW_DOMAIN_MAX = 7172.0, 23975.0

print(f"Clamping polynomial evaluation to fitted row domain: "
      f"[{ROW_DOMAIN_MIN:.0f}, {ROW_DOMAIN_MAX:.0f}]", flush=True)


def poly_row_clamped(coeffs, row):
    row_clamped = np.clip(row, ROW_DOMAIN_MIN, ROW_DOMAIN_MAX)
    return np.polyval(coeffs, row_clamped)


def inverse_map(x_out, y_out):
    y_guess = y_out.copy()
    x_guess = x_out.copy()
    for _ in range(FIXEDPOINT_ITERS):
        tx = poly_row_clamped(coeffs_x, y_guess)
        ty = poly_row_clamped(coeffs_y, y_guess)
        x0 = x_out - tx
        y0 = y_out - ty
        x_guess = R_inv[0, 0] * x0 + R_inv[0, 1] * y0
        y_guess = R_inv[1, 0] * x0 + R_inv[1, 1] * y0
    return x_guess, y_guess


# ============================================================
# WARP FULL IMAGE, BLOCK BY BLOCK, WITH VALIDITY REPORTING
# ============================================================

with rasterio.open(OHRC_PATH) as src:
    profile = src.profile.copy()
    width, height = src.width, src.height
    nodata_value = 0

    profile.update(
        compress="LZW", tiled=True,
        blockxsize=512, blockysize=512, nodata=nodata_value,
    )

    print(f"\nSource size: {width} x {height}", flush=True)
    print(f"Output: {OUTPUT_PATH}\n", flush=True)
    print(f"{'block':>6s} {'out_rows':>18s} {'src_rows_sampled':>20s} "
          f"{'valid_frac':>10s}", flush=True)

    with rasterio.open(OUTPUT_PATH, "w", **profile) as dst:
        n_blocks = (height + BLOCK_ROWS - 1) // BLOCK_ROWS

        for block_i, row_start in enumerate(range(0, height, BLOCK_ROWS)):
            row_end = min(row_start + BLOCK_ROWS, height)
            block_h = row_end - row_start

            out_cols = np.arange(width)
            out_rows = np.arange(row_start, row_end)
            grid_x, grid_y = np.meshgrid(out_cols, out_rows)

            src_x, src_y = inverse_map(grid_x.astype(np.float64), grid_y.astype(np.float64))

            src_row_min = int(np.floor(src_y.min())) - 2
            src_row_max = int(np.ceil(src_y.max())) + 2
            read_row_start = max(0, src_row_min)
            read_row_end = min(height, src_row_max)

            if read_row_end <= read_row_start:
                out_block = np.zeros((block_h, width), dtype=np.uint8)
            else:
                read_window = Window(0, read_row_start, width, read_row_end - read_row_start)
                source_block = src.read(1, window=read_window)

                local_src_row = src_y - read_row_start
                local_src_col = src_x

                out_block = map_coordinates(
                    source_block, [local_src_row, local_src_col],
                    order=1, mode="constant", cval=nodata_value, prefilter=False,
                ).astype(np.uint8)

                out_of_bounds = (
                    (local_src_row < 0) | (local_src_row >= source_block.shape[0]) |
                    (local_src_col < 0) | (local_src_col >= width)
                )
                out_block[out_of_bounds] = nodata_value

            dst.write(out_block, 1, window=Window(0, row_start, width, block_h))

            valid_frac = float(np.count_nonzero(out_block)) / out_block.size
            print(f"{block_i+1:>6d} [{row_start:>7d},{row_end:>7d}) "
                  f"[{read_row_start:>7d},{read_row_end:>7d}) {valid_frac:>10.4f}")

print(f"\nDone. Registered product: {OUTPUT_PATH}")
print("\nCheck the valid_frac column above:")
print("- Blocks covering OHRC's real footprint (~row 2360-24360) should")
print("  have valid_frac roughly matching the original OHRC coverage there.")
print("- Blocks entirely outside that range should show valid_frac ~ 0.0")
print("  (correctly blank, since there was never real OHRC content there).")
print("- If a block right at the edge (~row 20000-26000) looks unexpectedly")
print("  low compared to its neighbours, that indicates content got clipped")
print("  at the transition -- worth a visual QGIS check at that boundary.")

# ============================================================
# VISUALIZATION: False-Color Composite Overlay
# ============================================================
print("\nGenerating false-color composite visualization...")

import cv2

# We downsample significantly to fit the massive image into memory
DECIMATION = 16

def read_decimated_raster(path, decimation):
    with rasterio.open(path) as src:
        out_shape = (
            1,
            max(1, src.height // decimation),
            max(1, src.width // decimation)
        )
        data = src.read(
            1, out_shape=out_shape, resampling=Resampling.average
        )
        return data

# Read warped OHRC and NAC
try:
    ohrc_warped_dec = read_decimated_raster(OUTPUT_PATH, DECIMATION)
    nac_dec = read_decimated_raster(NAC_PATH, DECIMATION)
    
    # Ensure they match perfectly in dimensions (they should, since they are on the same grid)
    # But just in case rounding causes a 1px difference, we crop to the minimum
    min_h = min(ohrc_warped_dec.shape[0], nac_dec.shape[0])
    min_w = min(ohrc_warped_dec.shape[1], nac_dec.shape[1])
    ohrc_warped_dec = ohrc_warped_dec[:min_h, :min_w]
    nac_dec = nac_dec[:min_h, :min_w]
    
    # Create RGB composite:
    # R = OHRC Warped
    # G = NAC Reference
    # B = NAC Reference
    # Where they perfectly match and overlap, it appears grayscale/yellowish
    # Where only OHRC exists, it's red. Where only NAC exists, it's cyan/grayscale.
    rgb = np.zeros((min_h, min_w, 3), dtype=np.uint8)
    rgb[..., 0] = ohrc_warped_dec
    rgb[..., 1] = nac_dec
    rgb[..., 2] = nac_dec
    
    # Enhance contrast dynamically by clipping the top 99th percentile
    rgb_max = np.percentile(rgb[rgb > 0], 99)
    if rgb_max > 0:
        rgb = np.clip((rgb.astype(np.float32) / rgb_max) * 255.0, 0, 255).astype(np.uint8)
    
    diag_dir = MATCH_DIR / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    overlay_path = diag_dir / "final_warp_overlay.png"
    
    # Convert RGB to BGR for OpenCV
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    
    # Add a title bar at the top (black background, white text)
    title = "False-Color Composite (Red=OHRC Warped, Cyan=NAC)"
    title_height = 80
    bgr_with_title = np.zeros((bgr.shape[0] + title_height, bgr.shape[1], 3), dtype=np.uint8)
    bgr_with_title[title_height:, :] = bgr
    
    cv2.putText(bgr_with_title, title, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                1.5, (255, 255, 255), 3, cv2.LINE_AA)
    
    cv2.imwrite(str(overlay_path), bgr_with_title, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    print(f"Saved visualization: {overlay_path}")

except Exception as e:
    print(f"Failed to generate false-color visualization: {e}")