from pathlib import Path
import json
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import rasterio
import pandas as pd
from rasterio.windows import Window
from rasterio.enums import Resampling
from scipy.ndimage import map_coordinates
import cv2
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Sub-pixel Coordinate Warping (Stage 3)")
    parser.add_argument("--source_img", type=Path, required=True, help="Path to source image")
    parser.add_argument("--ref_img", type=Path, required=True, help="Path to reference image")
    parser.add_argument("--output_dir", type=Path, required=True, help="Run output directory")
    parser.add_argument("--config", type=Path, default=None, help="Optional config JSON")
    return parser.parse_args()


def run_warp(source_path: Path, ref_path: Path, output_dir: Path, custom_config: dict = None):
    output_dir = Path(output_dir)
    model_json = output_dir / "hybrid_transform_model.json"
    drift_profile_csv = output_dir / "along_track_drift_profile.csv"
    output_path = output_dir / f"{source_path.stem}_registered.tif"

    block_rows = 4000
    fixedpoint_iters = 4

    if custom_config and "warp" in custom_config:
        cfg = custom_config["warp"]
        block_rows = cfg.get("block_rows", block_rows)
        fixedpoint_iters = cfg.get("fixedpoint_iters", fixedpoint_iters)

    with open(model_json, "r", encoding="utf-8") as f:
        model = json.load(f)

    theta_rad = np.radians(model["rotation_deg"])
    scale = model["scale"]
    coeffs_x = np.array(model["tx_poly_coeffs"])
    coeffs_y = np.array(model["ty_poly_coeffs"])

    R_inv = (1.0 / scale) * np.array([
        [np.cos(theta_rad), np.sin(theta_rad)],
        [-np.sin(theta_rad), np.cos(theta_rad)],
    ])

    if drift_profile_csv.exists():
        seg_df = pd.read_csv(drift_profile_csv)
        row_domain_min = float(seg_df["row_start"].min())
        row_domain_max = float(seg_df["row_end"].max())
    else:
        row_domain_min, row_domain_max = 7000.0, 24000.0

    def poly_row_clamped(coeffs, row):
        row_clamped = np.clip(row, row_domain_min, row_domain_max)
        return np.polyval(coeffs, row_clamped)

    def inverse_map(x_out, y_out):
        y_guess = y_out.copy()
        x_guess = x_out.copy()
        for _ in range(fixedpoint_iters):
            tx = poly_row_clamped(coeffs_x, y_guess)
            ty = poly_row_clamped(coeffs_y, y_guess)
            x0 = x_out - tx
            y0 = y_out - ty
            x_guess = R_inv[0, 0] * x0 + R_inv[0, 1] * y0
            y_guess = R_inv[1, 0] * x0 + R_inv[1, 1] * y0
        return x_guess, y_guess

    print(f"[STAGE 3] Warping {source_path} -> {output_path}...")
    with rasterio.open(source_path) as src:
        profile = src.profile.copy()
        width, height = src.width, src.height
        nodata_val = 0

        profile.update(
            compress="LZW", tiled=True,
            blockxsize=512, blockysize=512, nodata=nodata_val,
        )

        with rasterio.open(output_path, "w", **profile) as dst:
            for row_start in range(0, height, block_rows):
                row_end = min(row_start + block_rows, height)
                block_h = row_end - row_start

                out_cols = np.arange(width)
                out_rows = np.arange(row_start, row_end)
                grid_x, grid_y = np.meshgrid(out_cols, out_rows)

                src_x, src_y = inverse_map(grid_x.astype(np.float64), grid_y.astype(np.float64))
                src_row_min = max(0, int(np.floor(src_y.min())) - 2)
                src_row_max = min(height, int(np.ceil(src_y.max())) + 2)

                if src_row_max <= src_row_min:
                    out_block = np.zeros((block_h, width), dtype=np.uint8)
                else:
                    read_win = Window(0, src_row_min, width, src_row_max - src_row_min)
                    source_block = src.read(1, window=read_win)

                    local_src_row = src_y - src_row_min
                    local_src_col = src_x

                    out_block = map_coordinates(
                        source_block, [local_src_row, local_src_col],
                        order=1, mode="constant", cval=nodata_val, prefilter=False,
                    ).astype(np.uint8)

                    out_of_bounds = (
                        (local_src_row < 0) | (local_src_row >= source_block.shape[0]) |
                        (local_src_col < 0) | (local_src_col >= width)
                    )
                    out_block[out_of_bounds] = nodata_val

                dst.write(out_block, 1, window=Window(0, row_start, width, block_h))

    print(f"[SUCCESS] Warped product saved: {output_path}")

    # Generate composite overlay
    try:
        decimation = 16
        with rasterio.open(output_path) as s, rasterio.open(ref_path) as r:
            out_shape = (1, max(1, s.height // decimation), max(1, s.width // decimation))
            ohrc_dec = s.read(1, out_shape=out_shape, resampling=Resampling.average)
            nac_dec = r.read(1, out_shape=out_shape, resampling=Resampling.average)

        min_h = min(ohrc_dec.shape[0], nac_dec.shape[0])
        min_w = min(ohrc_dec.shape[1], nac_dec.shape[1])
        rgb = np.zeros((min_h, min_w, 3), dtype=np.uint8)
        rgb[..., 0] = ohrc_dec[:min_h, :min_w]
        rgb[..., 1] = nac_dec[:min_h, :min_w]
        rgb[..., 2] = nac_dec[:min_h, :min_w]

        diag_dir = output_dir / "diagnostics"
        diag_dir.mkdir(parents=True, exist_ok=True)
        overlay_path = diag_dir / "final_warp_overlay.png"
        cv2.imwrite(str(overlay_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    except Exception as e:
        print(f"Warning: composite generation skipped: {e}")

    return output_path


def main():
    args = parse_args()
    custom_cfg = None
    if args.config and args.config.exists():
        with open(args.config, "r", encoding="utf-8") as f:
            custom_cfg = json.load(f)
    run_warp(args.source_img, args.ref_img, args.output_dir, custom_cfg)


if __name__ == "__main__":
    main()
