from pathlib import Path
import csv
import json
import logging
import platform
import random
import sys
import time
import os
import shutil
import argparse

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np
import rasterio
import torch

try:
    from src.models.matching import Matching
except ImportError:
    from ..models.matching import Matching


def parse_args():
    parser = argparse.ArgumentParser(description="Tiled Fine Matching (Stage 1)")
    parser.add_argument("--source_img", type=Path, required=True, help="Path to source image")
    parser.add_argument("--ref_img", type=Path, required=True, help="Path to reference image")
    parser.add_argument("--output_dir", type=Path, required=True, help="Run output directory")
    parser.add_argument("--config", type=Path, default=None, help="Path to custom JSON config")
    return parser.parse_args()


def run_tiled_matching(source_path: Path, ref_path: Path, output_dir: Path, custom_config: dict = None):
    output_dir = Path(output_dir)
    tile_dir = output_dir / "tile_pngs"
    viz_dir = output_dir / "match_visualizations"

    for d in (tile_dir, viz_dir):
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
    for d in (output_dir, tile_dir, viz_dir):
        d.mkdir(parents=True, exist_ok=True)

    coarse_result_json = output_dir / "coarse_alignment_result.json"
    if not coarse_result_json.exists():
        raise FileNotFoundError(
            f"Coarse alignment result not found at {coarse_result_json}. Run Stage 0 first."
        )

    with open(coarse_result_json, "r", encoding="utf-8") as f:
        coarse_res = json.load(f)

    offset_dx = coarse_res["offset_dx"]
    offset_dy = coarse_res["offset_dy"]
    print(f"[STAGE 1] Loaded coarse offset: dx={offset_dx:.2f} px, dy={offset_dy:.2f} px")

    # Config parameters
    tile_size = 1600
    tile_overlap = 400
    search_margin_px = 250
    min_valid_fraction = 0.10
    nodata_margin_px = 2
    max_keypoints = 2048
    nms_radius = 3
    keypoint_threshold = 0.005
    sinkhorn_iterations = 20
    match_threshold = 0.2
    min_match_confidence = 0.20
    random_seed = 42

    if custom_config and "tiled_matching" in custom_config:
        cfg = custom_config["tiled_matching"]
        tile_size = cfg.get("tile_size", tile_size)
        tile_overlap = cfg.get("tile_overlap", tile_overlap)
        search_margin_px = cfg.get("search_margin_px", search_margin_px)
        max_keypoints = cfg.get("max_keypoints", max_keypoints)

    step = tile_size - tile_overlap
    nac_window_size = tile_size + 2 * search_margin_px

    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    model_config = {
        "superpoint": {
            "nms_radius": nms_radius,
            "keypoint_threshold": keypoint_threshold,
            "max_keypoints": max_keypoints,
        },
        "superglue": {
            "weights": "outdoor",
            "sinkhorn_iterations": sinkhorn_iterations,
            "match_threshold": match_threshold,
        },
    }
    matching = Matching(model_config).eval().to(device)

    def read_general_window(src, row_start, col_start, height, width):
        src_h, src_w = src.height, src.width
        read_row_start = max(row_start, 0)
        read_col_start = max(col_start, 0)
        read_row_end = min(row_start + height, src_h)
        read_col_end = min(col_start + width, src_w)
        out = np.zeros((height, width), dtype=np.uint8)
        if read_row_end <= read_row_start or read_col_end <= read_col_start:
            return out
        window = rasterio.windows.Window(
            read_col_start, read_row_start,
            read_col_end - read_col_start, read_row_end - read_row_start,
        )
        data = src.read(1, window=window)
        out_row_off = read_row_start - row_start
        out_col_off = read_col_start - col_start
        out[out_row_off:out_row_off + data.shape[0], out_col_off:out_col_off + data.shape[1]] = data
        return out

    def to_tensor(image):
        return torch.from_numpy(image.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)

    def compute_bbox(path, decimation=40):
        with rasterio.open(path) as src:
            h, w = src.height, src.width
            small = src.read(1, out_shape=(1, h // decimation, w // decimation), resampling=rasterio.enums.Resampling.nearest)
            mask = small > 0
            rows = np.where(mask.any(axis=1))[0]
            cols = np.where(mask.any(axis=0))[0]
            if len(rows) == 0 or len(cols) == 0:
                return None
            return (
                max(0, rows.min() * decimation - decimation),
                min(h, (rows.max() + 1) * decimation + decimation),
                max(0, cols.min() * decimation - decimation),
                min(w, (cols.max() + 1) * decimation + decimation),
            )

    bbox = compute_bbox(source_path)
    if bbox is None:
        raise RuntimeError("Source raster has no valid pixels.")
    bbox_row_min, bbox_row_max, bbox_col_min, bbox_col_max = bbox

    with rasterio.open(source_path) as src_r, rasterio.open(ref_path) as ref_r:
        rows = list(range(bbox_row_min, bbox_row_max, step)) or [bbox_row_min]
        cols = list(range(bbox_col_min, bbox_col_max, step)) or [bbox_col_min]
        last_row, last_col = rows[-1], cols[-1]
        total_tiles = len(rows) * len(cols)

        matches_csv = output_dir / "ohrc_nac_superglue_matches.csv"
        core_matches_csv = output_dir / "ohrc_nac_superglue_matches_core_only.csv"
        summary_csv = output_dir / "tile_summary.csv"

        with open(matches_csv, "w", newline="", encoding="utf-8") as f_match, \
             open(core_matches_csv, "w", newline="", encoding="utf-8") as f_core, \
             open(summary_csv, "w", newline="", encoding="utf-8") as f_sum:

            match_writer = csv.writer(f_match)
            match_writer.writerow(["tile_id", "tile_row", "tile_col", "ohrc_x", "ohrc_y", "nac_x", "nac_y", "confidence", "is_core"])
            core_writer = csv.writer(f_core)
            core_writer.writerow(["tile_id", "tile_row", "tile_col", "ohrc_x", "ohrc_y", "nac_x", "nac_y", "confidence"])
            sum_writer = csv.writer(f_sum)
            sum_writer.writerow(["tile_id", "tile_row", "tile_col", "ohrc_valid_frac", "nac_valid_frac", "kp0", "kp1", "matches", "core_matches", "runtime_s", "status"])

            tile_id = 0
            total_core_matches = 0

            for row in rows:
                for col in cols:
                    tile_id += 1
                    t0 = time.perf_counter()
                    try:
                        ohrc_tile = read_general_window(src_r, row, col, tile_size, tile_size)
                        v_ohrc = float(np.count_nonzero(ohrc_tile)) / ohrc_tile.size
                        if v_ohrc < min_valid_fraction:
                            sum_writer.writerow([tile_id, row, col, f"{v_ohrc:.4f}", 0, 0, 0, 0, 0, "0.0", "skip_ohrc_nodata"])
                            continue

                        nac_r0 = int(round(row + offset_dy)) - search_margin_px
                        nac_c0 = int(round(col + offset_dx)) - search_margin_px
                        nac_tile = read_general_window(ref_r, nac_r0, nac_c0, nac_window_size, nac_window_size)
                        v_nac = float(np.count_nonzero(nac_tile)) / nac_tile.size
                        if v_nac < min_valid_fraction:
                            sum_writer.writerow([tile_id, row, col, f"{v_ohrc:.4f}", f"{v_nac:.4f}", 0, 0, 0, 0, "0.0", "skip_nac_nodata"])
                            continue

                        cv2.imwrite(str(tile_dir / f"tile_{tile_id:04d}_ohrc.png"), ohrc_tile)
                        cv2.imwrite(str(tile_dir / f"tile_{tile_id:04d}_nac.png"), nac_tile)

                        with torch.no_grad():
                            pred = matching({"image0": to_tensor(ohrc_tile), "image1": to_tensor(nac_tile)})

                        kp0 = pred["keypoints0"][0].detach().cpu().numpy()
                        kp1 = pred["keypoints1"][0].detach().cpu().numpy()
                        matches0 = pred["matches0"][0].detach().cpu().numpy()
                        conf0 = pred["matching_scores0"][0].detach().cpu().numpy()

                        r0_core, r1_core = row, (bbox_row_max if row == last_row else row + step)
                        c0_core, c1_core = col, (bbox_col_max if col == last_col else col + step)

                        valid_idx = np.where(matches0 > -1)[0]
                        core_count = 0
                        for idx0 in valid_idx:
                            idx1 = int(matches0[idx0])
                            if idx1 < 0: continue
                            conf = float(conf0[idx0])
                            if conf < min_match_confidence: continue

                            x0, y0 = kp0[idx0]
                            x1, y1 = kp1[idx1]
                            full_x0 = col + float(x0)
                            full_y0 = row + float(y0)
                            full_x1 = nac_c0 + float(x1)
                            full_y1 = nac_r0 + float(y1)

                            is_core = (c0_core <= full_x0 < c1_core and r0_core <= full_y0 < r1_core)
                            row_out = [tile_id, row, col, f"{full_x0:.3f}", f"{full_y0:.3f}", f"{full_x1:.3f}", f"{full_y1:.3f}", f"{conf:.4f}"]
                            match_writer.writerow(row_out + [int(is_core)])
                            if is_core:
                                core_writer.writerow(row_out)
                                core_count += 1

                        elapsed = time.perf_counter() - t0
                        total_core_matches += core_count
                        sum_writer.writerow([tile_id, row, col, f"{v_ohrc:.3f}", f"{v_nac:.3f}", len(kp0), len(kp1), len(valid_idx), core_count, f"{elapsed:.2f}", "ok"])
                        print(f"[{tile_id}/{total_tiles}] Core matches: {core_count}, Elapsed: {elapsed:.2f}s")
                    except Exception as e:
                        print(f"Error on tile {tile_id}: {e}")

    print(f"\n[SUCCESS] Tiled matching complete. Total core matches: {total_core_matches}")
    return core_matches_csv


def main():
    args = parse_args()
    custom_cfg = None
    if args.config and args.config.exists():
        with open(args.config, "r", encoding="utf-8") as f:
            custom_cfg = json.load(f)
    run_tiled_matching(args.source_img, args.ref_img, args.output_dir, custom_cfg)


if __name__ == "__main__":
    main()
