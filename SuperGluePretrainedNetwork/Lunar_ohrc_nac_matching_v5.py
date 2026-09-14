

from pathlib import Path
import csv
import json
import logging
import platform
import random
import sys
import time
import traceback
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np
import rasterio
import torch
import argparse

from models.matching import Matching

def parse_args():
    parser = argparse.ArgumentParser(description="Tiled Fine Matching (Stage 1)")
    parser.add_argument("--source_img", type=Path, required=True, help="Path to source image")
    parser.add_argument("--ref_img", type=Path, required=True, help="Path to reference image")
    parser.add_argument("--output_dir", type=Path, required=True, help="Run output directory")
    return parser.parse_args()

args = parse_args()


OHRC_PATH = args.source_img
NAC_PATH = args.ref_img
OUTPUT_DIR = args.output_dir
import shutil

TILE_DIR = OUTPUT_DIR / "tile_pngs"
VIZ_DIR = OUTPUT_DIR / "match_visualizations"

for d in (TILE_DIR, VIZ_DIR):
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)

for d in (OUTPUT_DIR, TILE_DIR, VIZ_DIR):
    d.mkdir(parents=True, exist_ok=True)


# ============================================================
# COARSE OFFSET  (now read dynamically from Stage 0)
# ============================================================

COARSE_RESULT_JSON = OUTPUT_DIR / "coarse_alignment_result.json"

if not COARSE_RESULT_JSON.exists():
    raise FileNotFoundError(
        f"Coarse alignment result not found at {COARSE_RESULT_JSON}. "
        "Please run coarse_alignment.py first."
    )

with open(COARSE_RESULT_JSON, "r", encoding="utf-8") as f:
    coarse_res = json.load(f)

# dx = nac_x - ohrc_x convention, i.e. nac_col ~= ohrc_col + OFFSET_DX
OFFSET_DX = coarse_res["offset_dx"]
OFFSET_DY = coarse_res["offset_dy"]

print(f"Loaded dynamic coarse offset: dx={OFFSET_DX:.2f}, dy={OFFSET_DY:.2f}")

# Max deviation observed across your 3 tie points was ~44 px. Margin is
# set to roughly 5-6x that, to comfortably absorb tie-point reading
# imprecision without making the NAC window unnecessarily huge (memory).
SEARCH_MARGIN_PX = 250


# ============================================================
# TILE CONFIGURATION  (unchanged from v4)
# ============================================================

TILE_SIZE = 1600
TILE_OVERLAP = 400
STEP = TILE_SIZE - TILE_OVERLAP
MIN_VALID_FRACTION = 0.10
NODATA_MARGIN_PX = 2

# NAC read window is enlarged to cover the search margin on all sides.
NAC_WINDOW_SIZE = TILE_SIZE + 2 * SEARCH_MARGIN_PX


# ============================================================
# BBOX RESTRICTION CONFIG
# ============================================================

BBOX_DECIMATION = 40   # cheap coarse scan to find OHRC's valid-data extent


# ============================================================
# SUPERPOINT / SUPERGLUE CONFIGURATION  (unchanged from v4)
# ============================================================

MAX_KEYPOINTS = 2048
NMS_RADIUS = 3
KEYPOINT_THRESHOLD = 0.005
SINKHORN_ITERATIONS = 20
MATCH_THRESHOLD = 0.2
MIN_MATCH_CONFIDENCE = 0.20
RANDOM_SEED = 42


# ============================================================
# OUTPUT OPTIONS
# ============================================================

SAVE_TILE_PNGS = True
SAVE_MATCH_VISUALIZATIONS = True
SAVE_SPATIAL_DIAGNOSTIC_PLOT = True
CSV_FLUSH_EVERY_N_TILES = 10


# ============================================================
# LOGGING
# ============================================================

LOG_PATH = OUTPUT_DIR / "run.log"
logger = logging.getLogger("lunar_matching_v5")
logger.setLevel(logging.INFO)
logger.handlers.clear()
_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
_fh = logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8")
_fh.setFormatter(_fmt)
logger.addHandler(_fh)
_ch = logging.StreamHandler(sys.stdout)
_ch.setFormatter(_fmt)
logger.addHandler(_ch)


def log(msg, level=logging.INFO):
    logger.log(level, msg)


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

log("=" * 80)
log("LUNAR OHRC-NAC TILED SUPERPOINT + SUPERGLUE (v5, offset-corrected)")
log("=" * 80)
log(f"Device: {DEVICE}")
if DEVICE == "cuda":
    log(f"GPU: {torch.cuda.get_device_name(0)}")
log(f"Coarse offset: dx={OFFSET_DX:.2f} px, dy={OFFSET_DY:.2f} px, "
    f"search margin={SEARCH_MARGIN_PX} px, NAC window={NAC_WINDOW_SIZE}px")


# ============================================================
# LOAD MODEL
# ============================================================

config = {
    "superpoint": {
        "nms_radius": NMS_RADIUS,
        "keypoint_threshold": KEYPOINT_THRESHOLD,
        "max_keypoints": MAX_KEYPOINTS,
    },
    "superglue": {
        "weights": "outdoor",
        "sinkhorn_iterations": SINKHORN_ITERATIONS,
        "match_threshold": MATCH_THRESHOLD,
    },
}

matching = Matching(config).eval().to(DEVICE)
log("SuperPoint + SuperGlue loaded.")

if DEVICE == "cuda":
    log(f"GPU memory allocated after model load: "
        f"{torch.cuda.memory_allocated() / (1024**2):.1f} MB")


# ============================================================
# RUN CONFIG DUMP
# ============================================================

run_config = {
    "ohrc_path": str(OHRC_PATH), "nac_path": str(NAC_PATH),
    "tile_size": TILE_SIZE, "tile_overlap": TILE_OVERLAP, "step": STEP,
    "offset_dx": OFFSET_DX, "offset_dy": OFFSET_DY,
    "search_margin_px": SEARCH_MARGIN_PX, "nac_window_size": NAC_WINDOW_SIZE,
    "offset_source": "mean of 3 manually-measured tie points, max deviation ~44px",
    "min_valid_fraction": MIN_VALID_FRACTION,
    "nodata_margin_px": NODATA_MARGIN_PX,
    "max_keypoints": MAX_KEYPOINTS, "nms_radius": NMS_RADIUS,
    "keypoint_threshold": KEYPOINT_THRESHOLD,
    "sinkhorn_iterations": SINKHORN_ITERATIONS,
    "match_threshold": MATCH_THRESHOLD,
    "min_match_confidence": MIN_MATCH_CONFIDENCE,
    "random_seed": RANDOM_SEED, "device": DEVICE,
    "gpu_name": torch.cuda.get_device_name(0) if DEVICE == "cuda" else None,
    "python_version": sys.version, "torch_version": torch.__version__,
    "cuda_version": torch.version.cuda, "opencv_version": cv2.__version__,
    "rasterio_version": rasterio.__version__, "platform": platform.platform(),
    "superglue_weights": "outdoor",
}
with open(OUTPUT_DIR / "run_config.json", "w", encoding="utf-8") as f:
    json.dump(run_config, f, indent=2)
log(f"Run configuration written to {OUTPUT_DIR / 'run_config.json'}")


# ============================================================
# HELPERS
# ============================================================

def read_general_window(src, row_start, col_start, height, width):
    """
    Read a (height x width) window starting at (row_start, col_start),
    which may be partially or fully outside the raster (negative start,
    or extending past the edge). Returns an array of exactly the
    requested size, zero-padded (NoData) wherever it falls outside the
    raster bounds. Handles both OHRC boundary tiles and NAC windows
    shifted by the coarse offset uniformly.
    """
    src_h, src_w = src.height, src.width

    read_row_start = max(row_start, 0)
    read_col_start = max(col_start, 0)
    read_row_end = min(row_start + height, src_h)
    read_col_end = min(col_start + width, src_w)

    out = np.zeros((height, width), dtype=np.uint8)

    if read_row_end <= read_row_start or read_col_end <= read_col_start:
        return out  # entirely outside the raster

    window = rasterio.windows.Window(
        read_col_start, read_row_start,
        read_col_end - read_col_start, read_row_end - read_row_start,
    )
    data = src.read(1, window=window)

    out_row_off = read_row_start - row_start
    out_col_off = read_col_start - col_start
    out[out_row_off:out_row_off + data.shape[0],
        out_col_off:out_col_off + data.shape[1]] = data

    return out


def valid_fraction(image):
    if image.size == 0:
        return 0.0
    return float(np.count_nonzero(image)) / float(image.size)


def to_tensor(image):
    image_float = image.astype(np.float32) / 255.0
    tensor = torch.from_numpy(image_float).unsqueeze(0).unsqueeze(0)
    return tensor.to(DEVICE)


def save_png(path, image):
    ok = cv2.imwrite(str(path), image, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise RuntimeError(f"Failed to save PNG: {path}")


def is_near_nodata(image, ix, iy, margin):
    h, w = image.shape
    y0, y1 = iy - margin, iy + margin + 1
    x0, x1 = ix - margin, ix + margin + 1
    if y0 < 0 or x0 < 0 or y1 > h or x1 > w:
        return True
    return bool(np.any(image[y0:y1, x0:x1] == 0))


def save_match_visualization(ohrc_image, nac_image, keypoints0, keypoints1,
                              valid_pairs, output_path):
    left = cv2.cvtColor(ohrc_image, cv2.COLOR_GRAY2BGR)
    right = cv2.cvtColor(nac_image, cv2.COLOR_GRAY2BGR)
    h1, w1 = left.shape[:2]
    h2, w2 = right.shape[:2]
    height = max(h1, h2)
    canvas = np.zeros((height, w1 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1] = left
    canvas[:h2, w1:w1 + w2] = right

    for idx0, idx1, confidence, is_core in valid_pairs:
        x0, y0 = keypoints0[idx0]
        x1, y1 = keypoints1[idx1]
        p0 = (int(round(x0)), int(round(y0)))
        p1 = (int(round(x1)) + w1, int(round(y1)))
        color = (0, 255, 0) if is_core else (0, 255, 255)
        cv2.line(canvas, p0, p1, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, p0, 3, (0, 0, 255), -1)
        cv2.circle(canvas, p1, 3, (0, 0, 255), -1)

    cv2.putText(canvas, "OHRC", (20, 35), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "NAC", (w1 + 20, 35), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (255, 255, 255), 2, cv2.LINE_AA)
    save_png(output_path, canvas)


def is_oom_error(exc):
    msg = str(exc).lower()
    return "out of memory" in msg or isinstance(
        exc, getattr(torch.cuda, "OutOfMemoryError", RuntimeError)
    )


def compute_ohrc_bbox(path, decimation):
    with rasterio.open(path) as src:
        h, w = src.height, src.width
        out_h, out_w = h // decimation, w // decimation
        small = src.read(
            1, out_shape=(1, out_h, out_w),
            resampling=rasterio.enums.Resampling.nearest,
        )
        mask = small > 0
        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]
        if len(rows) == 0 or len(cols) == 0:
            return None
        row_min = max(0, rows.min() * decimation - decimation)
        row_max = min(h, (rows.max() + 1) * decimation + decimation)
        col_min = max(0, cols.min() * decimation - decimation)
        col_max = min(w, (cols.max() + 1) * decimation + decimation)
        return row_min, row_max, col_min, col_max


# ============================================================
# BBOX RESTRICTION
# ============================================================

log("Computing OHRC valid-data bounding box...")
bbox = compute_ohrc_bbox(OHRC_PATH, BBOX_DECIMATION)
if bbox is None:
    raise RuntimeError("OHRC raster appears to have no valid data at all.")
bbox_row_min, bbox_row_max, bbox_col_min, bbox_col_max = bbox
log(f"OHRC bbox (row_min, row_max, col_min, col_max): {bbox}")


# ============================================================
# OPEN RASTERS
# ============================================================

with rasterio.open(OHRC_PATH) as ohrc, rasterio.open(NAC_PATH) as nac:

    if ohrc.width != nac.width or ohrc.height != nac.height:
        raise RuntimeError("OHRC and NAC dimensions differ.")
    if ohrc.crs != nac.crs or ohrc.transform != nac.transform:
        raise RuntimeError("OHRC and NAC CRS/transform differ.")

    width, height = ohrc.width, ohrc.height
    log(f"Full size: {width} x {height}")

    rows = list(range(bbox_row_min, bbox_row_max, STEP))
    cols = list(range(bbox_col_min, bbox_col_max, STEP))
    if not rows:
        rows = [bbox_row_min]
    if not cols:
        cols = [bbox_col_min]
    last_row = rows[-1]
    last_col = cols[-1]
    total_tiles = len(rows) * len(cols)

    log(f"Tile grid restricted to bbox: {len(rows)} rows x {len(cols)} cols "
        f"= {total_tiles} tiles (vs. iterating the full canvas)")

    def core_box(row, col):
        r0 = row
        r1 = bbox_row_max if row == last_row else row + STEP
        c0 = col
        c1 = bbox_col_max if col == last_col else col + STEP
        return r0, r1, c0, c1

    MATCHES_CSV = OUTPUT_DIR / "ohrc_nac_superglue_matches.csv"
    CORE_MATCHES_CSV = OUTPUT_DIR / "ohrc_nac_superglue_matches_core_only.csv"
    TILE_SUMMARY_CSV = OUTPUT_DIR / "tile_summary.csv"

    match_file = open(MATCHES_CSV, "w", newline="", encoding="utf-8")
    core_file = open(CORE_MATCHES_CSV, "w", newline="", encoding="utf-8")
    summary_file = open(TILE_SUMMARY_CSV, "w", newline="", encoding="utf-8")

    match_writer = csv.writer(match_file)
    match_writer.writerow([
        "tile_id", "tile_row", "tile_col", "ohrc_x", "ohrc_y",
        "nac_x", "nac_y", "confidence", "is_core",
    ])
    core_writer = csv.writer(core_file)
    core_writer.writerow([
        "tile_id", "tile_row", "tile_col", "ohrc_x", "ohrc_y",
        "nac_x", "nac_y", "confidence",
    ])
    summary_writer = csv.writer(summary_file)
    summary_writer.writerow([
        "tile_id", "tile_row", "tile_col",
        "ohrc_valid_fraction", "nac_valid_fraction",
        "superpoint_keypoints_ohrc", "superpoint_keypoints_nac",
        "superglue_valid_matches", "accepted_matches", "core_matches",
        "tile_runtime_sec", "status",
    ])

    try:
        processed_tiles = skipped_tiles = error_tiles = 0
        total_accepted_matches = total_core_matches = 0
        core_x_all, core_y_all = [], []
        tile_id = 0
        total_start = time.perf_counter()

        for row in rows:
            for col in cols:
                tile_id += 1
                tile_start = time.perf_counter()

                try:
                    ohrc_tile = read_general_window(ohrc, row, col, TILE_SIZE, TILE_SIZE)
                    ohrc_valid_fraction = valid_fraction(ohrc_tile)

                    if ohrc_valid_fraction < MIN_VALID_FRACTION:
                        skipped_tiles += 1
                        summary_writer.writerow([
                            tile_id, row, col, f"{ohrc_valid_fraction:.6f}", 0,
                            0, 0, 0, 0, 0, "0.0000", "skipped_low_ohrc_valid_fraction",
                        ])
                        continue

                    nac_row_start = int(round(row + OFFSET_DY)) - SEARCH_MARGIN_PX
                    nac_col_start = int(round(col + OFFSET_DX)) - SEARCH_MARGIN_PX
                    nac_tile = read_general_window(
                        nac, nac_row_start, nac_col_start,
                        NAC_WINDOW_SIZE, NAC_WINDOW_SIZE,
                    )
                    nac_valid_fraction = valid_fraction(nac_tile)

                    if nac_valid_fraction < MIN_VALID_FRACTION:
                        skipped_tiles += 1
                        log(f"[Tile {tile_id}/{total_tiles}] SKIP "
                            f"OHRC={ohrc_valid_fraction:.3f} NAC={nac_valid_fraction:.3f}")
                        summary_writer.writerow([
                            tile_id, row, col, f"{ohrc_valid_fraction:.6f}",
                            f"{nac_valid_fraction:.6f}", 0, 0, 0, 0, 0,
                            "0.0000", "skipped_low_nac_valid_fraction",
                        ])
                        continue

                    if SAVE_TILE_PNGS:
                        save_png(TILE_DIR / f"tile_{tile_id:04d}_ohrc.png", ohrc_tile)
                        save_png(TILE_DIR / f"tile_{tile_id:04d}_nac.png", nac_tile)

                    image0 = to_tensor(ohrc_tile)
                    image1 = to_tensor(nac_tile)

                    with torch.no_grad():
                        pred = matching({"image0": image0, "image1": image1})

                    keypoints0 = pred["keypoints0"][0].detach().cpu().numpy()
                    keypoints1 = pred["keypoints1"][0].detach().cpu().numpy()
                    matches0 = pred["matches0"][0].detach().cpu().numpy()
                    confidence0 = pred["matching_scores0"][0].detach().cpu().numpy()

                    r0_core, r1_core, c0_core, c1_core = core_box(row, col)
                    valid_pairs = []
                    valid_match_indices = np.where(matches0 > -1)[0]

                    for idx0 in valid_match_indices:
                        idx1 = int(matches0[idx0])
                        if idx1 < 0:
                            continue
                        confidence = float(confidence0[idx0])
                        if confidence < MIN_MATCH_CONFIDENCE:
                            continue

                        x0, y0 = keypoints0[idx0]
                        x1, y1 = keypoints1[idx1]
                        ix0, iy0 = int(round(x0)), int(round(y0))
                        ix1, iy1 = int(round(x1)), int(round(y1))

                        if not (0 <= ix0 < ohrc_tile.shape[1] and 0 <= iy0 < ohrc_tile.shape[0]):
                            continue
                        if not (0 <= ix1 < nac_tile.shape[1] and 0 <= iy1 < nac_tile.shape[0]):
                            continue
                        if is_near_nodata(ohrc_tile, ix0, iy0, NODATA_MARGIN_PX):
                            continue
                        if is_near_nodata(nac_tile, ix1, iy1, NODATA_MARGIN_PX):
                            continue

                        full_ohrc_x = col + float(x0)
                        full_ohrc_y = row + float(y0)
                        is_core = (
                            c0_core <= full_ohrc_x < c1_core
                            and r0_core <= full_ohrc_y < r1_core
                        )
                        valid_pairs.append((idx0, idx1, confidence, is_core))

                    accepted_count = len(valid_pairs)
                    core_count = sum(1 for p in valid_pairs if p[3])
                    total_valid_matches = len(valid_match_indices)

                    for idx0, idx1, confidence, is_core in valid_pairs:
                        x0, y0 = keypoints0[idx0]
                        x1, y1 = keypoints1[idx1]
                        full_ohrc_x = col + float(x0)
                        full_ohrc_y = row + float(y0)
                        full_nac_x = nac_col_start + float(x1)
                        full_nac_y = nac_row_start + float(y1)

                        row_out = [
                            tile_id, row, col,
                            f"{full_ohrc_x:.3f}", f"{full_ohrc_y:.3f}",
                            f"{full_nac_x:.3f}", f"{full_nac_y:.3f}",
                            f"{confidence:.6f}",
                        ]
                        match_writer.writerow(row_out + [int(is_core)])
                        if is_core:
                            core_writer.writerow(row_out)
                            if SAVE_SPATIAL_DIAGNOSTIC_PLOT:
                                core_x_all.append(full_ohrc_x)
                                core_y_all.append(full_ohrc_y)

                    if SAVE_MATCH_VISUALIZATIONS and accepted_count > 0:
                        save_match_visualization(
                            ohrc_tile, nac_tile, keypoints0, keypoints1,
                            valid_pairs, VIZ_DIR / f"tile_{tile_id:04d}_matches.png",
                        )

                    elapsed = time.perf_counter() - tile_start
                    summary_writer.writerow([
                        tile_id, row, col,
                        f"{ohrc_valid_fraction:.6f}", f"{nac_valid_fraction:.6f}",
                        len(keypoints0), len(keypoints1),
                        total_valid_matches, accepted_count, core_count,
                        f"{elapsed:.4f}", "ok",
                    ])

                    processed_tiles += 1
                    total_accepted_matches += accepted_count
                    total_core_matches += core_count

                    log(f"[Tile {tile_id}/{total_tiles}] "
                        f"OHRC={ohrc_valid_fraction:.2f} NAC={nac_valid_fraction:.2f} "
                        f"SP=({len(keypoints0)},{len(keypoints1)}) "
                        f"matches={accepted_count} core={core_count} time={elapsed:.2f}s")

                except Exception as exc:
                    error_tiles += 1
                    elapsed = time.perf_counter() - tile_start
                    oom = is_oom_error(exc)
                    log(f"[Tile {tile_id}/{total_tiles}] ERROR "
                        f"({'OOM' if oom else type(exc).__name__}): {exc}", level=logging.ERROR)
                    log(traceback.format_exc(), level=logging.DEBUG)
                    summary_writer.writerow([
                        tile_id, row, col, 0, 0, 0, 0, 0, 0, 0,
                        f"{elapsed:.4f}", "error_oom" if oom else "error",
                    ])
                    if DEVICE == "cuda":
                        torch.cuda.empty_cache()
                finally:
                    if tile_id % CSV_FLUSH_EVERY_N_TILES == 0:
                        match_file.flush()
                        core_file.flush()
                        summary_file.flush()

        total_elapsed = time.perf_counter() - total_start
    finally:
        match_file.close()
        core_file.close()
        summary_file.close()

    log("=" * 80)
    log("LUNAR SUPERGLUE MATCHING COMPLETE (v5)")
    log("=" * 80)
    log(f"Processed tiles       : {processed_tiles}")
    log(f"Skipped tiles         : {skipped_tiles}")
    log(f"Error tiles           : {error_tiles}")
    log(f"Accepted matches      : {total_accepted_matches}")
    log(f"Core (deduped) matches: {total_core_matches}")
    log(f"Total runtime (sec)   : {total_elapsed:.2f}")
    log(f"Matches CSV (all)     : {MATCHES_CSV}")
    log(f"Matches CSV (core)    : {CORE_MATCHES_CSV}")
    log(f"Tile summary CSV      : {TILE_SUMMARY_CSV}")

    if SAVE_SPATIAL_DIAGNOSTIC_PLOT:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(6, 6))
            ax.scatter(core_x_all, core_y_all, s=6, c="green")
            ax.set_xlim(bbox_col_min, bbox_col_max)
            ax.set_ylim(bbox_row_max, bbox_row_min)
            ax.set_title(f"Core match locations within OHRC bbox, n={len(core_x_all)}")
            ax.set_xlabel("OHRC column (px)")
            ax.set_ylabel("OHRC row (px)")
            fig.tight_layout()
            diag_path = OUTPUT_DIR / "spatial_coverage_core_matches.png"
            fig.savefig(diag_path, dpi=150)
            plt.close(fig)
            log(f"Spatial coverage diagnostic: {diag_path}")
        except ImportError:
            log("matplotlib not available; skipped diagnostic plot.", level=logging.WARNING)