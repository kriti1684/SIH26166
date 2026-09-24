"""
src/registration/tiled_matching.py
=======================================
Uniform Spatial Grid Tiler with Entropy Enforcement (Task 3.1).

This module partitions the overlapping BBox region into an N x M grid,
applies the coarse global offset to pre-position the search windows, and
extracts feature matches per-tile using structural maps. 
Supports both in-memory arrays and GeoTIFF streaming reads (Architectural Pillar 1).
It enforces uniform spatial distribution across the image grid, evaluated
using Shannon Entropy H(S).
"""

import csv
import math
from pathlib import Path
from typing import List, Tuple, Dict, Any, Union, Optional

import cv2
import numpy as np
import rasterio
from rasterio.windows import Window

try:
    from src.preprocessing.structural import compute_structural_representation
except ImportError:
    try:
        from ..preprocessing.structural import compute_structural_representation
    except ImportError:
        compute_structural_representation = None

try:
    from src.registration.loftr_matcher import LoFTRMatcher
except ImportError:
    try:
        from .loftr_matcher import LoFTRMatcher
    except ImportError:
        LoFTRMatcher = None

# ─── Constants ───────────────────────────────────────────────────────────────

DEFAULT_GRID = (4, 4)
MIN_POINTS_PER_TILE = 5
LOWE_RATIO = 0.75

# ─── Core Tiled Matching ────────────────────────────────────────────────────

def get_tile_bounds(
    image_shape: Tuple[int, int],
    tile_size: Union[int, Tuple[int, int]] = 1600,
    step_size: Optional[int] = None
) -> List[Dict[str, int]]:
    """
    Partition the image into overlapping tiles.
    Supports both (tile_size, step_size) pixel windows and grid_size=(rows, cols) tuples.
    """
    h, w = image_shape[:2]
    
    if isinstance(tile_size, (tuple, list)):
        n_rows, n_cols = tile_size
        step_y = max(1, h // n_rows)
        step_x = max(1, w // n_cols)
        tiles = []
        for r in range(n_rows):
            for c in range(n_cols):
                y0 = r * step_y
                x0 = c * step_x
                y1 = h if r == n_rows - 1 else (r + 1) * step_y
                x1 = w if c == n_cols - 1 else (c + 1) * step_x
                tiles.append({
                    "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                    "core_x0": x0, "core_y0": y0, "core_x1": x1, "core_y1": y1
                })
        return tiles

    if step_size is None:
        step_size = int(tile_size * 0.75)
    margin = (tile_size - step_size) // 2
    tiles = []
    
    for y0 in range(0, h, step_size):
        for x0 in range(0, w, step_size):
            y1 = min(h, y0 + tile_size)
            x1 = min(w, x0 + tile_size)
            
            # Require minimum size (e.g., at least 200px)
            if y1 - y0 >= 200 and x1 - x0 >= 200:
                core_x0 = x0 + margin if x0 > 0 else x0
                core_x1 = x1 - margin if x1 < w else x1
                core_y0 = y0 + margin if y0 > 0 else y0
                core_y1 = y1 - margin if y1 < h else y1
                
                tiles.append({
                    "x0": x0,
                    "y0": y0,
                    "x1": x1,
                    "y1": y1,
                    "w": x1 - x0,
                    "h": y1 - y0,
                    "core_x0": core_x0,
                    "core_y0": core_y0,
                    "core_x1": core_x1,
                    "core_y1": core_y1
                })
    if not tiles and h > 0 and w > 0:
        tiles.append({
            "x0": 0, "y0": 0, "x1": w, "y1": h,
            "w": w, "h": h,
            "core_x0": 0, "core_y0": 0, "core_x1": w, "core_y1": h
        })
    return tiles


def _match_loftr(s_img, r_img, loftr_matcher):
    if loftr_matcher is None:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)
        
    def to_u8_norm(arr):
        if arr.dtype == np.uint8:
            u8 = arr
        elif arr.max() <= 1.0:
            u8 = (arr * 255).astype(np.uint8)
        else:
            v = arr[arr > 0]
            if len(v) < 100:
                u8 = np.clip(arr, 0, 255).astype(np.uint8)
            else:
                p2, p98 = np.percentile(v, (2, 98))
                u8 = np.clip((arr - p2) / (p98 - p2 + 1e-6) * 255.0, 0, 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        return clahe.apply(u8)
        
    s_u8 = to_u8_norm(s_img)
    r_u8 = to_u8_norm(r_img)
    h_s, w_s = s_u8.shape
    h_r, w_r = r_u8.shape
    max_dim = 1024.0
    
    # Common isotropic scale factor so crater feature sizes match 1:1 in LoFTR
    common_scale = min(1.0, max_dim / max(h_s, w_s, h_r, w_r))
    
    nw_s = max(8, int(round(w_s * common_scale / 8.0)) * 8)
    nh_s = max(8, int(round(h_s * common_scale / 8.0)) * 8)
    s_u8_proc = cv2.resize(s_u8, (nw_s, nh_s), interpolation=cv2.INTER_AREA) if (nw_s != w_s or nh_s != h_s) else s_u8
    scale_s_x = w_s / nw_s
    scale_s_y = h_s / nh_s

    nw_r = max(8, int(round(w_r * common_scale / 8.0)) * 8)
    nh_r = max(8, int(round(h_r * common_scale / 8.0)) * 8)
    r_u8_proc = cv2.resize(r_u8, (nw_r, nh_r), interpolation=cv2.INTER_AREA) if (nw_r != w_r or nh_r != h_r) else r_u8
    scale_r_x = w_r / nw_r
    scale_r_y = h_r / nh_r
    
    src_pts, ref_pts, confs = loftr_matcher.match(s_u8_proc, r_u8_proc)
    
    if len(src_pts) > 0:
        high_conf = confs >= 0.25
        if np.sum(high_conf) >= 8:
            valid = high_conf
        else:
            valid = confs >= 0.18
        src_pts = src_pts[valid]
        ref_pts = ref_pts[valid]
        
    if len(src_pts) > 0:
        src_pts[:, 0] *= scale_s_x
        src_pts[:, 1] *= scale_s_y
        ref_pts[:, 0] *= scale_r_x
        ref_pts[:, 1] *= scale_r_y
        
    return src_pts, ref_pts


def match_tile(
    src_tile: np.ndarray,
    ref_tile: np.ndarray,
    method: str = "loftr",
    loftr_matcher = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract and match features between two tiles using LoFTR.
    """
    if method.lower() in ["loftr", "ensemble"] and loftr_matcher is not None:
        src_pts, ref_pts = _match_loftr(src_tile, ref_tile, loftr_matcher)
    else:
        raise ValueError(f"Unsupported or uninitialized matching method '{method}'. LoFTR is required for multi-modal lunar registration.")

    if len(src_pts) < 4:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)

    # Local tile RANSAC to remove outliers
    _, mask = cv2.estimateAffinePartial2D(src_pts, ref_pts, method=cv2.RANSAC, ransacReprojThreshold=5.0)
    
    if mask is None:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)
        
    mask = mask.ravel() == 1
    
    return src_pts[mask].astype(np.float32), ref_pts[mask].astype(np.float32)


# ─── Entropy Evaluation ──────────────────────────────────────────────────────

def compute_spatial_entropy(
    pts: np.ndarray,
    image_shape: Tuple[int, int],
    tile_size: int = 1600,
    step_size: int = 1200,
    grid_size: Optional[Tuple[int, int]] = None,
    **kwargs
) -> float:
    """
    Compute Shannon entropy H(S) of the point distribution across the grid.
    H(S) = - sum(p_i * log2(p_i))
    """
    if len(pts) == 0:
        return 0.0
        
    h, w = image_shape[:2]
    if grid_size is None and isinstance(tile_size, (tuple, list)):
        grid_size = tile_size

    if grid_size is not None:
        rows, cols = grid_size
        step_y = max(1.0, h / rows)
        step_x = max(1.0, w / cols)
    else:
        cols = math.ceil(w / step_size)
        rows = math.ceil(h / step_size)
        step_x = float(step_size)
        step_y = float(step_size)
        
    n_cells = rows * cols
    counts = np.zeros(n_cells, dtype=np.float64)
    
    for x, y in pts:
        c = min(int(x / step_x), cols - 1)
        r = min(int(y / step_y), rows - 1)
        c = max(0, c)
        r = max(0, r)
        idx = r * cols + c
        counts[idx] += 1
        
    p = counts / len(pts)
    p_nz = p[p > 0]
    
    entropy = -np.sum(p_nz * np.log2(p_nz))
    return float(entropy)


# ─── Master Tiled Matching ───────────────────────────────────────────────────

def _local_ncc_preposition(src_patch: np.ndarray, ref_patch_wide: np.ndarray, ds_factor: int = 4) -> Tuple[int, int]:
    """
    Returns (dx, dy) offset of src_patch within ref_patch_wide to maximize NCC.
    """
    if min(src_patch.shape) < ds_factor * 2 or min(ref_patch_wide.shape) < ds_factor * 2:
        return 0, 0
        
    s_ds = src_patch[::ds_factor, ::ds_factor].astype(np.float32)
    r_ds = ref_patch_wide[::ds_factor, ::ds_factor].astype(np.float32)
    
    # zero mean
    s_ds -= s_ds.mean()
    r_ds -= r_ds.mean()
    
    if s_ds.std() < 1e-3 or r_ds.std() < 1e-3:
        return 0, 0
        
    # Match template
    res = cv2.matchTemplate(r_ds, s_ds, cv2.TM_CCOEFF_NORMED)
    _, _, _, max_loc = cv2.minMaxLoc(res)
    
    best_x = max_loc[0] * ds_factor
    best_y = max_loc[1] * ds_factor
    
    return best_x, best_y


def draw_and_save_tile_matches(
    image0: np.ndarray,
    image1: np.ndarray,
    kpts0: np.ndarray,
    kpts1: np.ndarray,
    save_path: Union[str, Path],
    tile_id: int,
    margin: int = 20
):
    """
    Renders high-resolution side-by-side tile match visualization with line correspondences,
    matching the exact visual layout of lunar_matches_v5.
    """
    def norm_u8(arr):
        v = arr[arr > 0]
        if len(v) < 50:
            return np.clip(arr, 0, 255).astype(np.uint8)
        p2, p98 = np.percentile(v, (2, 98))
        return np.clip((arr - p2) / (p98 - p2 + 1e-6) * 255.0, 0, 255).astype(np.uint8)

    im0 = norm_u8(image0)
    im1 = norm_u8(image1)
    im0_bgr = cv2.cvtColor(im0, cv2.COLOR_GRAY2BGR)
    im1_bgr = cv2.cvtColor(im1, cv2.COLOR_GRAY2BGR)

    H0, W0 = im0.shape
    H1, W1 = im1.shape
    H, W = max(H0, H1), W0 + W1 + margin

    canvas = np.full((H, W, 3), 35, dtype=np.uint8)
    canvas[:H0, :W0] = im0_bgr
    canvas[:H1, W0 + margin:W0 + margin + W1] = im1_bgr

    for i, ((x0, y0), (x1, y1)) in enumerate(zip(kpts0, kpts1)):
        p0 = (int(round(x0)), int(round(y0)))
        p1 = (int(round(x1)) + W0 + margin, int(round(y1)))
        
        color = (0, 255, 128) # bright green
        cv2.line(canvas, p0, p1, color, 2, lineType=cv2.LINE_AA)
        cv2.circle(canvas, p0, 4, (0, 0, 255), -1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, p1, 4, (0, 0, 255), -1, lineType=cv2.LINE_AA)
        cv2.putText(canvas, str(i + 1), (p0[0] + 6, p0[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, str(i + 1), (p1[0] + 6, p1[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(canvas, f"OHRC Tile #{tile_id:04d}", (25, 45), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(canvas, f"OHRC Tile #{tile_id:04d}", (25, 45), cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, "NAC Reference Tile", (W0 + margin + 25, 45), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(canvas, "NAC Reference Tile", (W0 + margin + 25, 45), cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"LoFTR Matches: {len(kpts0)} points", (25, H - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2, cv2.LINE_AA)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), canvas)


def run_tiled_matching(
    src_input: Union[np.ndarray, str, Path],
    ref_input: Union[np.ndarray, str, Path],
    coarse_result: Optional[Dict[str, Any]] = None,
    output_csv: Optional[Union[str, Path]] = None,
    tile_size: int = 1600,
    step_size: int = 1200,
    method: str = "loftr",
    search_padding: int = 200,
    structural_method: str = "gradient",
    coarse_dx: Optional[float] = None,
    coarse_dy: Optional[float] = None,
    grid_size: Optional[Tuple[int, int]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Run tiled feature matching pre-positioned by the along-track drift model and wide-window NCC.
    Supports either pre-computed in-memory numpy arrays or file paths (windowed streaming reads).
    """
    viz_dir = output_csv.parent / "match_visualizations" if output_csv is not None else None
    tile_png_dir = output_csv.parent / "tile_pngs" if output_csv is not None else None
    if viz_dir:
        viz_dir.mkdir(parents=True, exist_ok=True)
    if tile_png_dir:
        tile_png_dir.mkdir(parents=True, exist_ok=True)

    is_src_file = isinstance(src_input, (str, Path))
    is_ref_file = isinstance(ref_input, (str, Path))

    src_ds = rasterio.open(src_input) if is_src_file else None
    ref_ds = rasterio.open(ref_input) if is_ref_file else None
    
    if coarse_result is None:
        coarse_result = {}
    elif not isinstance(coarse_result, dict):
        coarse_result = {}
        
    drift_model = coarse_result.get("drift_model", {})
    dy_slope = drift_model.get("dy_slope", 0.0)
    dy_intercept = drift_model.get("dy_intercept", coarse_result.get("dy", coarse_dy if coarse_dy is not None else 0.0))
    dx_slope = drift_model.get("dx_slope", 0.0)
    dx_intercept = drift_model.get("dx_intercept", coarse_result.get("dx", coarse_dx if coarse_dx is not None else 0.0))
    
    def predict_drift(r: float) -> Tuple[float, float]:
        return (dx_slope * r + dx_intercept, dy_slope * r + dy_intercept)

    try:
        if is_src_file:
            h_src, w_src = src_ds.height, src_ds.width
        else:
            h_src, w_src = src_input.shape[:2]

        if is_ref_file:
            h_ref, w_ref = ref_ds.height, ref_ds.width
        else:
            h_ref, w_ref = ref_input.shape[:2]

        # Adaptive tile sizing and stepping for long elongated swaths
        if max(h_src, h_ref) > 3500 and step_size > 800:
            eff_step = 800
            eff_tile = min(tile_size, 1200)
        else:
            eff_step = step_size
            eff_tile = tile_size

        tiles = get_tile_bounds((h_src, w_src), eff_tile, eff_step)
        
        # Adaptive search padding based on coarse alignment confidence:
        # If coarse confidence is low (< 0.15), expand window to 400px to absorb pointing uncertainty,
        # but keep it capped at <= 500px to maintain high crater contrast and fast GPU throughput.
        coarse_conf = float(coarse_result.get("confidence", 1.0)) if coarse_result else 1.0
        eff_padding = min(max(search_padding, 400), 500) if coarse_conf < 0.15 else search_padding
        
        all_src_pts = []
        all_ref_pts = []
        populated_cells = 0
        
        print(f"[TILED-MATCH] Running tiled matching ({len(tiles)} tiles) using {method.upper()} (padding={eff_padding}px)...")
        
        loftr_matcher = None
        if method.lower() in ["loftr", "ensemble"]:
            if LoFTRMatcher is not None:
                loftr_matcher = LoFTRMatcher()
            else:
                raise RuntimeError("LoFTRMatcher could not be loaded. PyTorch and LoFTR are required.")
                
        for t_idx, t in enumerate(tiles):
            # Source tile bounds
            x0, y0, x1, y1 = t["x0"], t["y0"], t["x1"], t["y1"]
            
            if is_src_file:
                win_src = Window(col_off=x0, row_off=y0, width=x1 - x0, height=y1 - y0)
                raw_src = src_ds.read(1, window=win_src).astype(np.float32)
            else:
                raw_src = src_input[y0:y1, x0:x1]
            
            # Predict reference center
            r_center = (y0 + y1) / 2.0
            pred_dx, pred_dy = predict_drift(r_center)
            
            ref_x0 = int(round(x0 + pred_dx))
            ref_y0 = int(round(y0 + pred_dy))
            ref_x1 = int(round(x1 + pred_dx))
            ref_y1 = int(round(y1 + pred_dy))
            
            # Add padding to reference search area for NCC pre-positioning
            search_x0 = max(0, ref_x0 - eff_padding)
            search_y0 = max(0, ref_y0 - eff_padding)
            search_x1 = min(w_ref, ref_x1 + eff_padding)
            search_y1 = min(h_ref, ref_y1 + eff_padding)
            
            if search_x1 <= search_x0 or search_y1 <= search_y0:
                continue
                
            # Pre-positioning: read reference window padded around predicted drift position
            if is_ref_file:
                win_ref = Window(col_off=search_x0, row_off=search_y0, width=search_x1 - search_x0, height=search_y1 - search_y0)
                raw_ref_wide = ref_ds.read(1, window=win_ref).astype(np.float32)
            else:
                raw_ref_wide = ref_input[search_y0:search_y1, search_x0:search_x1]

            # Check for empty / nodata patches
            if (raw_src > 0).mean() < 0.1 or (raw_ref_wide > 0).mean() < 0.1:
                continue

            # Extract and match using LoFTR on raw intensity / CLAHE patches
            t_src_pts, t_ref_pts = match_tile(
                raw_src, raw_ref_wide, 
                method=method, 
                loftr_matcher=loftr_matcher
            )
            
            if len(t_src_pts) > 0:
                # Map local tile coordinates back to global image coordinates
                g_src = t_src_pts.copy()
                g_src[:, 0] += x0
                g_src[:, 1] += y0
                
                g_ref = t_ref_pts.copy()
                g_ref[:, 0] += search_x0
                g_ref[:, 1] += search_y0
                
                # Filter out points in the overlap boundaries (Core-vs-Border Margin Filtering)
                core_x0, core_y0, core_x1, core_y1 = t["core_x0"], t["core_y0"], t["core_x1"], t["core_y1"]
                valid_core = (g_src[:, 0] >= core_x0) & (g_src[:, 0] < core_x1) & \
                             (g_src[:, 1] >= core_y0) & (g_src[:, 1] < core_y1)
                             
                g_src = g_src[valid_core]
                g_ref = g_ref[valid_core]
                
                if len(g_src) > 0:
                    all_src_pts.append(g_src)
                    all_ref_pts.append(g_ref)
                    if len(g_src) >= MIN_POINTS_PER_TILE:
                        populated_cells += 1

                    if viz_dir:
                        v_path = viz_dir / f"tile_{t_idx+1:04d}_matches.png"
                        draw_and_save_tile_matches(raw_src, raw_ref_wide, t_src_pts, t_ref_pts, v_path, t_idx + 1)
                    if tile_png_dir:
                        cv2.imwrite(str(tile_png_dir / f"tile_{t_idx+1:04d}_ohrc.png"), np.clip(raw_src, 0, 255).astype(np.uint8))
                        cv2.imwrite(str(tile_png_dir / f"tile_{t_idx+1:04d}_nac.png"), np.clip(raw_ref_wide, 0, 255).astype(np.uint8))
                        
        if not all_src_pts:
            print("  [TILED-MATCH] WARNING: No matches found across any tiles!")
            return {"total_matches": 0, "entropy": 0.0, "populated_cells": 0, "src_pts": np.empty((0, 2)), "ref_pts": np.empty((0, 2))}
            
        src_pts_arr = np.vstack(all_src_pts)
        ref_pts_arr = np.vstack(all_ref_pts)
        
        # Global RANSAC cleanup to ensure overall consistency
        H, mask = cv2.findHomography(src_pts_arr, ref_pts_arr, cv2.RANSAC, 5.0)
        if mask is not None:
            mask = mask.ravel() == 1
            src_pts_arr = src_pts_arr[mask]
            ref_pts_arr = ref_pts_arr[mask]
            
        total_matches = len(src_pts_arr)
        
        # Evaluate distribution
        entropy = compute_spatial_entropy(src_pts_arr, (h_src, w_src), tile_size, step_size)
        cols = math.ceil(w_src / step_size)
        rows = math.ceil(h_src / step_size)
        max_entropy = math.log2(cols * rows)
        
        print(f"  Total consistent matches: {total_matches}")
        print(f"  Populated cells (>{MIN_POINTS_PER_TILE} pts): {populated_cells} / {cols*rows}")
        print(f"  Spatial Entropy: {entropy:.2f} / {max_entropy:.2f}")
        
        # Write to CSV
        with open(output_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["src_x", "src_y", "ref_x", "ref_y"])
            for (sx, sy), (rx, ry) in zip(src_pts_arr, ref_pts_arr):
                writer.writerow([f"{sx:.2f}", f"{sy:.2f}", f"{rx:.2f}", f"{ry:.2f}"])
                
        print(f"  Matches saved to {output_csv}")
        
        return {
            "total_matches": total_matches,
            "entropy": entropy,
            "spatial_entropy": entropy,
            "max_entropy": max_entropy,
            "populated_cells": populated_cells,
            "src_pts": src_pts_arr,
            "ref_pts": ref_pts_arr
        }
    finally:
        if src_ds is not None:
            src_ds.close()
        if ref_ds is not None:
            ref_ds.close()
