import sys
import argparse
from pathlib import Path
import json
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import cv2
import rasterio
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch

try:
    from src.models.matching import Matching
except ImportError:
    from ..models.matching import Matching


def parse_args():
    parser = argparse.ArgumentParser(description="Coarse Alignment (Stage 0)")
    parser.add_argument("--source_img", type=Path, required=True, help="Path to source image (e.g. OHRC)")
    parser.add_argument("--ref_img", type=Path, required=True, help="Path to reference image (e.g. NAC)")
    parser.add_argument("--output_dir", type=Path, required=True, help="Run output directory")
    parser.add_argument("--config", type=Path, default=None, help="Path to custom JSON config")
    return parser.parse_args()


def run_coarse_alignment(source_path: Path, ref_path: Path, output_dir: Path, custom_config: dict = None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_json = output_dir / "coarse_alignment_result.json"

    # Default parameters
    target_long_side = 1600
    max_pyramid_levels = 5
    min_matches_to_accept = 5
    max_keypoints = 4096
    keypoint_threshold = 0.005
    nms_radius = 3
    sinkhorn_iterations = 20
    match_threshold = 0.2
    min_match_confidence = 0.20
    ransac_threshold_decimated_px = 6
    ransac_confidence = 0.999
    ransac_max_iters = 5000

    if custom_config and "coarse_alignment" in custom_config:
        cfg = custom_config["coarse_alignment"]
        target_long_side = cfg.get("target_long_side_px", target_long_side)
        max_pyramid_levels = cfg.get("max_pyramid_levels", max_pyramid_levels)
        min_matches_to_accept = cfg.get("min_matches_to_accept", min_matches_to_accept)
        max_keypoints = cfg.get("max_keypoints", max_keypoints)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[STAGE 0] Device: {device}")

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
    print("SuperPoint + SuperGlue loaded.\n")

    def read_decimated(path, decimation):
        with rasterio.open(path) as src:
            h, w = src.height, src.width
            out_h = max(1, round(h / decimation))
            out_w = max(1, round(w / decimation))
            data = src.read(
                1, out_shape=(1, out_h, out_w),
                resampling=rasterio.enums.Resampling.average,
            )
            scale_row = h / out_h
            scale_col = w / out_w
            return data, scale_row, scale_col, h, w

    def to_tensor(image):
        t = torch.from_numpy(image.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
        return t.to(device)

    def run_pass(source_img, ref_img):
        image0 = to_tensor(source_img)
        image1 = to_tensor(ref_img)
        with torch.no_grad():
            pred = matching({"image0": image0, "image1": image1})

        kp0 = pred["keypoints0"][0].detach().cpu().numpy()
        kp1 = pred["keypoints1"][0].detach().cpu().numpy()
        matches0 = pred["matches0"][0].detach().cpu().numpy()
        conf0 = pred["matching_scores0"][0].detach().cpu().numpy()

        pts0, pts1, confs = [], [], []
        for idx0 in np.where(matches0 > -1)[0]:
            idx1 = int(matches0[idx0])
            if idx1 < 0:
                continue
            confidence = float(conf0[idx0])
            if confidence < min_match_confidence:
                continue
            x0, y0 = kp0[idx0]
            x1, y1 = kp1[idx1]
            if source_img[int(round(y0)), int(round(x0))] == 0 or ref_img[int(round(y1)), int(round(x1))] == 0:
                continue
            pts0.append([x0, y0])
            pts1.append([x1, y1])
            confs.append(confidence)

        return (
            np.array(pts0, dtype=np.float64),
            np.array(pts1, dtype=np.float64),
            np.array(confs, dtype=np.float64),
            len(kp0), len(kp1),
        )

    def find_valid_bbox(img):
        valid = img > 0
        rows_any = np.any(valid, axis=1)
        cols_any = np.any(valid, axis=0)
        if not np.any(rows_any):
            return None
        r0 = int(np.argmax(rows_any))
        r1 = int(len(rows_any) - 1 - np.argmax(rows_any[::-1]))
        c0 = int(np.argmax(cols_any))
        c1 = int(len(cols_any) - 1 - np.argmax(cols_any[::-1]))
        return r0, r1, c0, c1

    with rasterio.open(source_path) as s, rasterio.open(ref_path) as r:
        if s.width != r.width or s.height != r.height:
            raise RuntimeError("Source and reference are not on the same grid.")
        full_height, full_width = s.height, s.width

    longest_side = max(full_height, full_width)
    base_decimation = max(1, round(longest_side / target_long_side))
    result = None
    decimation = base_decimation

    for level in range(max_pyramid_levels):
        print(f"\n--- Pyramid level {level+1}/{max_pyramid_levels}: decimation={decimation} ---")
        source_raw, scale_row, scale_col, _, _ = read_decimated(source_path, decimation)
        ref_raw, _, _, _, _ = read_decimated(ref_path, decimation)

        src_bbox = find_valid_bbox(source_raw)
        if src_bbox is None:
            decimation = max(1, decimation // 2)
            continue

        src_r0, src_r1, src_c0, src_c1 = src_bbox
        source_img = source_raw[src_r0:src_r1+1, src_c0:src_c1+1]

        if result is not None:
            pred_shift_r = result["offset_dy"] / scale_row
            pred_shift_c = result["offset_dx"] / scale_col
            margin_r = max(int((src_r1 - src_r0) * 0.5), 50)
            margin_c = max(int((src_c1 - src_c0) * 0.5), 50)
            ref_r0 = max(0, int(src_r0 + pred_shift_r) - margin_r)
            ref_r1 = min(ref_raw.shape[0] - 1, int(src_r1 + pred_shift_r) + margin_r)
            ref_c0 = max(0, int(src_c0 + pred_shift_c) - margin_c)
            ref_c1 = min(ref_raw.shape[1] - 1, int(src_c1 + pred_shift_c) + margin_c)
        else:
            ref_bbox = find_valid_bbox(ref_raw)
            if ref_bbox is None:
                decimation = max(1, decimation // 2)
                continue
            ref_r0, ref_r1, ref_c0, ref_c1 = ref_bbox

        ref_img = ref_raw[ref_r0:ref_r1+1, ref_c0:ref_c1+1]

        try:
            pts0, pts1, confs, n_kp0, n_kp1 = run_pass(source_img, ref_img)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" in str(e).lower() or "CUDA" in str(e):
                if device == "cuda":
                    torch.cuda.empty_cache()
                break
            raise
        finally:
            if device == "cuda":
                torch.cuda.empty_cache()

        if len(pts0) < min_matches_to_accept:
            decimation = max(1, decimation // 2)
            continue

        pts0_full = (pts0 + np.array([src_c0, src_r0])) * np.array([scale_col, scale_row])
        pts1_full = (pts1 + np.array([ref_c0, ref_r0])) * np.array([scale_col, scale_row])
        ransac_threshold_full = ransac_threshold_decimated_px * max(scale_row, scale_col)

        M, mask = cv2.estimateAffinePartial2D(
            pts0_full, pts1_full, method=cv2.RANSAC,
            ransacReprojThreshold=ransac_threshold_full,
            confidence=ransac_confidence, maxIters=ransac_max_iters,
        )

        if M is None:
            decimation = max(1, decimation // 2)
            continue

        mask = mask.ravel().astype(bool)
        n_inliers = int(mask.sum())
        if n_inliers < min_matches_to_accept:
            decimation = max(1, decimation // 2)
            continue

        src_in = pts0_full[mask]
        dst_in = pts1_full[mask]
        src_h = np.hstack([src_in, np.ones((len(src_in), 1))])
        predicted = (M @ src_h.T).T
        residuals = predicted - dst_in
        rmse_full = float(np.sqrt(np.mean(np.sum(residuals ** 2, axis=1))))

        a, b, tx = M[0]
        c, d, ty = M[1]
        scale = float(np.hypot(a, c))
        rotation_deg = float(np.degrees(np.arctan2(c, a)))

        x_center = (src_c0 + src_c1) / 2.0 * scale_col
        y_center = (src_r0 + src_r1) / 2.0 * scale_row
        ref_center = M @ np.array([x_center, y_center, 1.0])
        eff_dx = float(ref_center[0] - x_center)
        eff_dy = float(ref_center[1] - y_center)

        result = {
            "offset_dx": eff_dx, "offset_dy": eff_dy,
            "similarity_tx": float(tx), "similarity_ty": float(ty),
            "rotation_deg": rotation_deg, "scale": scale,
            "decimation_used": decimation,
            "n_matches": len(pts0_full), "n_inliers": n_inliers,
            "inlier_ratio": float(n_inliers / len(pts0_full)),
            "rmse_coarse_full_res_px": rmse_full,
            "pyramid_level": level + 1,
            "full_height": full_height, "full_width": full_width,
            "_source_img": source_img.copy(),
            "_ref_img": ref_img.copy(),
            "_src_in": src_in, "_dst_in": dst_in,
            "_src_c0": src_c0, "_src_r0": src_r0,
            "_ref_c0": ref_c0, "_ref_r0": ref_r0,
            "_scale_col": scale_col, "_scale_row": scale_row,
        }

        decimation = max(1, decimation // 2)
        if decimation < 8:
            break

    if result is None:
        raise RuntimeError("Coarse alignment failed at all pyramid levels.")

    # Save visual plot
    source_img = result.pop("_source_img")
    ref_img = result.pop("_ref_img")
    src_in = result.pop("_src_in")
    dst_in = result.pop("_dst_in")
    src_c0, src_r0 = result.pop("_src_c0"), result.pop("_src_r0")
    ref_c0, ref_r0 = result.pop("_ref_c0"), result.pop("_ref_r0")
    scale_col, scale_row = result.pop("_scale_col"), result.pop("_scale_row")

    with open(result_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\n[SUCCESS] Coarse alignment saved: {result_json}")

    # Generate visual plot
    try:
        diag_dir = output_dir / "diagnostics"
        diag_dir.mkdir(parents=True, exist_ok=True)
        src_local = (src_in / np.array([scale_col, scale_row])) - np.array([src_c0, src_r0])
        dst_local = (dst_in / np.array([scale_col, scale_row])) - np.array([ref_c0, ref_r0])

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        ax1.imshow(source_img, cmap='gray')
        ax1.set_title("Source (Coarse)")
        ax1.axis('off')
        ax2.imshow(ref_img, cmap='gray')
        ax2.set_title("Reference (Coarse)")
        ax2.axis('off')

        indices = np.linspace(0, len(src_local)-1, min(150, len(src_local)), dtype=int)
        for i in indices:
            xy1 = (src_local[i, 0], src_local[i, 1])
            xy2 = (dst_local[i, 0], dst_local[i, 1])
            con = ConnectionPatch(xyA=xy2, xyB=xy1, coordsA="data", coordsB="data",
                                  axesA=ax2, axesB=ax1, color="lime", lw=0.5, alpha=0.6)
            ax2.add_artist(con)

        plot_path = diag_dir / "coarse_alignment_visual.png"
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, facecolor='black', edgecolor='none')
        plt.close(fig)
    except Exception as e:
        print(f"Warning: could not save coarse plot: {e}")

    return result


def main():
    args = parse_args()
    custom_cfg = None
    if args.config and args.config.exists():
        with open(args.config, "r", encoding="utf-8") as f:
            custom_cfg = json.load(f)
    run_coarse_alignment(args.source_img, args.ref_img, args.output_dir, custom_cfg)


if __name__ == "__main__":
    main()
