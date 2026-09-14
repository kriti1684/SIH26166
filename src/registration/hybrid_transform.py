from pathlib import Path
import json
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Hybrid Transform Fitting (Stage 2 - Modeling)")
    parser.add_argument("--output_dir", type=Path, required=True, help="Run output directory containing the CSV")
    parser.add_argument("--config", type=Path, default=None, help="Optional config JSON")
    return parser.parse_args()


def fit_hybrid_transform(output_dir: Path, custom_config: dict = None):
    output_dir = Path(output_dir)
    core_csv = output_dir / "ohrc_nac_superglue_matches_core_only.csv"
    diag_dir = output_dir / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    min_confidence = 0.30
    segment_size_px = 2500
    ransac_threshold_px = 15
    min_points_for_robust_segment = 300
    poly_degree = 2
    sigma_clip_rounds = 3
    sigma_clip_threshold_px = 40
    consistency_threshold_px = 15
    pixel_size_m = 1.1523692688652

    if custom_config and "hybrid_transform" in custom_config:
        cfg = custom_config["hybrid_transform"]
        min_confidence = cfg.get("min_confidence", min_confidence)
        segment_size_px = cfg.get("segment_size_px", segment_size_px)
        ransac_threshold_px = cfg.get("ransac_threshold_px", ransac_threshold_px)
        poly_degree = cfg.get("poly_degree", poly_degree)
        sigma_clip_rounds = cfg.get("sigma_clip_rounds", sigma_clip_rounds)

    df = pd.read_csv(core_csv)
    df = df[df["confidence"] >= min_confidence].reset_index(drop=True)
    row_min, row_max = df["ohrc_y"].min(), df["ohrc_y"].max()
    segment_starts = np.arange(row_min, row_max, segment_size_px)

    seg_rotations, seg_scales = [], []
    for seg_start in segment_starts:
        seg_end = seg_start + segment_size_px
        subset = df[(df["ohrc_y"] >= seg_start) & (df["ohrc_y"] < seg_end)]
        if len(subset) < min_points_for_robust_segment:
            continue

        src = subset[["ohrc_x", "ohrc_y"]].to_numpy(dtype=np.float64)
        dst = subset[["nac_x", "nac_y"]].to_numpy(dtype=np.float64)
        M, mask = cv2.estimateAffine2D(
            src, dst, method=cv2.RANSAC, ransacReprojThreshold=ransac_threshold_px,
            confidence=0.999, maxIters=5000,
        )
        if M is None:
            continue
        a, b, tx = M[0]
        c, d, ty = M[1]
        scale_val = (np.hypot(a, c) + np.hypot(b, d)) / 2.0
        rot_deg = float(np.degrees(np.arctan2(c, a)))
        seg_rotations.append(rot_deg)
        seg_scales.append(scale_val)

    theta_deg = float(np.median(seg_rotations)) if seg_rotations else 0.0
    scale = float(np.median(seg_scales)) if seg_scales else 1.0
    theta_rad = np.radians(theta_deg)

    print(f"[HYBRID MODEL] Robust Rotation: {theta_deg:.3f} deg, Scale: {scale:.4f}")

    R = scale * np.array([
        [np.cos(theta_rad), -np.sin(theta_rad)],
        [np.sin(theta_rad), np.cos(theta_rad)],
    ])

    src_all = df[["ohrc_x", "ohrc_y"]].to_numpy(dtype=np.float64)
    dst_all = df[["nac_x", "nac_y"]].to_numpy(dtype=np.float64)
    rows_all = df["ohrc_y"].to_numpy(dtype=np.float64)

    rotated = (R @ src_all.T).T
    residual_tx = dst_all[:, 0] - rotated[:, 0]
    residual_ty = dst_all[:, 1] - rotated[:, 1]

    keep = np.ones(len(df), dtype=bool)
    coeffs_x = np.zeros(poly_degree + 1)
    coeffs_y = np.zeros(poly_degree + 1)

    for round_i in range(sigma_clip_rounds):
        coeffs_x = np.polyfit(rows_all[keep], residual_tx[keep], poly_degree)
        coeffs_y = np.polyfit(rows_all[keep], residual_ty[keep], poly_degree)
        pred_tx = np.polyval(coeffs_x, rows_all)
        pred_ty = np.polyval(coeffs_y, rows_all)
        resid = np.hypot(residual_tx - pred_tx, residual_ty - pred_ty)
        keep = resid <= sigma_clip_threshold_px

    def predict(pts):
        rot = (R @ pts.T).T
        rows = pts[:, 1]
        tx = np.polyval(coeffs_x, rows)
        ty = np.polyval(coeffs_y, rows)
        return rot + np.stack([tx, ty], axis=1)

    predicted_all = predict(src_all)
    residuals = predicted_all - dst_all
    rmse_x = float(np.sqrt(np.mean(residuals[:, 0] ** 2)))
    rmse_y = float(np.sqrt(np.mean(residuals[:, 1] ** 2)))
    rmse_combined = float(np.sqrt(np.mean(np.sum(residuals ** 2, axis=1))))
    dist = np.hypot(residuals[:, 0], residuals[:, 1])
    consistent = dist <= consistency_threshold_px
    consistency_ratio = float(consistent.mean())

    model_summary = {
        "rotation_deg": theta_deg, "scale": scale,
        "translation_poly_degree": poly_degree,
        "tx_poly_coeffs": coeffs_x.tolist(), "ty_poly_coeffs": coeffs_y.tolist(),
        "min_confidence_filter": min_confidence,
        "consistency_threshold_px": consistency_threshold_px,
        "overall_consistency_ratio": consistency_ratio,
        "overall_rmse_x_px": rmse_x, "overall_rmse_y_px": rmse_y,
        "overall_rmse_combined_px": rmse_combined,
        "overall_rmse_combined_m": rmse_combined * pixel_size_m,
    }

    model_json = output_dir / "hybrid_transform_model.json"
    with open(model_json, "w", encoding="utf-8") as f:
        json.dump(model_summary, f, indent=2)

    consistent_df = df[consistent].copy()
    consistent_csv = output_dir / "hybrid_model_consistent_matches.csv"
    consistent_df.to_csv(consistent_csv, index=False)

    print(f"[SUCCESS] Hybrid model fitted: {model_json}")
    print(f"Overall RMSE: {rmse_combined:.2f} px ({rmse_combined*pixel_size_m:.2f} m), Inliers: {consistency_ratio*100:.1f}%")
    return model_json


def main():
    args = parse_args()
    custom_cfg = None
    if args.config and args.config.exists():
        with open(args.config, "r", encoding="utf-8") as f:
            custom_cfg = json.load(f)
    fit_hybrid_transform(args.output_dir, custom_cfg)


if __name__ == "__main__":
    main()
