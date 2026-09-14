from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import argparse
import json


def parse_args():
    parser = argparse.ArgumentParser(description="Along-Track Drift Profiler (Stage 2 - Diagnostic)")
    parser.add_argument("--output_dir", type=Path, required=True, help="Run output directory containing the CSV")
    parser.add_argument("--config", type=Path, default=None, help="Optional config JSON")
    return parser.parse_args()


def run_drift_profile(output_dir: Path, custom_config: dict = None):
    output_dir = Path(output_dir)
    core_csv = output_dir / "ohrc_nac_superglue_matches_core_only.csv"
    diag_dir = output_dir / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    min_confidence = 0.30
    segment_size_px = 2500
    ransac_threshold_px = 15
    ransac_confidence = 0.999
    ransac_max_iters = 5000
    min_points_per_segment = 15

    if custom_config and "drift_profiler" in custom_config:
        cfg = custom_config["drift_profiler"]
        min_confidence = cfg.get("min_confidence", min_confidence)
        segment_size_px = cfg.get("segment_size_px", segment_size_px)
        ransac_threshold_px = cfg.get("ransac_threshold_px", ransac_threshold_px)

    def fit_affine(src, dst):
        M, mask = cv2.estimateAffine2D(
            src, dst, method=cv2.RANSAC,
            ransacReprojThreshold=ransac_threshold_px,
            confidence=ransac_confidence, maxIters=ransac_max_iters,
        )
        if M is None:
            return None
        mask = mask.ravel().astype(bool)
        src_in, dst_in = src[mask], dst[mask]
        src_h = np.hstack([src_in, np.ones((len(src_in), 1))])
        residuals = (M @ src_h.T).T - dst_in
        rmse = float(np.sqrt(np.mean(np.sum(residuals ** 2, axis=1))))

        a, b, tx = M[0]
        c, d, ty = M[1]
        return {
            "n_pts": len(src), "n_inliers": int(mask.sum()),
            "inlier_ratio": float(mask.sum() / len(src)), "rmse_px": rmse,
            "tx": tx, "ty": ty,
            "scale_x": float(np.hypot(a, c)), "scale_y": float(np.hypot(b, d)),
            "rotation_deg": float(np.degrees(np.arctan2(c, a))),
        }

    df = pd.read_csv(core_csv)
    df = df[df["confidence"] >= min_confidence].reset_index(drop=True)
    row_min, row_max = df["ohrc_y"].min(), df["ohrc_y"].max()
    segment_starts = np.arange(row_min, row_max, segment_size_px)

    records = []
    for seg_start in segment_starts:
        seg_end = seg_start + segment_size_px
        subset = df[(df["ohrc_y"] >= seg_start) & (df["ohrc_y"] < seg_end)]
        if len(subset) < min_points_per_segment:
            continue
        src = subset[["ohrc_x", "ohrc_y"]].to_numpy(dtype=np.float64)
        dst = subset[["nac_x", "nac_y"]].to_numpy(dtype=np.float64)
        r = fit_affine(src, dst)
        if r is not None:
            r["row_center"] = (seg_start + seg_end) / 2
            r["row_start"] = seg_start
            r["row_end"] = seg_end
            records.append(r)

    if len(records) < 2:
        print("[WARNING] Not enough valid segments for drift profile plot.")
        return None

    seg_df = pd.DataFrame(records)
    out_csv = output_dir / "along_track_drift_profile.csv"
    seg_df.to_csv(out_csv, index=False)

    fig, axes = plt.subplots(4, 1, figsize=(8, 12), sharex=True)
    axes[0].plot(seg_df["row_center"], seg_df["tx"], "o-", color="tab:blue")
    axes[0].set_ylabel("tx (px)")
    axes[0].set_title("Translation X vs OHRC row")
    axes[0].grid(alpha=0.3)

    axes[1].plot(seg_df["row_center"], seg_df["ty"], "o-", color="tab:orange")
    axes[1].set_ylabel("ty (px)")
    axes[1].set_title("Translation Y vs OHRC row")
    axes[1].grid(alpha=0.3)

    axes[2].plot(seg_df["row_center"], seg_df["rotation_deg"], "o-", color="tab:green")
    axes[2].set_ylabel("rotation (deg)")
    axes[2].set_title("Rotation vs OHRC row")
    axes[2].grid(alpha=0.3)

    axes[3].plot(seg_df["row_center"], seg_df["scale_x"], "o-", label="scale_x", color="tab:red")
    axes[3].plot(seg_df["row_center"], seg_df["scale_y"], "o-", label="scale_y", color="tab:purple")
    axes[3].axhline(1.0, color="gray", lw=0.8, ls="--")
    axes[3].set_ylabel("stretch factor")
    axes[3].set_xlabel("OHRC row (px)")
    axes[3].legend()
    axes[3].grid(alpha=0.3)

    fig.tight_layout()
    plot_path = diag_dir / "along_track_drift_profile.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"[SUCCESS] Drift profile saved: {out_csv} and {plot_path}")
    return out_csv


def main():
    args = parse_args()
    custom_cfg = None
    if args.config and args.config.exists():
        with open(args.config, "r", encoding="utf-8") as f:
            custom_cfg = json.load(f)
    run_drift_profile(args.output_dir, custom_cfg)


if __name__ == "__main__":
    main()
