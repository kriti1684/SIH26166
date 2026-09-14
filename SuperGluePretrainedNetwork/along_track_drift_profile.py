

from pathlib import Path

import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, required=True, help="Run output directory containing the CSV")
    return parser.parse_args()

args = parse_args()


MATCH_DIR = args.output_dir
CORE_CSV = MATCH_DIR / "ohrc_nac_superglue_matches_core_only.csv"
DIAG_DIR = MATCH_DIR / "diagnostics"
DIAG_DIR.mkdir(parents=True, exist_ok=True)

MIN_CONFIDENCE = 0.30
SEGMENT_SIZE_PX = 2500
RANSAC_THRESHOLD_PX = 15
RANSAC_CONFIDENCE = 0.999
RANSAC_MAX_ITERS = 5000
MIN_POINTS_PER_SEGMENT = 15


def fit_affine(src, dst, threshold_px):
    M, mask = cv2.estimateAffine2D(
        src, dst, method=cv2.RANSAC,
        ransacReprojThreshold=threshold_px,
        confidence=RANSAC_CONFIDENCE, maxIters=RANSAC_MAX_ITERS,
    )
    if M is None:
        return None
    mask = mask.ravel().astype(bool)
    src_in, dst_in = src[mask], dst[mask]
    src_h = np.hstack([src_in, np.ones((len(src_in), 1))])
    predicted = (M @ src_h.T).T
    residuals = predicted - dst_in
    rmse = float(np.sqrt(np.mean(np.sum(residuals ** 2, axis=1))))

    a, b, tx = M[0]
    c, d, ty = M[1]
    scale_x, scale_y = float(np.hypot(a, c)), float(np.hypot(b, d))
    rotation_deg = float(np.degrees(np.arctan2(c, a)))

    return {
        "n_pts": len(src), "n_inliers": int(mask.sum()),
        "inlier_ratio": mask.sum() / len(src), "rmse_px": rmse,
        "tx": tx, "ty": ty, "scale_x": scale_x, "scale_y": scale_y,
        "rotation_deg": rotation_deg,
    }


df = pd.read_csv(CORE_CSV)
df = df[df["confidence"] >= MIN_CONFIDENCE].reset_index(drop=True)

row_min, row_max = df["ohrc_y"].min(), df["ohrc_y"].max()
print(f"OHRC row range in matches: {row_min:.0f} to {row_max:.0f}")

segment_starts = np.arange(row_min, row_max, SEGMENT_SIZE_PX)

records = []
print(f"\n{'seg_row_range':>22s} {'center':>8s} {'n_pts':>6s} {'inliers':>8s} "
      f"{'ratio':>7s} {'rmse':>7s} {'tx':>9s} {'ty':>10s} {'scale_x':>8s} {'rot_deg':>8s}")

for seg_start in segment_starts:
    seg_end = seg_start + SEGMENT_SIZE_PX
    subset = df[(df["ohrc_y"] >= seg_start) & (df["ohrc_y"] < seg_end)]

    if len(subset) < MIN_POINTS_PER_SEGMENT:
        print(f"[{seg_start:>9.0f},{seg_end:>9.0f}] {'':>8s} {len(subset):>6d}  "
              f"(too few points, skipped)")
        continue

    src = subset[["ohrc_x", "ohrc_y"]].to_numpy(dtype=np.float64)
    dst = subset[["nac_x", "nac_y"]].to_numpy(dtype=np.float64)
    r = fit_affine(src, dst, RANSAC_THRESHOLD_PX)

    if r is None:
        print(f"[{seg_start:>9.0f},{seg_end:>9.0f}]  RANSAC failed")
        continue

    center = (seg_start + seg_end) / 2
    r["row_center"] = center
    r["row_start"] = seg_start
    r["row_end"] = seg_end
    records.append(r)

    print(f"[{seg_start:>9.0f},{seg_end:>9.0f}] {center:>8.0f} {r['n_pts']:>6d} "
          f"{r['n_inliers']:>8d} {r['inlier_ratio']:>7.3f} {r['rmse_px']:>7.2f} "
          f"{r['tx']:>9.2f} {r['ty']:>10.2f} {r['scale_x']:>8.4f} {r['rotation_deg']:>8.3f}")

if len(records) < 2:
    raise RuntimeError("Not enough valid segments to plot a drift profile.")

seg_df = pd.DataFrame(records)
seg_df.to_csv(MATCH_DIR / "along_track_drift_profile.csv", index=False)
print(f"\nSaved: {MATCH_DIR / 'along_track_drift_profile.csv'}")

# ============================================================
# PLOT: tx, ty, rotation, scale vs row position
# ============================================================

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
axes[3].set_ylabel("apparent stretch factor")
axes[3].set_xlabel("OHRC row (px)")
axes[3].set_title("Apparent affine 'scale' vs OHRC row\n"
                   "(NOTE: both images are already on the same resampled grid -- "
                   "this is NOT a real GSD/camera scale difference. It reflects "
                   "residual distortion from OHRC's georeferencing/reprojection.)")
axes[3].legend()
axes[3].grid(alpha=0.3)

fig.tight_layout()
plot_path = DIAG_DIR / "along_track_drift_profile.png"
fig.savefig(plot_path, dpi=150)
plt.close(fig)
print(f"Saved: {plot_path}")

# Also plot inlier ratio / point count per segment, as a data-density
# sanity check (thin segments = less reliable estimate there).
fig, ax1 = plt.subplots(figsize=(8, 4))
ax1.bar(seg_df["row_center"], seg_df["n_pts"], width=SEGMENT_SIZE_PX * 0.8,
        color="lightgray", label="n_pts (matches in segment)")
ax1.set_xlabel("OHRC row (px)")
ax1.set_ylabel("n_pts")
ax2 = ax1.twinx()
ax2.plot(seg_df["row_center"], seg_df["inlier_ratio"], "o-", color="tab:blue",
         label="inlier ratio")
ax2.set_ylabel("inlier ratio")
ax1.set_title("Match density and local fit quality per segment")
fig.tight_layout()
density_path = DIAG_DIR / "along_track_segment_density.png"
fig.savefig(density_path, dpi=150)
plt.close(fig)
print(f"Saved: {density_path}")