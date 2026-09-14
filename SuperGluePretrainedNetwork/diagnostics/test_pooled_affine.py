

from pathlib import Path

import numpy as np
import pandas as pd


MATCH_DIR = Path(r"C:\#Padhai\E\SIH1\SuperGluePretrainedNetwork\lunar_matches_v5")
CORE_CSV = MATCH_DIR / "ohrc_nac_superglue_matches_core_only.csv"

MIN_CONFIDENCE = 0.30
PIXEL_SIZE_M = 1.1523692688652

CLUSTERS = {
    "top_cluster": (6800, 9000),
    "middle_cluster": (9200, 14400),
    "lower_cluster": (19400, 24000),
}


def fit_ols_affine(src, dst):
    """
    Ordinary least squares affine fit: [x' y'] = [x y 1] @ P
    where P is 3x2. Solved directly via lstsq, no RANSAC, no threshold,
    every point contributes (robust outliers are handled by first
    restricting to the union of each cluster's own RANSAC inliers,
    computed separately below, before calling this).
    """
    A = np.hstack([src, np.ones((len(src), 1))])  # N x 3
    P, _, _, _ = np.linalg.lstsq(A, dst, rcond=None)  # 3 x 2
    M = P.T  # 2 x 3, same convention as cv2.estimateAffine2D output
    return M


def apply_affine(M, pts):
    A = np.hstack([pts, np.ones((len(pts), 1))])
    return (M @ A.T).T


def rmse_xy(predicted, actual):
    res = predicted - actual
    rmse_x = float(np.sqrt(np.mean(res[:, 0] ** 2)))
    rmse_y = float(np.sqrt(np.mean(res[:, 1] ** 2)))
    rmse_c = float(np.sqrt(np.mean(np.sum(res ** 2, axis=1))))
    return rmse_x, rmse_y, rmse_c


df = pd.read_csv(CORE_CSV)
df = df[df["confidence"] >= MIN_CONFIDENCE].reset_index(drop=True)

# Use each cluster's own previously-established RANSAC inliers as the
# clean input to the pooled fit (removes each cluster's local noise
# without letting RANSAC's global sampling imbalance throw away entire
# clusters). This directly tests: "if we trust each cluster's own
# genuine matches, does ONE affine explain all of them together?"
import cv2

pooled_src, pooled_dst = [], []
cluster_points = {}

for name, (row_min, row_max) in CLUSTERS.items():
    subset = df[(df["ohrc_y"] >= row_min) & (df["ohrc_y"] < row_max)]
    src = subset[["ohrc_x", "ohrc_y"]].to_numpy(dtype=np.float64)
    dst = subset[["nac_x", "nac_y"]].to_numpy(dtype=np.float64)

    M, mask = cv2.estimateAffine2D(
        src, dst, method=cv2.RANSAC, ransacReprojThreshold=15,
        confidence=0.999, maxIters=5000,
    )
    mask = mask.ravel().astype(bool)
    cluster_points[name] = (src[mask], dst[mask])
    pooled_src.append(src[mask])
    pooled_dst.append(dst[mask])
    print(f"{name}: using {mask.sum()} local RANSAC inliers as clean input")

pooled_src = np.vstack(pooled_src)
pooled_dst = np.vstack(pooled_dst)
print(f"\nPooled clean points from all 3 clusters: {len(pooled_src)}")

# Fit ONE ordinary-least-squares affine on the pooled, pre-cleaned points.
M_pooled = fit_ols_affine(pooled_src, pooled_dst)
a, b, tx = M_pooled[0]
c, d, ty = M_pooled[1]
scale_x, scale_y = float(np.hypot(a, c)), float(np.hypot(b, d))
rotation_deg = float(np.degrees(np.arctan2(c, a)))

print("\n--- POOLED ORDINARY LEAST-SQUARES AFFINE (all 3 clusters combined) ---")
print(f"Matrix:\n{M_pooled}")
print(f"Translation: tx={tx:.2f}, ty={ty:.2f}")
print(f"Scale: sx={scale_x:.4f}, sy={scale_y:.4f}")
print(f"Rotation: {rotation_deg:.3f} deg")

# Evaluate this single pooled model separately on each cluster's own
# clean points -- does it explain all three well, or does it still fail
# on the top cluster?
print("\n--- HOW WELL DOES THE SINGLE POOLED MODEL EXPLAIN EACH CLUSTER? ---")
print(f"{'cluster':>15s} {'n_pts':>6s} {'rmse_x':>8s} {'rmse_y':>8s} {'rmse_comb':>10s}")
for name, (src, dst) in cluster_points.items():
    predicted = apply_affine(M_pooled, src)
    rx, ry, rc = rmse_xy(predicted, dst)
    print(f"{name:>15s} {len(src):>6d} {rx:>8.2f} {ry:>8.2f} {rc:>10.2f}")

# Also report overall RMSE across all pooled points.
predicted_all = apply_affine(M_pooled, pooled_src)
rx, ry, rc = rmse_xy(predicted_all, pooled_dst)
print(f"{'ALL POOLED':>15s} {len(pooled_src):>6d} {rx:>8.2f} {ry:>8.2f} {rc:>10.2f}")

print("\nInterpretation:")
print("- If RMSE stays low (roughly comparable to each cluster's own local")
print("  RMSE, e.g. under ~10-15px) for ALL THREE clusters with this ONE")
print("  pooled model -> the drift IS linear/affine, RANSAC's global fit")
print("  just needed properly balanced input, not a different model.")
print("  Use this pooled OLS affine (or re-run RANSAC with balanced/")
print("  stratified sampling across clusters) as your registration model.")
print("- If RMSE stays low for middle/lower but blows up for top_cluster")
print("  even with this clean, unbiased fit -> the relationship genuinely")
print("  is NOT linear across the full strip, and a piecewise or low-order")
print("  polynomial model is needed instead.")