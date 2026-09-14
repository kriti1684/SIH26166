

from pathlib import Path

import numpy as np
import pandas as pd
import cv2


MATCH_DIR = Path(r"C:\#Padhai\E\SIH1\SuperGluePretrainedNetwork\lunar_matches_v5")
CORE_CSV = MATCH_DIR / "ohrc_nac_superglue_matches_core_only.csv"

MIN_CONFIDENCE = 0.30
RANSAC_THRESHOLD_PX = 15
RANSAC_CONFIDENCE = 0.999
RANSAC_MAX_ITERS = 5000

# Row ranges to test as separate clusters, based on the spatial plot.
# Adjust these if your plot's cluster boundaries look different.
CLUSTERS = {
    "top_cluster (rejected by global fit)": (6800, 9000),
    "middle_cluster (mostly accepted)": (9200, 14400),
    "lower_cluster (mostly accepted)": (19400, 24000),
}


def fit_affine(src, dst, threshold_px):
    M, inlier_mask = cv2.estimateAffine2D(
        src, dst, method=cv2.RANSAC,
        ransacReprojThreshold=threshold_px,
        confidence=RANSAC_CONFIDENCE, maxIters=RANSAC_MAX_ITERS,
    )
    if M is None:
        return None
    inlier_mask = inlier_mask.ravel().astype(bool)
    n_inliers = int(inlier_mask.sum())
    n_total = len(src)

    src_in = src[inlier_mask]
    dst_in = dst[inlier_mask]
    src_h = np.hstack([src_in, np.ones((len(src_in), 1))])
    predicted = (M @ src_h.T).T
    residuals = predicted - dst_in
    rmse_px = float(np.sqrt(np.mean(np.sum(residuals ** 2, axis=1))))

    a, b, tx = M[0]
    c, d, ty = M[1]
    scale_x, scale_y = float(np.hypot(a, c)), float(np.hypot(b, d))
    rotation_deg = float(np.degrees(np.arctan2(c, a)))

    return {
        "n_inliers": n_inliers, "n_total": n_total,
        "inlier_ratio": n_inliers / n_total, "rmse_px": rmse_px,
        "tx": tx, "ty": ty, "scale_x": scale_x, "scale_y": scale_y,
        "rotation_deg": rotation_deg,
    }


df = pd.read_csv(CORE_CSV)
df = df[df["confidence"] >= MIN_CONFIDENCE].reset_index(drop=True)
print(f"Total matches after confidence filter: {len(df)}\n")

print(f"{'cluster':45s} {'n_pts':>6s} {'inliers':>8s} {'ratio':>7s} "
      f"{'rmse_px':>8s} {'tx':>9s} {'ty':>10s} {'scale_x':>8s} {'rot_deg':>8s}")

for name, (row_min, row_max) in CLUSTERS.items():
    subset = df[(df["ohrc_y"] >= row_min) & (df["ohrc_y"] < row_max)]
    if len(subset) < 3:
        print(f"{name:45s} {len(subset):>6d}  (too few points to fit)")
        continue

    src = subset[["ohrc_x", "ohrc_y"]].to_numpy(dtype=np.float64)
    dst = subset[["nac_x", "nac_y"]].to_numpy(dtype=np.float64)
    r = fit_affine(src, dst, RANSAC_THRESHOLD_PX)

    if r is None:
        print(f"{name:45s} {len(subset):>6d}  (RANSAC failed to fit)")
        continue

    print(f"{name:45s} {len(subset):>6d} {r['n_inliers']:>8d} "
          f"{r['inlier_ratio']:>7.3f} {r['rmse_px']:>8.2f} "
          f"{r['tx']:>9.2f} {r['ty']:>10.2f} {r['scale_x']:>8.4f} "
          f"{r['rotation_deg']:>8.3f}")

print("\nInterpretation:")
print("- If the top cluster's OWN inlier ratio and RMSE (isolated) look")
print("  similarly good to the middle/lower clusters, but its (tx,ty)")
print("  differs notably from the others -> real local terrain-relief")
print("  offset. Expected and fine; a single global affine just can't")
print("  capture it, which is a modeling limitation, not a matching bug.")
print("- If the top cluster's own inlier ratio stays low / RMSE stays")
print("  high even fit on its own -> likely false/repetitive-texture")
print("  matches, not genuine correspondences. Worth visually checking")
print("  match_visualizations for tile 13 (and its bbox neighbours) to")
print("  confirm before including these points in anything downstream.")