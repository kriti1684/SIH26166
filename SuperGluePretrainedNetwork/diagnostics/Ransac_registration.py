

from pathlib import Path
import json

import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

MATCH_DIR = Path(r"C:\#Padhai\E\SIH1\SuperGluePretrainedNetwork\lunar_matches_v5")
CORE_CSV = MATCH_DIR / "ohrc_nac_superglue_matches_core_only.csv"

DIAG_DIR = MATCH_DIR / "diagnostics"
DIAG_DIR.mkdir(parents=True, exist_ok=True)

MIN_CONFIDENCE = 0.30

# Reprojection thresholds to sweep, in pixels.
RANSAC_THRESHOLDS_PX = [5, 10, 15, 20, 30]

# Which threshold's result to save as the "chosen" transform + inlier CSV.
# Pick based on the printed sweep table (a threshold where inlier ratio
# and RMSE both look reasonable, not just the one with the most inliers).
CHOSEN_THRESHOLD_PX = 15

RANSAC_CONFIDENCE = 0.999
RANSAC_MAX_ITERS = 5000

PIXEL_SIZE_M = 1.1523692688652


# ============================================================
# LOAD + FILTER
# ============================================================

df = pd.read_csv(CORE_CSV)
print(f"Total core matches           : {len(df)}")

df = df[df["confidence"] >= MIN_CONFIDENCE].reset_index(drop=True)
print(f"After confidence >= {MIN_CONFIDENCE}   : {len(df)}")

if len(df) < 3:
    raise RuntimeError(
        "Fewer than 3 matches remain after filtering -- cannot fit an "
        "affine transform. Lower MIN_CONFIDENCE or investigate why so "
        "few high-confidence matches exist."
    )

src_pts = df[["ohrc_x", "ohrc_y"]].to_numpy(dtype=np.float64)
dst_pts = df[["nac_x", "nac_y"]].to_numpy(dtype=np.float64)


# ============================================================
# RANSAC THRESHOLD SWEEP
# ============================================================

def fit_and_evaluate(src, dst, threshold_px):
    M, inlier_mask = cv2.estimateAffine2D(
        src, dst,
        method=cv2.RANSAC,
        ransacReprojThreshold=threshold_px,
        confidence=RANSAC_CONFIDENCE,
        maxIters=RANSAC_MAX_ITERS,
    )
    if M is None:
        return None

    inlier_mask = inlier_mask.ravel().astype(bool)
    n_inliers = int(inlier_mask.sum())
    n_total = len(src)
    inlier_ratio = n_inliers / n_total

    # Apply the fitted transform to the inlier source points, compare
    # against their actual destination points.
    src_inliers = src[inlier_mask]
    dst_inliers = dst[inlier_mask]
    src_h = np.hstack([src_inliers, np.ones((len(src_inliers), 1))])
    predicted = (M @ src_h.T).T
    residuals = predicted - dst_inliers
    rmse_px = float(np.sqrt(np.mean(np.sum(residuals ** 2, axis=1))))
    rmse_m = rmse_px * PIXEL_SIZE_M

    return {
        "threshold_px": threshold_px,
        "M": M,
        "inlier_mask": inlier_mask,
        "n_inliers": n_inliers,
        "n_total": n_total,
        "inlier_ratio": inlier_ratio,
        "rmse_px": rmse_px,
        "rmse_m": rmse_m,
    }


print("\n--- RANSAC THRESHOLD SWEEP (full affine) ---")
print(f"{'thresh_px':>10s} {'n_inliers':>10s} {'n_total':>8s} "
      f"{'inlier_ratio':>13s} {'rmse_px':>9s} {'rmse_m':>8s}")

results = {}
for t in RANSAC_THRESHOLDS_PX:
    r = fit_and_evaluate(src_pts, dst_pts, t)
    if r is None:
        print(f"{t:>10d}   RANSAC failed to find a model")
        continue
    results[t] = r
    print(f"{t:>10d} {r['n_inliers']:>10d} {r['n_total']:>8d} "
          f"{r['inlier_ratio']:>13.4f} {r['rmse_px']:>9.2f} {r['rmse_m']:>8.2f}")


# ============================================================
# REPORT THE CHOSEN THRESHOLD'S TRANSFORM
# ============================================================

if CHOSEN_THRESHOLD_PX not in results:
    raise RuntimeError(
        f"CHOSEN_THRESHOLD_PX={CHOSEN_THRESHOLD_PX} not in swept thresholds "
        f"or RANSAC failed for it. Pick one from the table above."
    )

chosen = results[CHOSEN_THRESHOLD_PX]
M = chosen["M"]

a, b, tx = M[0]
c, d, ty = M[1]
scale_x = float(np.hypot(a, c))
scale_y = float(np.hypot(b, d))
rotation_deg = float(np.degrees(np.arctan2(c, a)))

print(f"\n--- CHOSEN TRANSFORM (threshold = {CHOSEN_THRESHOLD_PX} px) ---")
print(f"Inliers        : {chosen['n_inliers']} / {chosen['n_total']} "
      f"({chosen['inlier_ratio']*100:.1f}%)")
print(f"RMSE           : {chosen['rmse_px']:.3f} px ({chosen['rmse_m']:.3f} m)")
print(f"Translation    : tx={tx:.2f} px, ty={ty:.2f} px")
print(f"Approx scale   : sx={scale_x:.4f}, sy={scale_y:.4f}")
print(f"Approx rotation: {rotation_deg:.3f} degrees")
print("(scale/rotation are approximate decompositions of a full affine "
      "matrix, which may also include shear -- see raw matrix below)")
print(f"Affine matrix (OHRC -> NAC):\n{M}")


# ============================================================
# SAVE OUTPUTS
# ============================================================

# Inlier-only correspondence CSV (RANSAC-consistent set).
inlier_mask = chosen["inlier_mask"]
inliers_df = df[inlier_mask].copy()
inliers_csv = MATCH_DIR / f"ransac_inliers_thresh{CHOSEN_THRESHOLD_PX}px.csv"
inliers_df.to_csv(inliers_csv, index=False)
print(f"\nSaved inlier correspondences: {inliers_csv}")

# Transform + evaluation summary JSON.
transform_summary = {
    "chosen_threshold_px": CHOSEN_THRESHOLD_PX,
    "min_confidence_filter": MIN_CONFIDENCE,
    "affine_matrix": M.tolist(),
    "translation_px": [tx, ty],
    "approx_scale": [scale_x, scale_y],
    "approx_rotation_deg": rotation_deg,
    "n_inliers": chosen["n_inliers"],
    "n_total_input": chosen["n_total"],
    "inlier_ratio": chosen["inlier_ratio"],
    "rmse_px": chosen["rmse_px"],
    "rmse_m": chosen["rmse_m"],
    "threshold_sweep": {
        str(t): {
            "n_inliers": r["n_inliers"], "n_total": r["n_total"],
            "inlier_ratio": r["inlier_ratio"], "rmse_px": r["rmse_px"],
            "rmse_m": r["rmse_m"],
        } for t, r in results.items()
    },
}
transform_json = MATCH_DIR / "ransac_transform.json"
with open(transform_json, "w", encoding="utf-8") as f:
    json.dump(transform_summary, f, indent=2)
print(f"Saved transform summary: {transform_json}")

# Spatial plot: inliers vs outliers over OHRC bbox (uniformity check).
fig, ax = plt.subplots(figsize=(6, 6))
outliers_df = df[~inlier_mask]
ax.scatter(outliers_df["ohrc_x"], outliers_df["ohrc_y"], s=6, c="red",
           alpha=0.4, label=f"outliers (n={len(outliers_df)})")
ax.scatter(inliers_df["ohrc_x"], inliers_df["ohrc_y"], s=6, c="green",
           alpha=0.6, label=f"inliers (n={len(inliers_df)})")
ax.invert_yaxis()
ax.set_xlabel("OHRC column (px)")
ax.set_ylabel("OHRC row (px)")
ax.set_title(f"RANSAC inliers/outliers, threshold={CHOSEN_THRESHOLD_PX}px")
ax.legend(loc="best", fontsize=8)
ax.set_aspect("equal", adjustable="datalim")
fig.tight_layout()
plot_path = DIAG_DIR / f"ransac_inliers_thresh{CHOSEN_THRESHOLD_PX}px.png"
fig.savefig(plot_path, dpi=150)
plt.close(fig)
print(f"Saved: {plot_path}")

print("\nNOTE: this script estimates and evaluates the global transform. "
      "Applying it to warp the FULL OHRC GeoTIFF (the actual registered "
      "product) is a separate next step, once you've confirmed this "
      "transform's inlier ratio/RMSE are acceptable.")