from pathlib import Path
import json

import numpy as np
import pandas as pd
# pyrefly: ignore [missing-import]
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
MIN_POINTS_FOR_ROBUST_SEGMENT = 300   # excludes the two thin/noisy segments
POLY_DEGREE = 2
SIGMA_CLIP_ROUNDS = 3
SIGMA_CLIP_THRESHOLD_PX = 40          # residual beyond this is excluded per round
CONSISTENCY_THRESHOLD_PX = 15         # same threshold used for the affine baseline
PIXEL_SIZE_M = 1.1523692688652

CLUSTERS = {
    "top_cluster": (6800, 9000),
    "middle_cluster": (9200, 14400),
    "lower_cluster": (19400, 24000),
}

# Baseline numbers from the earlier single global affine fit, for
# direct comparison (from ransac_transform.json at threshold=15px).
BASELINE_AFFINE = {
    "inlier_ratio": 0.4293, "rmse_px": 7.041,
    "top_cluster_accept_pct": 0.0,
}


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(CORE_CSV)
df = df[df["confidence"] >= MIN_CONFIDENCE].reset_index(drop=True)
print(f"Total matches after confidence >= {MIN_CONFIDENCE}: {len(df)}")

row_min, row_max = df["ohrc_y"].min(), df["ohrc_y"].max()
segment_starts = np.arange(row_min, row_max, SEGMENT_SIZE_PX)


# ============================================================
# STEP 1: ROBUST FIXED ROTATION + SCALE FROM WELL-SUPPORTED SEGMENTS
# ============================================================

seg_rotations, seg_scales = [], []
print("\n--- Per-segment fits used for robust rotation/scale estimate ---")
for seg_start in segment_starts:
    seg_end = seg_start + SEGMENT_SIZE_PX
    subset = df[(df["ohrc_y"] >= seg_start) & (df["ohrc_y"] < seg_end)]
    if len(subset) < MIN_POINTS_FOR_ROBUST_SEGMENT:
        print(f"[{seg_start:.0f},{seg_end:.0f}] n={len(subset)} -- excluded (too few points)")
        continue

    src = subset[["ohrc_x", "ohrc_y"]].to_numpy(dtype=np.float64)
    dst = subset[["nac_x", "nac_y"]].to_numpy(dtype=np.float64)
    M, mask = cv2.estimateAffine2D(
        src, dst, method=cv2.RANSAC, ransacReprojThreshold=RANSAC_THRESHOLD_PX,
        confidence=0.999, maxIters=5000,
    )
    if M is None:
        continue
    a, b, tx = M[0]
    c, d, ty = M[1]
    scale = (np.hypot(a, c) + np.hypot(b, d)) / 2.0
    rotation_deg = float(np.degrees(np.arctan2(c, a)))
    seg_rotations.append(rotation_deg)
    seg_scales.append(scale)
    print(f"[{seg_start:.0f},{seg_end:.0f}] n={len(subset)} rotation={rotation_deg:.3f} scale={scale:.4f} -- included")

theta_deg = float(np.median(seg_rotations))
scale = float(np.median(seg_scales))
theta_rad = np.radians(theta_deg)

print(f"\nRobust FIXED rotation: {theta_deg:.3f} deg")
print(f"Robust FIXED scale   : {scale:.4f}")

R = scale * np.array([
    [np.cos(theta_rad), -np.sin(theta_rad)],
    [np.sin(theta_rad), np.cos(theta_rad)],
])


# ============================================================
# STEP 2: ROW-DEPENDENT TRANSLATION VIA SIGMA-CLIPPED POLYNOMIAL FIT
# ============================================================

src_all = df[["ohrc_x", "ohrc_y"]].to_numpy(dtype=np.float64)
dst_all = df[["nac_x", "nac_y"]].to_numpy(dtype=np.float64)
rows_all = df["ohrc_y"].to_numpy(dtype=np.float64)

rotated = (R @ src_all.T).T   # rotation+scale applied, no translation yet
residual_tx = dst_all[:, 0] - rotated[:, 0]   # required translation per point
residual_ty = dst_all[:, 1] - rotated[:, 1]

keep = np.ones(len(df), dtype=bool)
for round_i in range(SIGMA_CLIP_ROUNDS):
    coeffs_x = np.polyfit(rows_all[keep], residual_tx[keep], POLY_DEGREE)
    coeffs_y = np.polyfit(rows_all[keep], residual_ty[keep], POLY_DEGREE)
    pred_tx = np.polyval(coeffs_x, rows_all)
    pred_ty = np.polyval(coeffs_y, rows_all)
    resid = np.hypot(residual_tx - pred_tx, residual_ty - pred_ty)
    new_keep = resid <= SIGMA_CLIP_THRESHOLD_PX
    print(f"Sigma-clip round {round_i+1}: kept {new_keep.sum()} / {len(df)} points")
    keep = new_keep

print(f"\nFinal tx(row) polynomial coeffs (degree {POLY_DEGREE}): {coeffs_x}")
print(f"Final ty(row) polynomial coeffs (degree {POLY_DEGREE}): {coeffs_y}")


# ============================================================
# FULL MODEL + EVALUATION
# ============================================================

def predict(ohrc_pts):
    rotated = (R @ ohrc_pts.T).T
    rows = ohrc_pts[:, 1]
    tx = np.polyval(coeffs_x, rows)
    ty = np.polyval(coeffs_y, rows)
    return rotated + np.stack([tx, ty], axis=1)


predicted_all = predict(src_all)
residuals = predicted_all - dst_all
rmse_x = float(np.sqrt(np.mean(residuals[:, 0] ** 2)))
rmse_y = float(np.sqrt(np.mean(residuals[:, 1] ** 2)))
rmse_combined = float(np.sqrt(np.mean(np.sum(residuals ** 2, axis=1))))
dist = np.hypot(residuals[:, 0], residuals[:, 1])
consistent = dist <= CONSISTENCY_THRESHOLD_PX
consistency_ratio = float(consistent.mean())

print("\n" + "=" * 90)
print("HYBRID MODEL (fixed rotation+scale, row-dependent translation) -- OVERALL")
print("=" * 90)
print(f"n_points               : {len(df)}")
print(f"Within {CONSISTENCY_THRESHOLD_PX}px (\"consistency ratio\") : "
      f"{consistent.sum()} / {len(df)} ({consistency_ratio*100:.1f}%)")
print(f"RMSE_X / RMSE_Y / RMSE : {rmse_x:.2f} / {rmse_y:.2f} / {rmse_combined:.2f} px "
      f"({rmse_combined*PIXEL_SIZE_M:.2f} m)")

print(f"\nBASELINE (single global affine, same 15px threshold): "
      f"inlier_ratio={BASELINE_AFFINE['inlier_ratio']*100:.1f}%, "
      f"RMSE={BASELINE_AFFINE['rmse_px']:.2f}px")

# Per-cluster comparison, especially the previously-rejected top cluster.
print("\n--- PER-CLUSTER: hybrid model consistency ratio & RMSE ---")
print(f"{'cluster':>15s} {'n_pts':>6s} {'consist_ratio':>14s} {'rmse_px':>9s}")
for name, (r0, r1) in CLUSTERS.items():
    mask_c = (df["ohrc_y"] >= r0) & (df["ohrc_y"] < r1)
    if mask_c.sum() == 0:
        continue
    d = dist[mask_c.to_numpy()]
    ratio = float((d <= CONSISTENCY_THRESHOLD_PX).mean())
    rmse_c = float(np.sqrt(np.mean(d ** 2)))
    print(f"{name:>15s} {int(mask_c.sum()):>6d} {ratio*100:>13.1f}% {rmse_c:>9.2f}")

print("\n(Recall: global affine gave the top_cluster 0.0% acceptance.)")


# ============================================================
# SAVE MODEL + DIAGNOSTICS
# ============================================================

model_summary = {
    "rotation_deg": theta_deg, "scale": scale,
    "translation_poly_degree": POLY_DEGREE,
    "tx_poly_coeffs": coeffs_x.tolist(), "ty_poly_coeffs": coeffs_y.tolist(),
    "min_confidence_filter": MIN_CONFIDENCE,
    "consistency_threshold_px": CONSISTENCY_THRESHOLD_PX,
    "overall_consistency_ratio": consistency_ratio,
    "overall_rmse_x_px": rmse_x, "overall_rmse_y_px": rmse_y,
    "overall_rmse_combined_px": rmse_combined,
    "overall_rmse_combined_m": rmse_combined * PIXEL_SIZE_M,
    "baseline_affine_for_comparison": BASELINE_AFFINE,
}
with open(MATCH_DIR / "hybrid_transform_model.json", "w", encoding="utf-8") as f:
    json.dump(model_summary, f, indent=2)
print(f"\nSaved: {MATCH_DIR / 'hybrid_transform_model.json'}")

# Plot: residual tx/ty vs row, with fitted polynomial curves overlaid.
row_line = np.linspace(row_min, row_max, 300)
fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
axes[0].scatter(rows_all[keep], residual_tx[keep], s=4, alpha=0.3, color="tab:blue", label="kept points")
axes[0].scatter(rows_all[~keep], residual_tx[~keep], s=4, alpha=0.3, color="lightgray", label="clipped outliers")
axes[0].plot(row_line, np.polyval(coeffs_x, row_line), color="red", lw=2, label="fitted tx(row)")
axes[0].set_ylabel("required tx (px)")
axes[0].legend(fontsize=8)
axes[0].set_title("Row-dependent translation fit: X")

axes[1].scatter(rows_all[keep], residual_ty[keep], s=4, alpha=0.3, color="tab:orange", label="kept points")
axes[1].scatter(rows_all[~keep], residual_ty[~keep], s=4, alpha=0.3, color="lightgray", label="clipped outliers")
axes[1].plot(row_line, np.polyval(coeffs_y, row_line), color="red", lw=2, label="fitted ty(row)")
axes[1].set_ylabel("required ty (px)")
axes[1].set_xlabel("OHRC row (px)")
axes[1].legend(fontsize=8)
axes[1].set_title("Row-dependent translation fit: Y")

fig.tight_layout()
fit_plot_path = DIAG_DIR / "hybrid_translation_fit.png"
fig.savefig(fit_plot_path, dpi=150)
plt.close(fig)
print(f"Saved: {fit_plot_path}")

# Plot: final residual magnitude vs row (should be small/flat everywhere
# if the model is good, unlike the global affine which spiked at the
# top cluster).
fig, ax = plt.subplots(figsize=(8, 4))
ax.scatter(rows_all, dist, s=4, alpha=0.3, color="tab:green")
ax.axhline(CONSISTENCY_THRESHOLD_PX, color="red", ls="--", lw=1,
           label=f"{CONSISTENCY_THRESHOLD_PX}px threshold")
ax.set_xlabel("OHRC row (px)")
ax.set_ylabel("residual distance (px)")
ax.set_title("Hybrid model residuals vs row (compare to global affine's top-cluster failure)")
ax.legend()
fig.tight_layout()
resid_plot_path = DIAG_DIR / "hybrid_residuals_vs_row.png"
fig.savefig(resid_plot_path, dpi=150)
plt.close(fig)
print(f"Saved: {resid_plot_path}")

# Save inlier/consistent correspondence set for downstream use.
consistent_df = df[consistent].copy()
consistent_csv = MATCH_DIR / "hybrid_model_consistent_matches.csv"
consistent_df.to_csv(consistent_csv, index=False)
print(f"Saved: {consistent_csv}")