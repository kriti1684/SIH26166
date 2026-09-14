from pathlib import Path
import json

import numpy as np
import pandas as pd


MATCH_DIR = Path(r"C:\#Padhai\E\SIH1\SuperGluePretrainedNetwork\lunar_matches_v5")
CONSISTENT_CSV = MATCH_DIR / "hybrid_model_consistent_matches.csv"
MODEL_JSON = MATCH_DIR / "hybrid_transform_model.json"
PIXEL_SIZE_M = 1.1523692688652

with open(MODEL_JSON, "r", encoding="utf-8") as f:
    model = json.load(f)

theta_rad = np.radians(model["rotation_deg"])
scale = model["scale"]
coeffs_x = np.array(model["tx_poly_coeffs"])
coeffs_y = np.array(model["ty_poly_coeffs"])

R = scale * np.array([
    [np.cos(theta_rad), -np.sin(theta_rad)],
    [np.sin(theta_rad), np.cos(theta_rad)],
])

df = pd.read_csv(CONSISTENT_CSV)
src = df[["ohrc_x", "ohrc_y"]].to_numpy(dtype=np.float64)
dst = df[["nac_x", "nac_y"]].to_numpy(dtype=np.float64)

rotated = (R @ src.T).T
rows = src[:, 1]
tx = np.polyval(coeffs_x, rows)
ty = np.polyval(coeffs_y, rows)
predicted = rotated + np.stack([tx, ty], axis=1)

residuals = predicted - dst
rmse_x = float(np.sqrt(np.mean(residuals[:, 0] ** 2)))
rmse_y = float(np.sqrt(np.mean(residuals[:, 1] ** 2)))
rmse_combined = float(np.sqrt(np.mean(np.sum(residuals ** 2, axis=1))))

print(f"Consistent-subset n_points: {len(df)}")
print(f"RMSE_X: {rmse_x:.3f} px")
print(f"RMSE_Y: {rmse_y:.3f} px")
print(f"RMSE (combined): {rmse_combined:.3f} px = {rmse_combined * PIXEL_SIZE_M:.3f} m")

print("\n--- Comparison ---")
print(f"{'Model':<20s} {'Inlier/consistent ratio':>25s} {'RMSE (px)':>12s}")
print(f"{'Global affine':<20s} {model['baseline_affine_for_comparison']['inlier_ratio']*100:>24.1f}% "
      f"{model['baseline_affine_for_comparison']['rmse_px']:>12.3f}")
print(f"{'Hybrid model':<20s} {model['overall_consistency_ratio']*100:>24.1f}% "
      f"{rmse_combined:>12.3f}")

model["consistent_subset_rmse_x_px"] = rmse_x
model["consistent_subset_rmse_y_px"] = rmse_y
model["consistent_subset_rmse_combined_px"] = rmse_combined
model["consistent_subset_rmse_combined_m"] = rmse_combined * PIXEL_SIZE_M

with open(MODEL_JSON, "w", encoding="utf-8") as f:
    json.dump(model, f, indent=2)
print(f"\nUpdated: {MODEL_JSON}")