import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


MATCH_DIR = Path(r"C:\#Padhai\E\SIH1\SuperGluePretrainedNetwork\lunar_matches_v5")
CORE_CSV = MATCH_DIR / "ohrc_nac_superglue_matches_core_only.csv"
SUMMARY_CSV = MATCH_DIR / "tile_summary.csv"

DIAG_DIR = MATCH_DIR / "diagnostics"
DIAG_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CORE_CSV)
print(f"Total core matches: {len(df)}")

df["dx"] = df["nac_x"] - df["ohrc_x"]
df["dy"] = df["nac_y"] - df["ohrc_y"]

print("\n--- RESIDUAL DISPLACEMENT (after coarse offset correction) ---")
print(df[["dx", "dy"]].describe())

dx_std, dy_std = df["dx"].std(), df["dy"].std()
dx_med, dy_med = df["dx"].median(), df["dy"].median()
print(f"\nMedian (dx, dy) = ({dx_med:.2f}, {dy_med:.2f}) px")
print(f"Std    (dx, dy) = ({dx_std:.2f}, {dy_std:.2f}) px")
print("Tight std around a consistent median = genuine matches with a small")
print("residual local misalignment (expected, since the coarse offset was")
print("only approximate). Large/scattered std = likely contamination from")
print("repetitive-texture false matches.")

print("\n--- CONFIDENCE ---")
print(df["confidence"].describe())

# Flag outlier matches: far from the residual median offset.
df["dist_from_median"] = np.hypot(df["dx"] - dx_med, df["dy"] - dy_med)
outlier_frac = (df["dist_from_median"] > 50).mean()
print(f"\nFraction of core matches >50px from the median residual offset: "
      f"{outlier_frac*100:.1f}%")

# Scatter of residual displacement
fig, ax = plt.subplots(figsize=(6, 6))
sc = ax.scatter(df["dx"], df["dy"], c=df["confidence"], cmap="viridis", s=6, alpha=0.5)
ax.axhline(dy_med, color="red", lw=0.5, ls="--")
ax.axvline(dx_med, color="red", lw=0.5, ls="--")
ax.set_xlabel("residual dx (px)")
ax.set_ylabel("residual dy (px)")
ax.set_title(f"Residual displacement, n={len(df)} (red = median)")
fig.colorbar(sc, ax=ax, label="confidence")
fig.tight_layout()
fig.savefig(DIAG_DIR / "v5_residual_displacement.png", dpi=150)
plt.close(fig)
print(f"\nSaved: {DIAG_DIR / 'v5_residual_displacement.png'}")

# Zoomed-in version (most matches should be within a small window if clustered)
fig, ax = plt.subplots(figsize=(6, 6))
sc = ax.scatter(df["dx"], df["dy"], c=df["confidence"], cmap="viridis", s=8, alpha=0.5)
ax.set_xlim(dx_med - 100, dx_med + 100)
ax.set_ylim(dy_med - 100, dy_med + 100)
ax.axhline(dy_med, color="red", lw=0.5, ls="--")
ax.axvline(dx_med, color="red", lw=0.5, ls="--")
ax.set_xlabel("residual dx (px)")
ax.set_ylabel("residual dy (px)")
ax.set_title("Residual displacement, zoomed to +/-100px around median")
fig.colorbar(sc, ax=ax, label="confidence")
fig.tight_layout()
fig.savefig(DIAG_DIR / "v5_residual_displacement_zoomed.png", dpi=150)
plt.close(fig)
print(f"Saved: {DIAG_DIR / 'v5_residual_displacement_zoomed.png'}")

# Per-tile breakdown, sorted by core match count, to identify the
# highest-yield tiles worth visually inspecting first.
if SUMMARY_CSV.exists():
    summary = pd.read_csv(SUMMARY_CSV)
    ok = summary[summary["status"] == "ok"].sort_values("core_matches", ascending=False)
    print("\n--- TOP 10 TILES BY CORE MATCH COUNT (inspect these visualizations first) ---")
    print(ok[["tile_id", "tile_row", "tile_col", "ohrc_valid_fraction",
              "nac_valid_fraction", "core_matches"]].head(10).to_string(index=False))
    