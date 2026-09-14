"""
Expected-offset check for the OHRC-NAC tiled SuperGlue baseline.

You reported (in QGIS) that a feature which SHOULD be at
    lat, lon = (-13.794553, 25.199587)     <- "expected" / correct location
is actually appearing at
    lat, lon = (-13.889246, 25.204260)     <- "actual" location in the
                                               georeferenced OHRC

This script:
  1. Converts both lat/lon points into pixel coordinates on the SAME
     grid as your matches CSV, using the actual CRS/transform stored in
     the GeoTIFF (via pyproj), not a manual sphere approximation.
  2. Reports the expected pixel displacement (dx, dy) and its magnitude
     in pixels and meters.
  3. Loads your existing matches CSVs (core-only and all) and flags
     which matches fall within a tolerance of that expected offset, so
     you know exactly which tile_id / visualization PNGs to look at.

This is purely a diagnostic/inspection step on ALREADY-COMPUTED matches.
It does not add a phase-correlation or displacement-estimation stage to
the matching pipeline itself, and it does not touch the source GeoTIFFs.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import pyproj


# ============================================================
# CONFIG
# ============================================================

# Either raster works; they're confirmed to share the same grid/transform.
REFERENCE_TIF = Path(r"C:\#Padhai\E\SIH1\normalized\OHRC_normalized.tif")

MATCH_DIR = Path(r"C:\#Padhai\E\SIH1\SuperGluePretrainedNetwork\lunar_matches_v4")
MATCHES_CORE_CSV = MATCH_DIR / "ohrc_nac_superglue_matches_core_only.csv"
MATCHES_ALL_CSV = MATCH_DIR / "ohrc_nac_superglue_matches.csv"

DIAG_DIR = MATCH_DIR / "diagnostics"
DIAG_DIR.mkdir(parents=True, exist_ok=True)

# "Expected" = where the feature should be (i.e. the geometrically
# correct / reference location). "Actual" = where it is actually found
# in the imperfectly-georeferenced OHRC.
EXPECTED_LAT, EXPECTED_LON = -13.794553, 25.199587
ACTUAL_LAT, ACTUAL_LON = -13.889246, 25.204260

# How close (in pixels) a match's own (dx, dy) must be to the expected
# offset vector to be flagged as "plausibly genuine". Start generous;
# tighten once you've visually confirmed a few.
TOLERANCE_PX = 80.0


# ============================================================
# 1. CONVERT LAT/LON -> PIXEL COORDINATES USING THE REAL CRS
# ============================================================

with rasterio.open(REFERENCE_TIF) as src:
    raster_crs = src.crs
    transform = src.transform
    px_size = src.res[0]

print(f"Raster CRS: {raster_crs}")
print(f"Pixel size: {px_size:.6f} m")

geodetic_crs = pyproj.CRS(raster_crs).geodetic_crs

if geodetic_crs is None:
    # Fallback: manual equirectangular sphere approximation.
    # Assumes standard parallel = 0 (equator), mean lunar radius.
    print("WARNING: could not derive geodetic CRS automatically; "
          "falling back to manual equirectangular sphere approximation "
          "(standard parallel = 0, R = 1737400 m). Verify against your "
          "ISIS mapping parameters if precision matters.")

    R = 1737400.0  # mean lunar radius, meters

    def lonlat_to_xy(lon, lat):
        x = R * np.radians(lon)
        y = R * np.radians(lat)
        return x, y

    expected_x, expected_y = lonlat_to_xy(EXPECTED_LON, EXPECTED_LAT)
    actual_x, actual_y = lonlat_to_xy(ACTUAL_LON, ACTUAL_LAT)

else:
    print(f"Geodetic CRS derived: {geodetic_crs.name}")
    transformer = pyproj.Transformer.from_crs(
        geodetic_crs, raster_crs, always_xy=True
    )
    expected_x, expected_y = transformer.transform(EXPECTED_LON, EXPECTED_LAT)
    actual_x, actual_y = transformer.transform(ACTUAL_LON, ACTUAL_LAT)

# Map (x, y) in meters -> pixel (col, row) via the inverse affine transform.
inv_transform = ~transform
expected_col, expected_row = inv_transform * (expected_x, expected_y)
actual_col, actual_row = inv_transform * (actual_x, actual_y)

# Convention matching the matches CSV: dx = nac_x - ohrc_x.
# "expected" (correct/reference location) plays the role of the NAC-side
# truth; "actual" (where OHRC content really is) plays the role of the
# OHRC-side (shifted) coordinate.
expected_dx = expected_col - actual_col
expected_dy = expected_row - actual_row
expected_mag_px = float(np.hypot(expected_dx, expected_dy))
expected_mag_m = expected_mag_px * px_size

print("\n--- EXPECTED OFFSET (from your reported lat/lon shift) ---")
print(f"Expected pixel (col,row): ({expected_col:.2f}, {expected_row:.2f})")
print(f"Actual   pixel (col,row): ({actual_col:.2f}, {actual_row:.2f})")
print(f"Expected dx (nac-like - ohrc-like): {expected_dx:.2f} px")
print(f"Expected dy (nac-like - ohrc-like): {expected_dy:.2f} px")
print(f"Expected displacement magnitude   : {expected_mag_px:.2f} px "
      f"({expected_mag_m/1000:.3f} km)")
print("\nNOTE: sign convention assumes 'expected' corresponds to the "
      "NAC-side (reference) coordinate and 'actual' corresponds to the "
      "OHRC-side (shifted) coordinate, matching dx = nac_x - ohrc_x in "
      "your matches CSV. If flagged matches turn out visually inverted "
      "(pointing the opposite way), simply negate expected_dx/expected_dy "
      "and re-run.")


# ============================================================
# 2. FLAG MATCHES NEAR THE EXPECTED OFFSET
# ============================================================

def analyze(csv_path, label):
    if not csv_path.exists():
        print(f"\n{csv_path} not found, skipping.")
        return None

    df = pd.read_csv(csv_path)
    if len(df) == 0:
        print(f"\n{label}: CSV is empty.")
        return df

    df["dx"] = df["nac_x"] - df["ohrc_x"]
    df["dy"] = df["nac_y"] - df["ohrc_y"]
    df["dist_from_expected_px"] = np.hypot(
        df["dx"] - expected_dx, df["dy"] - expected_dy
    )

    flagged = df[df["dist_from_expected_px"] <= TOLERANCE_PX].copy()
    flagged = flagged.sort_values("dist_from_expected_px")

    print(f"\n--- {label} ---")
    print(f"Total matches            : {len(df)}")
    print(f"Within {TOLERANCE_PX:.0f} px of expected offset: {len(flagged)}")

    if len(flagged) > 0:
        print("\nFlagged matches (closest first):")
        cols = ["tile_id", "tile_row", "tile_col", "ohrc_x", "ohrc_y",
                "nac_x", "nac_y", "confidence", "dx", "dy",
                "dist_from_expected_px"]
        print(flagged[cols].to_string(index=False))

        tile_ids = sorted(flagged["tile_id"].unique().tolist())
        print(f"\nTile IDs to inspect in match_visualizations\\ "
              f"(tile_XXXX_matches.png): {tile_ids}")

    return df, flagged


core_result = analyze(MATCHES_CORE_CSV, "CORE (deduplicated) matches")
all_result = analyze(MATCHES_ALL_CSV, "ALL accepted matches (incl. overlap duplicates)")


# ============================================================
# 3. SAVE FLAGGED SUBSET + ANNOTATED SCATTER PLOT
# ============================================================

if all_result is not None:
    df_all, flagged_all = all_result

    if len(flagged_all) > 0:
        out_csv = DIAG_DIR / "matches_near_expected_offset.csv"
        flagged_all.to_csv(out_csv, index=False)
        print(f"\nSaved flagged subset: {out_csv}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 6))
        sc = ax.scatter(df_all["dx"], df_all["dy"], c=df_all["confidence"],
                         cmap="viridis", s=15, alpha=0.6, label="all matches")
        ax.scatter([expected_dx], [expected_dy], c="red", marker="*",
                   s=250, label="expected offset", zorder=5)
        circle = plt.Circle((expected_dx, expected_dy), TOLERANCE_PX,
                             color="red", fill=False, linestyle="--",
                             label=f"{TOLERANCE_PX:.0f}px tolerance")
        ax.add_patch(circle)
        ax.axhline(0, color="gray", lw=0.5)
        ax.axvline(0, color="gray", lw=0.5)
        ax.set_xlabel("dx = nac_x - ohrc_x (px)")
        ax.set_ylabel("dy = nac_y - ohrc_y (px)")
        ax.set_title("All match displacements vs. expected offset")
        ax.legend(loc="best", fontsize=8)
        ax.set_aspect("equal", adjustable="datalim")
        fig.colorbar(sc, ax=ax, label="confidence")
        fig.tight_layout()
        plot_path = DIAG_DIR / "displacement_vs_expected_offset.png"
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"Saved: {plot_path}")
    except ImportError:
        print("matplotlib not available; skipped annotated scatter plot.")
