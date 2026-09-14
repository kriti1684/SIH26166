import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models.matching import Matching


REGISTERED_PATH = Path(
    r"C:\#Padhai\E\SIH1\SuperGluePretrainedNetwork\lunar_matches_v5\OHRC_registered_v2.tif"
)
NAC_PATH = Path(r"C:\#Padhai\E\SIH1\normalized\NAC_normalized.tif")

OUT_DIR = Path(
    r"C:\#Padhai\E\SIH1\SuperGluePretrainedNetwork\lunar_matches_v5\diagnostics"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

TILE_SIZE = 1600
MIN_MATCH_CONFIDENCE = 0.50   # raised from 0.20 -- cleaner signal, fewer weak/noisy matches
NODATA_MARGIN_PX = 2

# Sigma-clipping applied AFTER matching, on the residuals themselves, to
# separate genuine sub-pixel-consistent matches from remaining outliers.
CLIP_ROUNDS = 3
CLIP_THRESHOLD_PX = 25.0

TEST_LOCATIONS = [
    ("top (~row 8400)", 7622, 1800),
    ("upper-middle (~row 10900)", 10122, 1800),
    ("middle (~row 13400)", 12622, 1800),
    ("lower (~row 20900)", 20122, 1800),
    ("bottom (~row 23400)", 22622, 1800),
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

config = {
    "superpoint": {"nms_radius": 3, "keypoint_threshold": 0.005, "max_keypoints": 2048},
    "superglue": {"weights": "outdoor", "sinkhorn_iterations": 20, "match_threshold": 0.2},
}
matching = Matching(config).eval().to(DEVICE)
print("SuperPoint + SuperGlue loaded.\n")


def read_window(src, row, col, size):
    row_end = min(row + size, src.height)
    col_end = min(col + size, src.width)
    h, w = row_end - row, col_end - col
    if h <= 0 or w <= 0:
        return None
    out = np.zeros((size, size), dtype=np.uint8)
    window = rasterio.windows.Window(col, row, w, h)
    data = src.read(1, window=window)
    out[:h, :w] = data
    return out


def to_tensor(image):
    t = torch.from_numpy(image.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
    return t.to(DEVICE)


def is_near_nodata(image, ix, iy, margin):
    h, w = image.shape
    y0, y1 = iy - margin, iy + margin + 1
    x0, x1 = ix - margin, ix + margin + 1
    if y0 < 0 or x0 < 0 or y1 > h or x1 > w:
        return True
    return bool(np.any(image[y0:y1, x0:x1] == 0))


all_records = []

with rasterio.open(REGISTERED_PATH) as ohrc_reg, rasterio.open(NAC_PATH) as nac:
    for label, row, col in TEST_LOCATIONS:
        ohrc_tile = read_window(ohrc_reg, row, col, TILE_SIZE)
        nac_tile = read_window(nac, row, col, TILE_SIZE)

        if ohrc_tile is None or nac_tile is None:
            print(f"{label}: window read failed, skipping.")
            continue

        ohrc_valid = np.count_nonzero(ohrc_tile) / ohrc_tile.size
        nac_valid = np.count_nonzero(nac_tile) / nac_tile.size
        if ohrc_valid < 0.05 or nac_valid < 0.05:
            print(f"{label}: too little valid data (OHRC={ohrc_valid:.2f}, "
                  f"NAC={nac_valid:.2f}), skipping.")
            continue

        image0 = to_tensor(ohrc_tile)
        image1 = to_tensor(nac_tile)

        with torch.no_grad():
            pred = matching({"image0": image0, "image1": image1})

        kp0 = pred["keypoints0"][0].detach().cpu().numpy()
        kp1 = pred["keypoints1"][0].detach().cpu().numpy()
        matches0 = pred["matches0"][0].detach().cpu().numpy()
        conf0 = pred["matching_scores0"][0].detach().cpu().numpy()

        valid_idx = np.where(matches0 > -1)[0]
        n_accepted = 0
        for idx0 in valid_idx:
            idx1 = int(matches0[idx0])
            if idx1 < 0:
                continue
            confidence = float(conf0[idx0])
            if confidence < MIN_MATCH_CONFIDENCE:
                continue

            x0, y0 = kp0[idx0]
            x1, y1 = kp1[idx1]
            ix0, iy0 = int(round(x0)), int(round(y0))
            ix1, iy1 = int(round(x1)), int(round(y1))

            if not (0 <= ix0 < TILE_SIZE and 0 <= iy0 < TILE_SIZE):
                continue
            if not (0 <= ix1 < TILE_SIZE and 0 <= iy1 < TILE_SIZE):
                continue
            if is_near_nodata(ohrc_tile, ix0, iy0, NODATA_MARGIN_PX):
                continue
            if is_near_nodata(nac_tile, ix1, iy1, NODATA_MARGIN_PX):
                continue

            dx = float(x1 - x0)
            dy = float(y1 - y0)
            all_records.append({
                "location": label, "row": row, "col": col,
                "dx": dx, "dy": dy, "confidence": confidence,
            })
            n_accepted += 1

        print(f"{label:30s} SP=({len(kp0)},{len(kp1)}) "
              f"accepted_matches(conf>={MIN_MATCH_CONFIDENCE})={n_accepted}")

if not all_records:
    raise RuntimeError("No matches found at any test location -- cannot verify registration.")

df = pd.DataFrame(all_records)

# ------------------------------------------------------------
# Sigma-clipping on residuals to isolate the genuine, consistent core
# from any remaining outlier matches.
# ------------------------------------------------------------
keep = np.ones(len(df), dtype=bool)
for round_i in range(CLIP_ROUNDS):
    med_dx = df.loc[keep, "dx"].median()
    med_dy = df.loc[keep, "dy"].median()
    dist = np.hypot(df["dx"] - med_dx, df["dy"] - med_dy)
    new_keep = (dist <= CLIP_THRESHOLD_PX).to_numpy()
    print(f"Clip round {round_i+1}: median=({med_dx:.2f},{med_dy:.2f}), "
          f"kept {new_keep.sum()}/{len(df)}")
    keep = new_keep

df["kept"] = keep
csv_path = OUT_DIR / "registration_verification_residuals_v2.csv"
df.to_csv(csv_path, index=False)

print("\n--- RAW (all conf>=0.5 matches) per-location median dx,dy ---")
print(df.groupby("location")[["dx", "dy"]].median())

clean = df[df["kept"]]
print(f"\n--- AFTER outlier clipping ({clean.shape[0]}/{len(df)} points kept) ---")
print("Per-location median (dx, dy) and MAD (robust spread):")
for label in clean["location"].unique():
    sub = clean[clean["location"] == label]
    mad_dx = float(np.median(np.abs(sub["dx"] - sub["dx"].median())))
    mad_dy = float(np.median(np.abs(sub["dy"] - sub["dy"].median())))
    print(f"  {label:28s} n={len(sub):4d}  median=({sub['dx'].median():6.2f},"
          f"{sub['dy'].median():7.2f})  MAD=({mad_dx:5.2f},{mad_dy:5.2f})")

overall_med_dx = clean["dx"].median()
overall_med_dy = clean["dy"].median()
overall_mad_dx = float(np.median(np.abs(clean["dx"] - overall_med_dx)))
overall_mad_dy = float(np.median(np.abs(clean["dy"] - overall_med_dy)))

print(f"\nOVERALL (clean): median dx={overall_med_dx:.2f} (MAD={overall_mad_dx:.2f}), "
      f"median dy={overall_med_dy:.2f} (MAD={overall_mad_dy:.2f}) px")
print(f"\n(For reference: BEFORE any correction, the offset was approximately "
      f"dx=-128px, dy=-2488px.)")

fig, ax = plt.subplots(figsize=(6, 6))
for label in df["location"].unique():
    sub_kept = df[(df["location"] == label) & (df["kept"])]
    sub_clipped = df[(df["location"] == label) & (~df["kept"])]
    ax.scatter(sub_kept["dx"], sub_kept["dy"], s=15, alpha=0.7, label=f"{label} (kept)")
    ax.scatter(sub_clipped["dx"], sub_clipped["dy"], s=10, alpha=0.2, color="gray")
ax.axhline(0, color="gray", lw=0.5)
ax.axvline(0, color="gray", lw=0.5)
ax.set_xlabel("dx (px)")
ax.set_ylabel("dy (px)")
ax.set_title(f"Residuals after registration, conf>={MIN_MATCH_CONFIDENCE}, "
             f"outlier-clipped\n(gray = clipped outliers)")
ax.legend(fontsize=7)
ax.set_aspect("equal", adjustable="datalim")
fig.tight_layout()
plot_path = OUT_DIR / "registration_verification_v2.png"
fig.savefig(plot_path, dpi=150)
plt.close(fig)
print(f"\nSaved: {plot_path}")
print(f"Saved: {csv_path}")