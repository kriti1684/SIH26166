

from pathlib import Path

import cv2
import numpy as np
import pandas as pd


MATCH_DIR = Path(r"C:\#Padhai\E\SIH1\SuperGluePretrainedNetwork\lunar_matches_v5")
TILE_DIR = MATCH_DIR / "tile_pngs"
MATCHES_CSV = MATCH_DIR / "ohrc_nac_superglue_matches.csv"

OUT_DIR = MATCH_DIR / "diagnostics" / "clean_tile_views"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Which tiles to export. Add/remove tile IDs as needed.
TILE_IDS = [13, 7, 16, 19, 22]

# How many matched points to number/mark per tile (evenly sampled, not
# just the first N, so you see spread across the tile).
MAX_MARKERS_PER_TILE = 15


def load_tile_images(tile_id):
    ohrc_path = TILE_DIR / f"tile_{tile_id:04d}_ohrc.png"
    nac_path = TILE_DIR / f"tile_{tile_id:04d}_nac.png"
    if not ohrc_path.exists() or not nac_path.exists():
        print(f"Tile {tile_id}: PNGs not found ({ohrc_path.name}), skipping.")
        return None, None
    ohrc = cv2.imread(str(ohrc_path), cv2.IMREAD_GRAYSCALE)
    nac = cv2.imread(str(nac_path), cv2.IMREAD_GRAYSCALE)
    return ohrc, nac


def stitch_clean(ohrc, nac, gap=20):
    h = max(ohrc.shape[0], nac.shape[0])
    w1, w2 = ohrc.shape[1], nac.shape[1]
    canvas = np.full((h, w1 + gap + w2, 3), 40, dtype=np.uint8)
    canvas[:ohrc.shape[0], :w1] = cv2.cvtColor(ohrc, cv2.COLOR_GRAY2BGR)
    canvas[:nac.shape[0], w1 + gap:w1 + gap + w2] = cv2.cvtColor(nac, cv2.COLOR_GRAY2BGR)
    cv2.putText(canvas, "OHRC", (20, 35), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "NAC", (w1 + gap + 20, 35), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas, w1, gap


def add_numbered_markers(canvas, w1, gap, points_ohrc, points_nac):
    """points_ohrc/points_nac: list of (x, y) local-tile pixel coords,
    same order/index so point i on each side share the same number."""
    for i, ((x0, y0), (x1, y1)) in enumerate(zip(points_ohrc, points_nac)):
        p0 = (int(round(x0)), int(round(y0)))
        p1 = (int(round(x1)) + w1 + gap, int(round(y1)))
        color = (0, 220, 0)
        for p in (p0, p1):
            cv2.circle(canvas, p, 5, color, -1)
            cv2.circle(canvas, p, 6, (0, 0, 0), 1)
        # number label, offset slightly so it doesn't sit on the dot
        cv2.putText(canvas, str(i), (p0[0] + 8, p0[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, str(i), (p1[0] + 8, p1[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    return canvas


matches_df = None
if MATCHES_CSV.exists():
    matches_df = pd.read_csv(MATCHES_CSV)

for tile_id in TILE_IDS:
    ohrc, nac = load_tile_images(tile_id)
    if ohrc is None:
        continue

    # Version 1: fully clean, no markers at all.
    clean_canvas, w1, gap = stitch_clean(ohrc, nac)
    clean_path = OUT_DIR / f"tile_{tile_id:04d}_clean.png"
    cv2.imwrite(str(clean_path), clean_canvas)
    print(f"Saved: {clean_path}")

    # Version 2: same, but with a handful of numbered dot pairs (no lines).
    if matches_df is not None:
        tile_matches = matches_df[matches_df["tile_id"] == tile_id]
        if len(tile_matches) > 0:
            # Even sampling across the available matches, not just the top.
            n = min(MAX_MARKERS_PER_TILE, len(tile_matches))
            idx = np.linspace(0, len(tile_matches) - 1, n).astype(int)
            sampled = tile_matches.iloc[idx]

            tile_row = int(sampled["tile_row"].iloc[0])
            tile_col = int(sampled["tile_col"].iloc[0])

            # Convert full-image coords back to LOCAL tile coords for
            # drawing. OHRC local = full - (tile_col, tile_row).
            points_ohrc = list(zip(
                sampled["ohrc_x"] - tile_col, sampled["ohrc_y"] - tile_row
            ))

            # NAC local coords: we don't have the NAC window origin
            # stored directly in this CSV, but we can recover it since
            # nac_x/nac_y are already full-image NAC coords and the tile
            # PNG was saved starting at the NAC window's own origin. If
            # you used the v5 script, the NAC window top-left equals
            # (tile_col + OFFSET_DX - MARGIN, tile_row + OFFSET_DY - MARGIN).
            OFFSET_DX, OFFSET_DY, MARGIN = -127.75, -2488.45, 250
            nac_origin_x = round(tile_col + OFFSET_DX) - MARGIN
            nac_origin_y = round(tile_row + OFFSET_DY) - MARGIN
            points_nac = list(zip(
                sampled["nac_x"] - nac_origin_x, sampled["nac_y"] - nac_origin_y
            ))

            marked_canvas, _, _ = stitch_clean(ohrc, nac)
            marked_canvas = add_numbered_markers(
                marked_canvas, w1, gap, points_ohrc, points_nac
            )
            marked_path = OUT_DIR / f"tile_{tile_id:04d}_numbered.png"
            cv2.imwrite(str(marked_path), marked_canvas)
            print(f"Saved: {marked_path} ({n} numbered pairs)")
        else:
            print(f"Tile {tile_id}: no matches in CSV to mark.")

print(f"\nAll outputs in: {OUT_DIR}")
print("Open the _clean.png files first to just look at the terrain side by")
print("side. Open the _numbered.png files to check specific point pairs --")
print("same number on OHRC and NAC sides should be the SAME real feature.")