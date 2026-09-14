"""
Automatic coarse global alignment (Stage 0).

Replaces the manual QGIS tie-point step with an automatic one, so this
pipeline works on ANY future source/reference pair (OHRC-NAC, TMC-WAC,
IIRS-WAC, ...) without a human reading coordinates off a map.

Method: read both full rasters at a heavy decimation (so the whole
image fits in one SuperGlue pass), run SuperPoint+SuperGlue ONCE on
that small pair, fit a robust similarity transform (RANSAC) from the
matches, then scale the result back up to full-resolution pixel units.
This is the same algorithm as the fine-matching stage, just applied at
a coarser scale first -- coarse-to-fine, not a different technique.

Adaptive pyramid: starts at a decimation chosen so the image's longest
side is close to TARGET_LONG_SIDE_PX. If too few confident matches are
found, decimation is halved (finer detail, bigger image) and retried,
up to MAX_PYRAMID_LEVELS times, before giving up.

Output: coarse_alignment_result.json containing OFFSET_DX, OFFSET_DY
(and rotation/scale) in FULL-RESOLUTION pixel units, ready to be fed
directly into the tiled fine-matching script (in place of the manually
typed constants used for the OHRC-NAC case).
"""

import sys
import argparse
from pathlib import Path
import json
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import cv2
import rasterio
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch

from models.matching import Matching


def parse_args():
    parser = argparse.ArgumentParser(description="Coarse Alignment (Stage 0)")
    parser.add_argument("--source_img", type=Path, required=True, help="Path to source image (e.g. OHRC)")
    parser.add_argument("--ref_img", type=Path, required=True, help="Path to reference image (e.g. NAC)")
    parser.add_argument("--output_dir", type=Path, required=True, help="Run output directory")
    return parser.parse_args()


args = parse_args()

SOURCE_PATH = args.source_img
REFERENCE_PATH = args.ref_img
OHRC_PATH = args.source_img
NAC_PATH = args.ref_img
OUTPUT_DIR = args.output_dir
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULT_JSON = OUTPUT_DIR / "coarse_alignment_result.json"

# Target longest-dimension size (px) for the coarse pass. Chosen so the
# whole image fits in one SuperGlue call comfortably on a 6GB GPU,
# regardless of the source raster's native size.
TARGET_LONG_SIDE_PX = 1600

MAX_PYRAMID_LEVELS = 5          # how many times to halve decimation and retry
MIN_MATCHES_TO_ACCEPT = 5       # below this, treat the level as a failure
                                # (similarity transform has 4 DOF; 5 inliers = 2.5× minimum)
MAX_KEYPOINTS = 4096            # generous, since only ONE pass is run per level
KEYPOINT_THRESHOLD = 0.005
NMS_RADIUS = 3
SINKHORN_ITERATIONS = 20
MATCH_THRESHOLD = 0.2
MIN_MATCH_CONFIDENCE = 0.20

# RANSAC threshold for the coarse similarity fit, in DECIMATED pixels
# (scaled up per level -- see below). A coarse pass only needs a rough
# global estimate, so this is deliberately generous.
RANSAC_THRESHOLD_DECIMATED_PX = 6
RANSAC_CONFIDENCE = 0.999
RANSAC_MAX_ITERS = 5000


# ============================================================
# SETUP
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

config = {
    "superpoint": {
        "nms_radius": NMS_RADIUS,
        "keypoint_threshold": KEYPOINT_THRESHOLD,
        "max_keypoints": MAX_KEYPOINTS,
    },
    "superglue": {
        "weights": "outdoor",
        "sinkhorn_iterations": SINKHORN_ITERATIONS,
        "match_threshold": MATCH_THRESHOLD,
    },
}
matching = Matching(config).eval().to(DEVICE)
print("SuperPoint + SuperGlue loaded.\n")


def read_decimated(path, decimation):
    with rasterio.open(path) as src:
        h, w = src.height, src.width
        out_h = max(1, round(h / decimation))
        out_w = max(1, round(w / decimation))
        data = src.read(
            1, out_shape=(1, out_h, out_w),
            resampling=rasterio.enums.Resampling.average,
        )
        # actual scale factors (out_shape rounding means these aren't
        # exactly `decimation` -- use the real ratio for accurate
        # rescaling back to full-resolution coordinates).
        scale_row = h / out_h
        scale_col = w / out_w
        return data, scale_row, scale_col, h, w


def to_tensor(image):
    t = torch.from_numpy(image.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
    return t.to(DEVICE)


def run_coarse_pass(source_img, ref_img):
    image0 = to_tensor(source_img)
    image1 = to_tensor(ref_img)
    with torch.no_grad():
        pred = matching({"image0": image0, "image1": image1})

    kp0 = pred["keypoints0"][0].detach().cpu().numpy()
    kp1 = pred["keypoints1"][0].detach().cpu().numpy()
    matches0 = pred["matches0"][0].detach().cpu().numpy()
    conf0 = pred["matching_scores0"][0].detach().cpu().numpy()

    pts0, pts1, confs = [], [], []
    for idx0 in np.where(matches0 > -1)[0]:
        idx1 = int(matches0[idx0])
        if idx1 < 0:
            continue
        confidence = float(conf0[idx0])
        if confidence < MIN_MATCH_CONFIDENCE:
            continue
        x0, y0 = kp0[idx0]
        x1, y1 = kp1[idx1]
        if source_img[int(round(y0)), int(round(x0))] == 0:
            continue
        if ref_img[int(round(y1)), int(round(x1))] == 0:
            continue
        pts0.append([x0, y0])
        pts1.append([x1, y1])
        confs.append(confidence)

    return (
        np.array(pts0, dtype=np.float64),
        np.array(pts1, dtype=np.float64),
        np.array(confs, dtype=np.float64),
        len(kp0), len(kp1),
    )


def find_valid_bbox(img):
    """Return (row_min, row_max, col_min, col_max) of the bounding box
    enclosing all non-zero pixels.  Returns None if the image is all-zero."""
    valid = img > 0
    rows_any = np.any(valid, axis=1)
    cols_any = np.any(valid, axis=0)
    if not np.any(rows_any):
        return None
    r0 = int(np.argmax(rows_any))
    r1 = int(len(rows_any) - 1 - np.argmax(rows_any[::-1]))
    c0 = int(np.argmax(cols_any))
    c1 = int(len(cols_any) - 1 - np.argmax(cols_any[::-1]))
    return r0, r1, c0, c1


# ============================================================
# ADAPTIVE PYRAMID
# ============================================================

with rasterio.open(SOURCE_PATH) as s, rasterio.open(REFERENCE_PATH) as r:
    if s.width != r.width or s.height != r.height:
        raise RuntimeError(
            "Source and reference are not on the same grid. This coarse "
            "alignment step assumes the standard preprocessing pipeline "
            "(georeferencing + resampling) has already been applied, as "
            "it has for OHRC/NAC here."
        )
    full_height, full_width = s.height, s.width

longest_side = max(full_height, full_width)
base_decimation = max(1, round(longest_side / TARGET_LONG_SIDE_PX))

result = None
decimation = base_decimation

for level in range(MAX_PYRAMID_LEVELS):
    print(f"\n--- Pyramid level {level+1}/{MAX_PYRAMID_LEVELS}: decimation={decimation} ---")

    source_raw, scale_row, scale_col, full_h, full_w = read_decimated(SOURCE_PATH, decimation)
    ref_raw, _, _, _, _ = read_decimated(REFERENCE_PATH, decimation)
    print(f"Decimated size: {source_raw.shape[0]} x {source_raw.shape[1]} "
          f"(scale_row={scale_row:.3f}, scale_col={scale_col:.3f})")

    # ---- Crop both images to their valid-data bounding boxes ----
    # Removes the massive NoData surround that confuses SuperGlue's
    # attention mechanism (OHRC is ~8% valid data, 92% black at this
    # grid).  Crop offsets are tracked and added back before scaling
    # to full-resolution coordinates.
    src_bbox = find_valid_bbox(source_raw)
    if src_bbox is None:
        print("Source has no valid data at this decimation — skipping.")
        decimation = max(1, decimation // 2)
        continue

    src_r0, src_r1, src_c0, src_c1 = src_bbox
    source_img = source_raw[src_r0:src_r1+1, src_c0:src_c1+1]
    print(f"Source crop: rows [{src_r0}..{src_r1}], cols [{src_c0}..{src_c1}] "
          f"=> {source_img.shape[0]}x{source_img.shape[1]}")

    # ---- Crop reference: guided by previous result if available ----
    if result is not None:
        # Use the coarse offset to centre the NAC crop around where
        # the matching terrain is predicted to be, with a generous
        # margin to absorb the coarse estimate's error.
        pred_shift_r = result["offset_dy"] / scale_row
        pred_shift_c = result["offset_dx"] / scale_col
        src_extent_r = src_r1 - src_r0
        src_extent_c = src_c1 - src_c0
        margin_r = max(int(src_extent_r * 0.5), 50)
        margin_c = max(int(src_extent_c * 0.5), 50)
        ref_r0 = max(0, int(src_r0 + pred_shift_r) - margin_r)
        ref_r1 = min(ref_raw.shape[0] - 1, int(src_r1 + pred_shift_r) + margin_r)
        ref_c0 = max(0, int(src_c0 + pred_shift_c) - margin_c)
        ref_c1 = min(ref_raw.shape[1] - 1, int(src_c1 + pred_shift_c) + margin_c)
        print(f"Ref  GUIDED crop: rows [{ref_r0}..{ref_r1}], cols [{ref_c0}..{ref_c1}] "
              f"=> {ref_r1-ref_r0+1}x{ref_c1-ref_c0+1}")
    else:
        # No prior estimate — use NAC's full valid-data bbox
        ref_bbox = find_valid_bbox(ref_raw)
        if ref_bbox is None:
            print("Reference has no valid data at this decimation — skipping.")
            decimation = max(1, decimation // 2)
            continue
        ref_r0, ref_r1, ref_c0, ref_c1 = ref_bbox
        print(f"Ref    crop: rows [{ref_r0}..{ref_r1}], cols [{ref_c0}..{ref_c1}] "
              f"=> {ref_r1-ref_r0+1}x{ref_c1-ref_c0+1}")

    ref_img = ref_raw[ref_r0:ref_r1+1, ref_c0:ref_c1+1]

    # Guard against GPU OOM at finer levels.
    try:
        pts0, pts1, confs, n_kp0, n_kp1 = run_coarse_pass(source_img, ref_img)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if "out of memory" in str(e).lower() or "CUDA" in str(e):
            print(f"GPU OOM at decimation={decimation} — stopping refinement.")
            torch.cuda.empty_cache()
            break
        raise
    finally:
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    print(f"SuperPoint keypoints: ({n_kp0}, {n_kp1}), accepted matches: {len(pts0)}")

    if len(pts0) < MIN_MATCHES_TO_ACCEPT:
        print(f"Too few matches ({len(pts0)} < {MIN_MATCHES_TO_ACCEPT}) at this level.")
        decimation = max(1, decimation // 2)
        continue

    # Map crop-local coords → full-decimated → full-resolution.
    # pts are (x, y) i.e. (col, row), so x uses col offset/scale.
    pts0_full = (pts0 + np.array([src_c0, src_r0])) * np.array([scale_col, scale_row])
    pts1_full = (pts1 + np.array([ref_c0, ref_r0])) * np.array([scale_col, scale_row])

    ransac_threshold_full = RANSAC_THRESHOLD_DECIMATED_PX * max(scale_row, scale_col)

    M, mask = cv2.estimateAffinePartial2D(
        pts0_full, pts1_full, method=cv2.RANSAC,
        ransacReprojThreshold=ransac_threshold_full,
        confidence=RANSAC_CONFIDENCE, maxIters=RANSAC_MAX_ITERS,
    )

    if M is None:
        print("RANSAC failed to find a model at this level.")
        decimation = max(1, decimation // 2)
        continue

    mask = mask.ravel().astype(bool)
    n_inliers = int(mask.sum())
    inlier_ratio = n_inliers / len(pts0_full)
    print(f"RANSAC: {n_inliers}/{len(pts0_full)} inliers ({inlier_ratio*100:.1f}%), "
          f"threshold={ransac_threshold_full:.1f}px (full-res)")

    if n_inliers < MIN_MATCHES_TO_ACCEPT:
        print("Too few RANSAC inliers to trust this level.")
        decimation = max(1, decimation // 2)
        continue

    src_in = pts0_full[mask]
    dst_in = pts1_full[mask]
    src_h = np.hstack([src_in, np.ones((len(src_in), 1))])
    predicted = (M @ src_h.T).T
    residuals = predicted - dst_in
    rmse_full = float(np.sqrt(np.mean(np.sum(residuals ** 2, axis=1))))

    a, b, tx = M[0]
    c, d, ty = M[1]
    scale = float(np.hypot(a, c))
    rotation_deg = float(np.degrees(np.arctan2(c, a)))

    # The similarity transform's tx/ty are at the ORIGIN (0,0).  Because
    # M also contains rotation+scale, tx/ty differ substantially from the
    # pure-translation offset that v5's tiled matching needs.  Evaluate
    # the effective displacement at the CENTRE of the source's valid data
    # — this is what a constant-offset tile reader should use.
    x_center = (src_c0 + src_c1) / 2.0 * scale_col  # col → x
    y_center = (src_r0 + src_r1) / 2.0 * scale_row  # row → y
    ref_center = M @ np.array([x_center, y_center, 1.0])
    eff_dx = float(ref_center[0] - x_center)
    eff_dy = float(ref_center[1] - y_center)

    print(f"SUCCESS at decimation={decimation}: "
          f"rotation={rotation_deg:.3f}deg, scale={scale:.4f}, "
          f"RMSE={rmse_full:.1f}px, "
          f"similarity (tx={tx:.1f}, ty={ty:.1f}), "
          f"effective offset at data centre: dx={eff_dx:.2f}, dy={eff_dy:.2f}")

    result = {
        "offset_dx": eff_dx, "offset_dy": eff_dy,
        "similarity_tx": float(tx), "similarity_ty": float(ty),
        "rotation_deg": rotation_deg, "scale": scale,
        "decimation_used": decimation,
        "n_matches": len(pts0_full), "n_inliers": n_inliers,
        "inlier_ratio": inlier_ratio,
        "rmse_coarse_full_res_px": rmse_full,
        "pyramid_level": level + 1,
        "full_height": full_height, "full_width": full_width,
        
        # Internal arrays kept for visualization at the end
        "_source_img": source_img.copy(),
        "_ref_img": ref_img.copy(),
        "_src_in": src_in,
        "_dst_in": dst_in,
        "_src_c0": src_c0, "_src_r0": src_r0,
        "_ref_c0": ref_c0, "_ref_r0": ref_r0,
        "_scale_col": scale_col, "_scale_row": scale_row,
    }

    # Don't break — continue to finer levels for refinement.
    # The next level will use this result for guided NAC cropping.
    decimation = max(1, decimation // 2)
    if decimation < 8 and result is not None:
        print(f"Reached minimum decimation — stopping refinement.")
        break

if result is None:
    raise RuntimeError(
        "Coarse alignment failed at all pyramid levels. This pair may "
        "have too little shared texture, too extreme an offset for "
        "TARGET_LONG_SIDE_PX, or a non-trivial rotation/scale outside "
        "what a single coarse pass can resolve. Consider lowering "
        "TARGET_LONG_SIDE_PX (more detail per pass) or MIN_MATCHES_TO_ACCEPT."
    )

# Extract visualization arrays BEFORE JSON dump
source_img = result.pop("_source_img")
ref_img = result.pop("_ref_img")
src_in = result.pop("_src_in")
dst_in = result.pop("_dst_in")
src_c0, src_r0 = result.pop("_src_c0"), result.pop("_src_r0")
ref_c0, ref_r0 = result.pop("_ref_c0"), result.pop("_ref_r0")
scale_col, scale_row = result.pop("_scale_col"), result.pop("_scale_row")

with open(RESULT_JSON, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2)

print(f"\nSaved: {RESULT_JSON}")
print("\nThis OFFSET_DX / OFFSET_DY (and rotation/scale, if you choose to "
      "use them) should be fed directly into the tiled fine-matching "
      "script's config, replacing any manually-typed constants.")

# ------------------------------------------------------------
# Sanity check ONLY for this known OHRC-NAC case, where we already have
# a manually-verified answer to compare against. This comparison won't
# exist for a genuinely new, unverified dataset -- it's here purely to
# validate that the automatic method agrees with the trusted manual
# result before relying on it for anything new.
#
# The manual offset was measured as a PURE TRANSLATION near the centre
# of the overlapping region.  Our offset_dx/offset_dy is also evaluated
# at the centre of the source data, so the comparison is fair.
# ------------------------------------------------------------
KNOWN_MANUAL_OFFSET = {"dx": -127.75, "dy": -2488.45}
err_dx = result['offset_dx'] - KNOWN_MANUAL_OFFSET['dx']
err_dy = result['offset_dy'] - KNOWN_MANUAL_OFFSET['dy']
err_dist = np.hypot(err_dx, err_dy)
print(f"\n[Sanity check for THIS dataset only]")
print(f"  Manual tie-point offset : dx={KNOWN_MANUAL_OFFSET['dx']:.2f}, "
      f"dy={KNOWN_MANUAL_OFFSET['dy']:.2f}")
print(f"  Automatic (at centre)   : dx={result['offset_dx']:.2f}, "
      f"dy={result['offset_dy']:.2f}")
print(f"  Difference              : dx={err_dx:.2f}px, dy={err_dy:.2f}px "
      f"(Euclidean={err_dist:.1f}px)")
if err_dist < 250:
    print(f"  => WITHIN v5 search margin (250px). Fine matching should work.")
else:
    print(f"  => WARNING: exceeds v5 search margin (250px). May need to increase "
          f"SEARCH_MARGIN_PX.")

# ============================================================
# VISUALIZATION: Coarse Alignment Match Plot
# ============================================================
print("\nGenerating coarse alignment visualization...")

# Convert inlier full-res coordinates back to the local crop pixel space
src_local = (src_in / np.array([scale_col, scale_row])) - np.array([src_c0, src_r0])
dst_local = (dst_in / np.array([scale_col, scale_row])) - np.array([ref_c0, ref_r0])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
ax1.imshow(source_img, cmap='gray')
ax1.set_title(f"Source (decimated x{result['decimation_used']})")
ax1.axis('off')

ax2.imshow(ref_img, cmap='gray')
ax2.set_title(f"Reference (decimated x{result['decimation_used']})")
ax2.axis('off')

# Draw matches
max_lines = 150 # cap to avoid a messy plot
indices = np.linspace(0, len(src_local)-1, min(max_lines, len(src_local)), dtype=int)
for i in indices:
    xy1 = (src_local[i, 0], src_local[i, 1])
    xy2 = (dst_local[i, 0], dst_local[i, 1])
    con = ConnectionPatch(xyA=xy2, xyB=xy1, coordsA="data", coordsB="data",
                          axesA=ax2, axesB=ax1, color="lime", lw=0.5, alpha=0.6)
    ax2.add_artist(con)

# Draw points
ax1.scatter(src_local[indices, 0], src_local[indices, 1], s=4, c="lime", edgecolors="none")
ax2.scatter(dst_local[indices, 0], dst_local[indices, 1], s=4, c="lime", edgecolors="none")

# Add text box with metrics
textstr = '\n'.join((
    "Coarse Alignment Metrics",
    "------------------------",
    f"Offset DX: {result['offset_dx']:.1f} px",
    f"Offset DY: {result['offset_dy']:.1f} px",
    f"Rotation : {result['rotation_deg']:.2f} deg",
    f"Scale    : {result['scale']:.4f}",
    f"RMSE     : {result['rmse_coarse_full_res_px']:.1f} px",
    f"Inliers  : {result['n_inliers']} ({result['inlier_ratio']*100:.1f}%)"
))
props = dict(boxstyle='round', facecolor='black', alpha=0.7, edgecolor='lime')
fig.text(0.5, 0.05, textstr, fontsize=12, color='white', 
         verticalalignment='bottom', horizontalalignment='center', bbox=props, family='monospace')

plt.tight_layout()
diag_dir = OUTPUT_DIR / "diagnostics"
diag_dir.mkdir(parents=True, exist_ok=True)
plot_path = diag_dir / "coarse_alignment_visual.png"
plt.savefig(plot_path, dpi=150, facecolor='black', edgecolor='none')
plt.close(fig)
print(f"Saved: {plot_path}")