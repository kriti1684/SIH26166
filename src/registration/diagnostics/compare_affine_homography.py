

from pathlib import Path
import json

import numpy as np
import pandas as pd
import cv2


MATCH_DIR = Path(r"C:\#Padhai\E\SIH1\SuperGluePretrainedNetwork\lunar_matches_v5")
CORE_CSV = MATCH_DIR / "ohrc_nac_superglue_matches_core_only.csv"
DIAG_DIR = MATCH_DIR / "diagnostics"
DIAG_DIR.mkdir(parents=True, exist_ok=True)

MIN_CONFIDENCE = 0.30
RANSAC_THRESHOLDS_PX = [5, 10, 15, 20, 30]
CHOSEN_THRESHOLD_PX = 15
RANSAC_CONFIDENCE = 0.999
RANSAC_MAX_ITERS = 5000
PIXEL_SIZE_M = 1.1523692688652

CLUSTERS = {
    "top_cluster": (6800, 9000),
    "middle_cluster": (9200, 14400),
    "lower_cluster": (19400, 24000),
}


# ============================================================
# FIT FUNCTIONS
# ============================================================

def fit_affine(src, dst, threshold_px):
    M, mask = cv2.estimateAffine2D(
        src, dst, method=cv2.RANSAC,
        ransacReprojThreshold=threshold_px,
        confidence=RANSAC_CONFIDENCE, maxIters=RANSAC_MAX_ITERS,
    )
    if M is None:
        return None
    mask = mask.ravel().astype(bool)
    src_in, dst_in = src[mask], dst[mask]
    src_h = np.hstack([src_in, np.ones((len(src_in), 1))])
    predicted = (M @ src_h.T).T
    return summarize(mask, len(src), predicted, dst_in, M, "affine")


def fit_homography(src, dst, threshold_px):
    M, mask = cv2.findHomography(
        src, dst, method=cv2.RANSAC,
        ransacReprojThreshold=threshold_px,
        confidence=RANSAC_CONFIDENCE, maxIters=RANSAC_MAX_ITERS,
    )
    if M is None:
        return None
    mask = mask.ravel().astype(bool)
    src_in, dst_in = src[mask], dst[mask]
    predicted = cv2.perspectiveTransform(
        src_in.reshape(-1, 1, 2), M
    ).reshape(-1, 2)
    return summarize(mask, len(src), predicted, dst_in, M, "homography")


def summarize(mask, n_total, predicted, dst_in, M, model_name):
    residuals = predicted - dst_in
    rmse_x = float(np.sqrt(np.mean(residuals[:, 0] ** 2)))
    rmse_y = float(np.sqrt(np.mean(residuals[:, 1] ** 2)))
    rmse_combined = float(np.sqrt(np.mean(np.sum(residuals ** 2, axis=1))))
    n_inliers = int(mask.sum())
    return {
        "model": model_name, "M": M, "mask": mask,
        "n_inliers": n_inliers, "n_total": n_total,
        "inlier_ratio": n_inliers / n_total,
        "rmse_x_px": rmse_x, "rmse_y_px": rmse_y,
        "rmse_combined_px": rmse_combined,
        "rmse_x_m": rmse_x * PIXEL_SIZE_M, "rmse_y_m": rmse_y * PIXEL_SIZE_M,
        "rmse_combined_m": rmse_combined * PIXEL_SIZE_M,
    }


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(CORE_CSV)
df = df[df["confidence"] >= MIN_CONFIDENCE].reset_index(drop=True)
print(f"Total matches after confidence >= {MIN_CONFIDENCE}: {len(df)}\n")

src_all = df[["ohrc_x", "ohrc_y"]].to_numpy(dtype=np.float64)
dst_all = df[["nac_x", "nac_y"]].to_numpy(dtype=np.float64)


# ============================================================
# GLOBAL THRESHOLD SWEEP: AFFINE VS HOMOGRAPHY
# ============================================================

print("=" * 100)
print("GLOBAL FIT: AFFINE vs HOMOGRAPHY, threshold sweep")
print("=" * 100)
print(f"{'thresh':>7s} {'model':>10s} {'inliers':>8s} {'ratio':>7s} "
      f"{'rmse_x':>8s} {'rmse_y':>8s} {'rmse_comb':>10s}")

global_results = {"affine": {}, "homography": {}}
for t in RANSAC_THRESHOLDS_PX:
    ra = fit_affine(src_all, dst_all, t)
    rh = fit_homography(src_all, dst_all, t)
    if ra:
        global_results["affine"][t] = ra
        print(f"{t:>7d} {'affine':>10s} {ra['n_inliers']:>8d} "
              f"{ra['inlier_ratio']:>7.3f} {ra['rmse_x_px']:>8.2f} "
              f"{ra['rmse_y_px']:>8.2f} {ra['rmse_combined_px']:>10.2f}")
    if rh:
        global_results["homography"][t] = rh
        print(f"{t:>7d} {'homography':>10s} {rh['n_inliers']:>8d} "
              f"{rh['inlier_ratio']:>7.3f} {rh['rmse_x_px']:>8.2f} "
              f"{rh['rmse_y_px']:>8.2f} {rh['rmse_combined_px']:>10.2f}")


# ============================================================
# PER-CLUSTER FIT: AFFINE VS HOMOGRAPHY
# ============================================================

print("\n" + "=" * 100)
print(f"PER-CLUSTER FIT at threshold={CHOSEN_THRESHOLD_PX}px: does homography rescue the top cluster?")
print("=" * 100)
print(f"{'cluster':>15s} {'model':>10s} {'n_pts':>6s} {'inliers':>8s} "
      f"{'ratio':>7s} {'rmse_x':>8s} {'rmse_y':>8s} {'rmse_comb':>10s}")

cluster_results = {}
for name, (row_min, row_max) in CLUSTERS.items():
    subset = df[(df["ohrc_y"] >= row_min) & (df["ohrc_y"] < row_max)]
    if len(subset) < 4:
        print(f"{name:>15s}  too few points ({len(subset)})")
        continue
    src_c = subset[["ohrc_x", "ohrc_y"]].to_numpy(dtype=np.float64)
    dst_c = subset[["nac_x", "nac_y"]].to_numpy(dtype=np.float64)

    ra = fit_affine(src_c, dst_c, CHOSEN_THRESHOLD_PX)
    rh = fit_homography(src_c, dst_c, CHOSEN_THRESHOLD_PX)
    cluster_results[name] = {"affine": ra, "homography": rh}

    if ra:
        print(f"{name:>15s} {'affine':>10s} {len(subset):>6d} "
              f"{ra['n_inliers']:>8d} {ra['inlier_ratio']:>7.3f} "
              f"{ra['rmse_x_px']:>8.2f} {ra['rmse_y_px']:>8.2f} "
              f"{ra['rmse_combined_px']:>10.2f}")
    if rh:
        print(f"{name:>15s} {'homography':>10s} {len(subset):>6d} "
              f"{rh['n_inliers']:>8d} {rh['inlier_ratio']:>7.3f} "
              f"{rh['rmse_x_px']:>8.2f} {rh['rmse_y_px']:>8.2f} "
              f"{rh['rmse_combined_px']:>10.2f}")


# ============================================================
# HOW MANY OF THE GLOBALLY-REJECTED TOP-CLUSTER POINTS DOES
# THE GLOBAL HOMOGRAPHY NOW ACCEPT? (direct before/after check)
# ============================================================

if CHOSEN_THRESHOLD_PX in global_results["affine"] and CHOSEN_THRESHOLD_PX in global_results["homography"]:
    top_row_min, top_row_max = CLUSTERS["top_cluster"]
    top_mask_bool = ((df["ohrc_y"] >= top_row_min) & (df["ohrc_y"] < top_row_max)).to_numpy()
    n_top = int(top_mask_bool.sum())

    affine_global_mask = global_results["affine"][CHOSEN_THRESHOLD_PX]["mask"]
    homography_global_mask = global_results["homography"][CHOSEN_THRESHOLD_PX]["mask"]

    n_top_affine_inliers = int((affine_global_mask & top_mask_bool).sum())
    n_top_homography_inliers = int((homography_global_mask & top_mask_bool).sum())

    print("\n" + "=" * 100)
    print(f"TOP CLUSTER ({n_top} points) -- accepted by GLOBAL fit, before vs after:")
    print(f"  Global AFFINE     accepts: {n_top_affine_inliers} / {n_top} "
          f"({n_top_affine_inliers/n_top*100:.1f}%)")
    print(f"  Global HOMOGRAPHY accepts: {n_top_homography_inliers} / {n_top} "
          f"({n_top_homography_inliers/n_top*100:.1f}%)")
    print("=" * 100)


# ============================================================
# SAVE CHOSEN HOMOGRAPHY TRANSFORM + INLIERS
# ============================================================

if CHOSEN_THRESHOLD_PX in global_results["homography"]:
    rh = global_results["homography"][CHOSEN_THRESHOLD_PX]
    mask = rh["mask"]
    inliers_df = df[mask].copy()
    inliers_csv = MATCH_DIR / f"homography_inliers_thresh{CHOSEN_THRESHOLD_PX}px.csv"
    inliers_df.to_csv(inliers_csv, index=False)

    def strip(r):
        if r is None:
            return None
        return {k: v for k, v in r.items() if k not in ("M", "mask")}

    summary = {
        "chosen_threshold_px": CHOSEN_THRESHOLD_PX,
        "min_confidence_filter": MIN_CONFIDENCE,
        "homography_matrix": rh["M"].tolist(),
        "n_inliers": rh["n_inliers"], "n_total_input": rh["n_total"],
        "inlier_ratio": rh["inlier_ratio"],
        "rmse_x_px": rh["rmse_x_px"], "rmse_y_px": rh["rmse_y_px"],
        "rmse_combined_px": rh["rmse_combined_px"],
        "rmse_combined_m": rh["rmse_combined_m"],
        "global_sweep": {
            str(t): {
                "affine": strip(global_results["affine"].get(t)),
                "homography": strip(global_results["homography"].get(t)),
            } for t in RANSAC_THRESHOLDS_PX
        },
    }
    with open(MATCH_DIR / "homography_transform.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved: {inliers_csv}")
    print(f"Saved: {MATCH_DIR / 'homography_transform.json'}")