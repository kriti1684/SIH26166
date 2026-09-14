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
import argparse
import json

try:
    from src.models.matching import Matching
except ImportError:
    from ..models.matching import Matching


def parse_args():
    parser = argparse.ArgumentParser(description="Registration Verification & Quality Control (Stage 4)")
    parser.add_argument("--registered_img", type=Path, required=True, help="Path to registered output image")
    parser.add_argument("--ref_img", type=Path, required=True, help="Path to reference image (e.g. NAC)")
    parser.add_argument("--output_dir", type=Path, required=True, help="Run output directory")
    parser.add_argument("--config", type=Path, default=None, help="Optional config JSON")
    return parser.parse_args()


def run_verification(registered_path: Path, ref_path: Path, output_dir: Path, custom_config: dict = None):
    output_dir = Path(output_dir)
    diag_dir = output_dir / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    tile_size = 1600
    min_match_confidence = 0.50
    nodata_margin_px = 2
    clip_rounds = 3
    clip_threshold_px = 25.0

    if custom_config and "verifier" in custom_config:
        cfg = custom_config["verifier"]
        tile_size = cfg.get("tile_size", tile_size)
        min_match_confidence = cfg.get("min_match_confidence", min_match_confidence)
        clip_rounds = cfg.get("clip_rounds", clip_rounds)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[STAGE 4] Device: {device}")

    model_config = {
        "superpoint": {"nms_radius": 3, "keypoint_threshold": 0.005, "max_keypoints": 2048},
        "superglue": {"weights": "outdoor", "sinkhorn_iterations": 20, "match_threshold": 0.2},
    }
    matching = Matching(model_config).eval().to(device)

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
        return torch.from_numpy(image.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)

    def is_near_nodata(image, ix, iy, margin):
        h, w = image.shape
        y0, y1 = iy - margin, iy + margin + 1
        x0, x1 = ix - margin, ix + margin + 1
        if y0 < 0 or x0 < 0 or y1 > h or x1 > w:
            return True
        return bool(np.any(image[y0:y1, x0:x1] == 0))

    with rasterio.open(registered_path) as src_r:
        h = src_r.height

    test_locations = [
        ("top", int(h * 0.25), 1800),
        ("upper-middle", int(h * 0.40), 1800),
        ("middle", int(h * 0.50), 1800),
        ("lower", int(h * 0.75), 1800),
        ("bottom", int(h * 0.85), 1800),
    ]

    all_records = []
    with rasterio.open(registered_path) as ohrc_reg, rasterio.open(ref_path) as nac:
        for label, row, col in test_locations:
            ohrc_tile = read_window(ohrc_reg, row, col, tile_size)
            nac_tile = read_window(nac, row, col, tile_size)

            if ohrc_tile is None or nac_tile is None:
                continue

            ohrc_valid = np.count_nonzero(ohrc_tile) / ohrc_tile.size
            nac_valid = np.count_nonzero(nac_tile) / nac_tile.size
            if ohrc_valid < 0.05 or nac_valid < 0.05:
                continue

            with torch.no_grad():
                pred = matching({"image0": to_tensor(ohrc_tile), "image1": to_tensor(nac_tile)})

            kp0 = pred["keypoints0"][0].detach().cpu().numpy()
            kp1 = pred["keypoints1"][0].detach().cpu().numpy()
            matches0 = pred["matches0"][0].detach().cpu().numpy()
            conf0 = pred["matching_scores0"][0].detach().cpu().numpy()

            valid_idx = np.where(matches0 > -1)[0]
            n_acc = 0
            for idx0 in valid_idx:
                idx1 = int(matches0[idx0])
                if idx1 < 0: continue
                conf = float(conf0[idx0])
                if conf < min_match_confidence: continue

                x0, y0 = kp0[idx0]
                x1, y1 = kp1[idx1]
                ix0, iy0 = int(round(x0)), int(round(y0))
                ix1, iy1 = int(round(x1)), int(round(y1))

                if not (0 <= ix0 < tile_size and 0 <= iy0 < tile_size): continue
                if not (0 <= ix1 < tile_size and 0 <= iy1 < tile_size): continue
                if is_near_nodata(ohrc_tile, ix0, iy0, nodata_margin_px): continue
                if is_near_nodata(nac_tile, ix1, iy1, nodata_margin_px): continue

                dx = float(x1 - x0)
                dy = float(y1 - y0)
                all_records.append({
                    "location": label, "row": row, "col": col,
                    "dx": dx, "dy": dy, "confidence": conf,
                })
                n_acc += 1
            print(f"Location '{label}' ({row}, {col}): Accepted verification matches = {n_acc}")

    if not all_records:
        print("[WARNING] No verification matches found.")
        return None

    df = pd.DataFrame(all_records)
    keep = np.ones(len(df), dtype=bool)
    for _ in range(clip_rounds):
        med_dx = df.loc[keep, "dx"].median()
        med_dy = df.loc[keep, "dy"].median()
        dist = np.hypot(df["dx"] - med_dx, df["dy"] - med_dy)
        keep = (dist <= clip_threshold_px).to_numpy()

    df["kept"] = keep
    csv_path = diag_dir / "registration_verification_residuals.csv"
    df.to_csv(csv_path, index=False)

    clean = df[df["kept"]]
    med_dx = float(clean["dx"].median())
    med_dy = float(clean["dy"].median())
    mad_dx = float(np.median(np.abs(clean["dx"] - med_dx)))
    mad_dy = float(np.median(np.abs(clean["dy"] - med_dy)))

    print(f"\n[VERIFICATION RESULTS]")
    print(f"Residual Median: dx = {med_dx:.2f} px (MAD={mad_dx:.2f}), dy = {med_dy:.2f} px (MAD={mad_dy:.2f})")

    fig, ax = plt.subplots(figsize=(6, 6))
    for label in df["location"].unique():
        sub_k = df[(df["location"] == label) & (df["kept"])]
        ax.scatter(sub_k["dx"], sub_k["dy"], s=15, alpha=0.7, label=f"{label}")
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)
    ax.set_xlabel("dx (px)")
    ax.set_ylabel("dy (px)")
    ax.set_title("Residual Displacements After Registration (px)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    plot_path = diag_dir / "registration_verification.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    return csv_path


def main():
    args = parse_args()
    custom_cfg = None
    if args.config and args.config.exists():
        with open(args.config, "r", encoding="utf-8") as f:
            custom_cfg = json.load(f)
    run_verification(args.registered_img, args.ref_img, args.output_dir, custom_cfg)


if __name__ == "__main__":
    main()
