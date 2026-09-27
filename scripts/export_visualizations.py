from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window


def draw_matching_tile(
    src_img: np.ndarray,
    ref_img: np.ndarray,
    src_pts: np.ndarray,
    ref_pts: np.ndarray,
    save_path: Path,
    tile_id: int,
    margin: int = 20
):
    """
    Renders high-res side-by-side correspondence plot with matching lines.
    """
    def norm_u8(arr):
        v = arr[arr > 0]
        if len(v) < 50:
            return np.clip(arr, 0, 255).astype(np.uint8)
        p2, p98 = np.percentile(v, (2, 98))
        return np.clip((arr - p2) / (p98 - p2 + 1e-6) * 255.0, 0, 255).astype(np.uint8)

    im0 = norm_u8(src_img)
    im1 = norm_u8(ref_img)

    im0_bgr = cv2.cvtColor(im0, cv2.COLOR_GRAY2BGR)
    im1_bgr = cv2.cvtColor(im1, cv2.COLOR_GRAY2BGR)

    H0, W0 = im0.shape
    H1, W1 = im1.shape
    H, W = max(H0, H1), W0 + W1 + margin

    canvas = np.full((H, W, 3), 35, dtype=np.uint8)
    canvas[:H0, :W0] = im0_bgr
    canvas[:H1, W0 + margin:W0 + margin + W1] = im1_bgr

    for i, ((x0, y0), (x1, y1)) in enumerate(zip(src_pts, ref_pts)):
        p0 = (int(round(x0)), int(round(y0)))
        p1 = (int(round(x1)) + W0 + margin, int(round(y1)))
        
        color = (0, 255, 128)
        cv2.line(canvas, p0, p1, color, 2, lineType=cv2.LINE_AA)
        cv2.circle(canvas, p0, 4, (0, 0, 255), -1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, p1, 4, (0, 0, 255), -1, lineType=cv2.LINE_AA)
        
        cv2.putText(canvas, str(i + 1), (p0[0] + 6, p0[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, str(i + 1), (p1[0] + 6, p1[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(canvas, f"OHRC Tile #{tile_id:04d}", (25, 45), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(canvas, f"OHRC Tile #{tile_id:04d}", (25, 45), cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, "NAC Reference Tile", (W0 + margin + 25, 45), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(canvas, "NAC Reference Tile", (W0 + margin + 25, 45), cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 1, cv2.LINE_AA)
    
    cv2.putText(canvas, f"LoFTR Matches: {len(src_pts)} verified tie points", (25, H - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2, cv2.LINE_AA)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), canvas)
    print(f"  [SAVED] {save_path.name} ({len(src_pts)} points)")


def main():
    run_dir = Path("projects/run_v2")
    src_tif = run_dir / "harmonized" / "bbox_overlap_source_cammap.tif"
    ref_tif = run_dir / "harmonized" / "bbox_overlap_ref_cropped.tif"
    csv_path = run_dir / "candidate_matches.csv"

    if not csv_path.exists():
        print("No candidate_matches.csv found.")
        return

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} candidate matches.")

    viz_dir = run_dir / "match_visualizations"
    tile_dir = run_dir / "tile_pngs"
    viz_dir.mkdir(parents=True, exist_ok=True)
    tile_dir.mkdir(parents=True, exist_ok=True)

    tile_size = 1600
    margin = 250

    with rasterio.open(src_tif) as src_ds, rasterio.open(ref_tif) as ref_ds:
        w_src, h_src = src_ds.width, src_ds.height
        w_ref, h_ref = ref_ds.width, ref_ds.height

        df["grid_x"] = (df["src_x"] // 1200).astype(int)
        df["grid_y"] = (df["src_y"] // 1200).astype(int)

        groups = df.groupby(["grid_x", "grid_y"])
        tile_idx = 1

        for (gx, gy), group in groups:
            x0 = max(0, int(gx * 1200))
            y0 = max(0, int(gy * 1200))
            x1 = min(w_src, x0 + tile_size)
            y1 = min(h_src, y0 + tile_size)

            win_src = Window(col_off=x0, row_off=y0, width=x1 - x0, height=y1 - y0)
            src_crop = src_ds.read(1, window=win_src).astype(np.float32)

            ref_pts_x = group["ref_x"].values
            ref_pts_y = group["ref_y"].values

            rx0 = max(0, int(np.min(ref_pts_x) - margin))
            ry0 = max(0, int(np.min(ref_pts_y) - margin))
            rx1 = min(w_ref, int(np.max(ref_pts_x) + margin))
            ry1 = min(h_ref, int(np.max(ref_pts_y) + margin))

            win_ref = Window(col_off=rx0, row_off=ry0, width=rx1 - rx0, height=ry1 - ry0)
            ref_crop = ref_ds.read(1, window=win_ref).astype(np.float32)

            local_src = np.column_stack([group["src_x"].values - x0, group["src_y"].values - y0])
            local_ref = np.column_stack([group["ref_x"].values - rx0, group["ref_y"].values - ry0])

            cv2.imwrite(str(tile_dir / f"tile_{tile_idx:04d}_ohrc.png"), np.clip(src_crop, 0, 255).astype(np.uint8))
            cv2.imwrite(str(tile_dir / f"tile_{tile_idx:04d}_nac.png"), np.clip(ref_crop, 0, 255).astype(np.uint8))

            match_out = viz_dir / f"tile_{tile_idx:04d}_matches.png"
            draw_matching_tile(
                src_img=src_crop,
                ref_img=ref_crop,
                src_pts=local_src,
                ref_pts=local_ref,
                save_path=match_out,
                tile_id=tile_idx
            )
            tile_idx += 1

    print(f"\n[DONE] Generated {tile_idx - 1} match visualizations in {viz_dir}")


if __name__ == "__main__":
    main()
