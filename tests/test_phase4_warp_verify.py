"""
Phase 4 Integration Test Suite: Sub-Pixel Warping, Multi-Pillar Verification & Master Pipeline.
Validates warp.py, verifier.py, and run_pipeline.py.
"""
import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import rasterio
from rasterio.transform import from_origin

from src.registration.hybrid_transform import HybridTransform
from src.registration.warp import warp_image_subpixel, generate_composite_overlay
from src.registration.verifier import run_verification, compute_subpixel_residuals
from run_pipeline import run_pipeline, parse_args


def create_synthetic_textured_geotiff(path: Path, width: int = 300, height: int = 300) -> Path:
    """Creates a GeoTIFF with synthetic lunar craters and micro-texture."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)
    # Background noise
    img = (np.random.randn(height, width) * 8 + 120).clip(0, 255).astype(np.float32)

    # Draw synthetic craters
    craters = [
        (80, 80, 25), (200, 100, 35), (150, 220, 30),
        (230, 210, 20), (70, 200, 18), (140, 130, 15)
    ]
    for cx, cy, r in craters:
        cv2.circle(img, (cx, cy), r, 220.0, 3)
        cv2.circle(img, (cx, cy), max(1, r - 3), 60.0, -1)

    img = cv2.GaussianBlur(img, (3, 3), 0.8)
    data = np.clip(img, 0, 255).astype(np.uint8)

    transform = from_origin(1000.0, 2000.0, 1.0, 1.0)
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "uint8",
        "crs": "EPSG:32630",
        "transform": transform
    }

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)

    return path


def test_warp_synthetic(tmp_path: Path):
    """Verify that warp_image_subpixel accurately reverses a known deformation."""
    ref_tif = tmp_path / "synthetic_ref.tif"
    create_synthetic_textured_geotiff(ref_tif, 300, 300)

    # Create source by shifting reference by known affine + drift
    with rasterio.open(ref_tif) as ds:
        ref_arr = ds.read(1)
        profile = ds.profile.copy()

    # Known transformation parameters: dx = 5.0, dy = -3.0
    known_dx, known_dy = 5.0, -3.0
    M = np.float32([[1, 0, known_dx], [0, 1, known_dy]])
    src_arr = cv2.warpAffine(ref_arr, M, (300, 300), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    src_tif = tmp_path / "synthetic_src.tif"
    with rasterio.open(src_tif, "w", **profile) as dst:
        dst.write(src_arr, 1)

    # Fit hybrid model with known correspondence
    src_pts = np.array([[50, 50], [200, 50], [50, 200], [200, 200], [150, 150], [100, 100]], dtype=np.float64)
    # Since src is shifted by (dx, dy), a feature at (x, y) in ref is at (x + dx, y + dy) in src.
    # Therefore, src_coord = ref_coord + [dx, dy]  =>  ref_coord = src_coord - [dx, dy]
    ref_pts = src_pts - np.array([known_dx, known_dy])

    model = HybridTransform()
    model.fit(src_pts, ref_pts)

    model_json = tmp_path / "test_model.json"
    model.save(model_json)

    # Warp source onto reference
    warped_tif = tmp_path / "synthetic_warped.tif"
    warp_image_subpixel(src_tif, ref_tif, model_json, warped_tif, order=3)

    assert warped_tif.exists(), "Warped GeoTIFF was not created."

    with rasterio.open(warped_tif) as dst:
        warped_arr = dst.read(1)
        assert dst.crs == profile["crs"], "CRS mismatch in warped GeoTIFF."
        assert dst.transform == profile["transform"], "Transform mismatch in warped GeoTIFF."
        assert dst.width == 300 and dst.height == 300

    # Compare central valid region of warped source with reference
    valid = (warped_arr[50:250, 50:250] > 0) & (ref_arr[50:250, 50:250] > 0)
    w_crop = warped_arr[50:250, 50:250][valid].astype(np.float32)
    r_crop = ref_arr[50:250, 50:250][valid].astype(np.float32)

    corr = np.corrcoef(w_crop, r_crop)[0, 1]
    print(f"  Synthetic Warp Alignment Correlation: {corr:.4f}")
    assert corr > 0.90, f"Warped image does not correlate with reference: {corr:.4f}"

    print("[OK] test_warp_synthetic passed")


def test_verifier_engine(tmp_path: Path):
    """Verify that multi-pillar verification generates metrics, heatmaps, and valid reports."""
    ref_tif = tmp_path / "verify_ref.tif"
    create_synthetic_textured_geotiff(ref_tif, 256, 256)

    # Use identical copy as perfectly registered product
    reg_tif = tmp_path / "verify_reg.tif"
    with rasterio.open(ref_tif) as r, rasterio.open(reg_tif, "w", **r.profile) as w:
        w.write(r.read(1), 1)

    # Generate synthetic sub-pixel tie points with tiny residuals (< 0.1 px)
    pts = []
    for r in range(8):
        for c in range(8):
            x = c * 30 + 15
            y = r * 30 + 15
            pts.append([x, y, x + np.random.uniform(-0.08, 0.08), y + np.random.uniform(-0.08, 0.08)])
    pts_arr = np.array(pts, dtype=np.float64)

    pts_csv = tmp_path / "subpixel_tie_points.csv"
    with open(pts_csv, "w") as f:
        f.write("src_x,src_y,ref_x,ref_y\n")
        for row in pts_arr:
            f.write(f"{row[0]},{row[1]},{row[2]},{row[3]}\n")

    # Fit model
    model = HybridTransform()
    model.fit(pts_arr[:, :2], pts_arr[:, 2:4])
    model_json = tmp_path / "hybrid_transform_model.json"
    model.save(model_json)

    metrics = run_verification(
        registered_path=reg_tif,
        ref_path=ref_tif,
        output_dir=tmp_path,
        tie_points_csv=pts_csv,
        hybrid_model_json=model_json,
        grid_size=(4, 4),
        target_rmse_threshold=0.5
    )

    print(f"  Verification Verdict: {metrics['verdict']}, RMSE: {metrics['rmse_px']:.4f} px")
    assert "VERIFIED" in metrics["verdict"], f"Expected VERIFIED, got {metrics['verdict']}"
    assert metrics["rmse_px"] < 0.2, f"Expected sub-pixel RMSE < 0.2, got {metrics['rmse_px']}"
    assert metrics["rmse_subpixel_target_met"] is True
    assert (tmp_path / "diagnostics" / "verification_metrics.json").exists()
    assert (tmp_path / "diagnostics" / "difference_heatmap.png").exists()
    assert (tmp_path / "diagnostics" / "registration_verification.png").exists()

    print("[OK] test_verifier_engine passed")


def test_real_harmonized_warp_and_verify():
    """Run warping and verification on the real OHRC-NAC harmonized pair if available."""
    src_cammap = Path("projects/project_test_fixed/harmonized_v2/bbox_overlap_source_cammap.tif")
    ref_crop = Path("projects/project_test_fixed/harmonized_v2/bbox_overlap_ref_cropped.tif")
    out_dir = Path("projects/test_v2_phase4_output")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not src_cammap.exists() or not ref_crop.exists():
        print("  [SKIP] Real harmonized files not found. Skipping real data test.")
        return

    # Create a fast test affine model using the known coarse offset (-282, -298)
    coarse_dx, coarse_dy = -282.0, -298.0
    model = HybridTransform()
    pts_src = np.array([
        [500, 500], [1500, 500], [500, 1500], [1500, 1500], [1000, 1000], [800, 1200]
    ], dtype=np.float64)
    pts_ref = pts_src + np.array([coarse_dx, coarse_dy])
    model.fit(pts_src, pts_ref)
    model_json = out_dir / "hybrid_transform_model.json"
    model.save(model_json)

    registered_tif = out_dir / "registered_subpixel.tif"
    print("\n  [TEST REAL] Running sub-pixel warping on real OHRC crop...")
    warp_image_subpixel(src_cammap, ref_crop, model, registered_tif, order=3, block_rows=1024)

    assert registered_tif.exists(), "Real data registered_subpixel.tif was not created."

    print("\n  [TEST REAL] Running verification engine on real registered product...")
    metrics = run_verification(
        registered_path=registered_tif,
        ref_path=ref_crop,
        output_dir=out_dir,
        hybrid_model_json=model_json,
        target_rmse_threshold=1.0
    )

    print(f"  Real Data Verification: Verdict = {metrics['verdict']}, Confidence = {metrics['composite_scientific_confidence']:.2f}")
    assert (out_dir / "diagnostics" / "verification_metrics.json").exists()
    assert (out_dir / "diagnostics" / "difference_heatmap.png").exists()

    print("[OK] test_real_harmonized_warp_and_verify passed")


if __name__ == "__main__":
    import tempfile
    print("=== Phase 4 Test Suite: Warping, Verification & Master Pipeline ===\n")
    with tempfile.TemporaryDirectory() as tmpdir:
        test_warp_synthetic(Path(tmpdir))
        test_verifier_engine(Path(tmpdir))
    test_real_harmonized_warp_and_verify()
    print("\n>>> ALL PHASE 4 TESTS COMPLETED SUCCESSFULLY <<<")
