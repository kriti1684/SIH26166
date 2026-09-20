import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import rasterio
from rasterio.transform import from_origin

from src.preprocessing.bounding_overlap import compute_geographic_overlap
from src.preprocessing.band_selector import (
    get_iirs_band_wavelength,
    compute_band_entropy,
    select_best_band_for_reference,
    extract_or_synthesize_band
)


def test_iirs_band_wavelength_math():
    w0 = get_iirs_band_wavelength(0)
    assert w0 == 800.0
    w10 = get_iirs_band_wavelength(10)
    assert w10 == 800.0 + 10 * 16.5


def test_compute_band_entropy():
    # Uniform constant image should have low entropy
    flat = np.ones((50, 50), dtype=np.float32) * 100.0
    ent_flat = compute_band_entropy(flat)
    assert ent_flat == 0.0

    # Textured image with noise should have higher entropy
    np.random.seed(42)
    noise = np.random.uniform(10, 200, (50, 50)).astype(np.float32)
    ent_noise = compute_band_entropy(noise)
    assert ent_noise > 3.0


def test_iirs_synthetic_cube_selection(tmp_path):
    # Create a synthetic 10-band IIRS cube
    cube_path = tmp_path / "synthetic_iirs.tif"
    profile = {
        "driver": "GTiff",
        "height": 100,
        "width": 100,
        "count": 10,
        "dtype": "float32",
        "crs": "EPSG:4326",
        "transform": from_origin(0, 0, 1, 1)
    }

    with rasterio.open(cube_path, "w", **profile) as dst:
        for b in range(1, 11):
            # Make band 3 have highest contrast and variance
            if b == 3:
                data = np.random.uniform(50, 250, (100, 100)).astype(np.float32)
            else:
                data = np.ones((100, 100), dtype=np.float32) * (b * 10.0)
            dst.write(data, b)

    result = select_best_band_for_reference(cube_path, reference_sensor="WAC", max_swir_band=10)
    assert result["selected_band"] == 3
    assert result["entropy"] > 0

    # Test extraction
    out_band = tmp_path / "extracted_band.tif"
    extract_or_synthesize_band(cube_path, out_band, selected_band=result["selected_band"])
    assert out_band.exists()
    with rasterio.open(out_band) as src:
        assert src.count == 1
        assert src.shape == (100, 100)


def test_bounding_overlap_on_real_data():
    source_p = Path("projects/project_test_fixed/normalized/ch2_ohr_ncp_20210331T2033243734_d_img_d18_normalized.tif")
    ref_p = Path("projects/project_test_fixed/normalized/M1179837753RE_normalized.tif")

    if not source_p.exists() or not ref_p.exists():
        print("Test raster files not found, skipping real data check.")
        return

    res = compute_geographic_overlap(source_p, ref_p)
    assert res["has_overlap"] is True
    assert res["src_overlap_pct"] > 0
    assert res["ref_overlap_pct"] > 0
    assert res["src_window"].width > 0
    assert res["ref_window"].width > 0
    print("Bounding overlap test passed!")

if __name__ == "__main__":
    from tempfile import TemporaryDirectory
    print("Running Phase 1 tests...")
    test_iirs_band_wavelength_math()
    print("[OK] test_iirs_band_wavelength_math passed")
    test_compute_band_entropy()
    print("[OK] test_compute_band_entropy passed")
    with TemporaryDirectory() as tmp_dir:
        test_iirs_synthetic_cube_selection(Path(tmp_dir))
    print("[OK] test_iirs_synthetic_cube_selection passed")
    test_bounding_overlap_on_real_data()
    print("[OK] test_bounding_overlap_on_real_data passed")
    print("\n>>> ALL PHASE 1 TESTS PASSED SUCCESSFULLY! <<<")
