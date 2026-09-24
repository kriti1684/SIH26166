"""
main.py
=======
ChandaShakti: Universal Sub-Pixel Multi-Modal Lunar Image Co-Registration Engine
ISRO Smart India Hackathon (SIH 2024) — Problem Statement SIH26166.

Features:
  - Phase 1: Robust scale harmonization, SPICE ray-tracing & structural enhancement
  - Phase 2: Dual-method coarse alignment (Structural FFT + Crater Rim Consensus Voting)
  - Phase 3: Dense LoFTR attention matching + Sub-Pixel continuous Gauss-Newton ECC
  - Phase 4: 3-Layer physics-grounded hybrid transformation (Affine + Drift + TPS)
  - Phase 5: Streaming block-wise bicubic warping & multi-pillar scientific verification

Usage Example:
  python main.py --source <path> --reference <path> \
                 --sensor_src <OHRC|IIRS|TMC> --sensor_ref <NAC|WAC|SELENE> \
                 --out_dir <path>
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import sys
import time
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import rasterio

# Import src components
from src.preprocessing.ingest import ensure_georeferenced
from src.preprocessing.scale_harmonizer import crop_and_harmonize_overlap
from src.preprocessing.band_selector import extract_or_synthesize_band
from src.registration.coarse_alignment import run_coarse_alignment
from src.registration.tiled_matching import run_tiled_matching
from src.registration.subpixel_ecc import refine_matches_subpixel
from src.registration.hybrid_transform import HybridTransform
from src.registration.warp import warp_image_subpixel
from src.registration.verifier import run_verification


def parse_args():
    parser = argparse.ArgumentParser(
        description="ChandaShakti: Universal Sub-Pixel Lunar Registration Engine (ISRO SIH 26166 v2.0)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # Required parameters
    parser.add_argument("--source", "-s", type=Path, required=True,
                        help="Path to source image (OHRC, IIRS, TMC-2)")
    parser.add_argument("--reference", "-r", type=Path, required=True,
                        help="Path to reference image (LRO NAC, WAC, SELENE TC)")
    parser.add_argument("--sensor_src", choices=["OHRC", "IIRS", "TMC", "TMC2"], default="OHRC",
                        help="Source sensor type (default: OHRC)")
    parser.add_argument("--sensor_ref", choices=["NAC", "WAC", "SELENE", "TC"], default="NAC",
                        help="Reference sensor type (default: NAC)")
    parser.add_argument("--out_dir", "-o", type=Path, default=Path("projects/run_v2"),
                        help="Output directory for results (default: projects/run_v2)")

    # Optional tuning flags
    parser.add_argument("--grid_size", type=int, nargs=2, default=[4, 4],
                        help="Tiled matching grid rows and cols (default: 4 4)")
    parser.add_argument("--method", choices=["loftr", "sift", "crater"], default="loftr",
                        help="Feature matching and pooling method (default: loftr)")
    parser.add_argument("--structural_method", choices=["phase_congruency", "gradient"], default="phase_congruency",
                        help="Structural feature representation method (default: phase_congruency)")
    parser.add_argument("--coarse_method", choices=["auto", "fft", "crater"], default="auto",
                        help="Coarse alignment solver (default: auto)")
    parser.add_argument("--poly_degree", type=int, default=2,
                        help="Degree of scanline drift polynomial (default: 2)")
    parser.add_argument("--tps_smoothing", type=float, default=0.05,
                        help="Smoothing factor for Thin Plate Spline (default: 0.05)")
    parser.add_argument("--ransac_threshold", type=float, default=1.2,
                        help="RANSAC sub-pixel inlier threshold in pixels (default: 1.2)")
    parser.add_argument("--warp_order", type=int, default=3,
                        help="Interpolation spline order for warping: 3=bicubic, 1=bilinear (default: 3)")
    parser.add_argument("--wac_band", type=int, default=7,
                        help="WAC spectral band for push-frame de-interleaving: 7=689nm (Red, default), 4=566nm (Green), 3=415nm (Blue)")
    parser.add_argument("--force", action="store_true",
                        help="Force recomputation of intermediate cached products")
    parser.add_argument("--skip_verify", action="store_true",
                        help="Skip Phase 4 verification engine")

    return parser.parse_args()


if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def print_banner():
    print("""
+===========================================================================+
|                                CHANDASHAKTI                               |
|         Universal Sub-Pixel Multi-Modal Lunar Registration Engine         |
|                     ISRO SIH 26166 - Next-Gen Pipeline                    |
|              Precision Target: < 0.2 px | Memory: Windowed BBox           |
+===========================================================================+
    """)


def run_pipeline(args):
    start_total_time = time.time()
    print_banner()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    diag_dir = out_dir / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    print(">> RUN CONFIGURATION:")
    print(f"  * Source:     {args.source} [{args.sensor_src}]")
    print(f"  * Reference:  {args.reference} [{args.sensor_ref}]")
    print(f"  * Output Dir: {out_dir}")
    print(f"  * Grid Size:  {args.grid_size[0]} x {args.grid_size[1]}")
    print(f"  * Match Method: {args.method.upper()}")
    print(f"  * Structural: {args.structural_method}")
    print(f"  * Warp Order: {args.warp_order} (Bicubic Spline)")
    print("-" * 75)

    # ──────────────────────────────────────────────────────────────────────────
    # STAGE 1: Bounding-Box Intersect & Scale Harmonization (Cam2Map)
    # ──────────────────────────────────────────────────────────────────────────
    t0 = time.time()
    print("\n[STAGE 1/5] Bounding-Box Overlap & Scale Harmonization...")

    actual_source_path = args.source

    # Hyperspectral band selection if source is IIRS
    if args.sensor_src == "IIRS":
        print("  [IIRS] Detected Hyperspectral cube. Performing solar-reflective SWIR band selection...")
        band_out = out_dir / "iirs_selected_band.tif"
        actual_source_path = extract_or_synthesize_band(
            iirs_path=args.source,
            output_path=band_out,
            ref_wavelength=689.0, # Target WAC/NAC visual band
            use_synthesis=True
        )

    # Georeference inputs if unprojected raw formats (PDS3 / PDS4)
    georef_dir = out_dir / "georeferenced"
    georef_dir.mkdir(parents=True, exist_ok=True)
    source_geo = ensure_georeferenced(actual_source_path, args.sensor_src, georef_dir, force=args.force, wac_band=getattr(args, "wac_band", 7))
    ref_geo = ensure_georeferenced(args.reference, args.sensor_ref, georef_dir, force=args.force, wac_band=getattr(args, "wac_band", 7))

    harmonized_dir = out_dir / "harmonized"
    harmonized_dir.mkdir(parents=True, exist_ok=True)

    source_cammap, ref_cropped, harm_meta = crop_and_harmonize_overlap(
        source_path=source_geo,
        ref_path=ref_geo,
        output_dir=harmonized_dir,
        prefix="bbox_overlap",
        force_recompute=args.force
    )
    print(f"  [OK] Stage 1 finished in {time.time() - t0:.2f}s")

    # ──────────────────────────────────────────────────────────────────────────
    # STAGE 2: Modality-Invariant Structure & Dual-Method Coarse Alignment
    # ──────────────────────────────────────────────────────────────────────────
    t0 = time.time()
    print("\n[STAGE 2/5] Modality-Invariant Structural Extraction & Coarse Alignment...")

    # Run dual-method coarse alignment
    coarse_res = run_coarse_alignment(
        source_harmonized_path=source_cammap,
        ref_cropped_path=ref_cropped,
        output_dir=out_dir,
        structural_method=args.structural_method
    )

    coarse_dx = coarse_res["dx"]
    coarse_dy = coarse_res["dy"]
    coarse_conf = coarse_res.get("confidence", 0.5)

    print(f"  Coarse Global Shift: dx = {coarse_dx:.2f} px, dy = {coarse_dy:.2f} px (Conf: {coarse_conf:.2f})")
    print(f"  [OK] Stage 2 finished in {time.time() - t0:.2f}s")

    # ──────────────────────────────────────────────────────────────────────────
    # STAGE 3: Uniform Tiled Matching & Sub-Pixel Continuous ECC Refinement
    # ──────────────────────────────────────────────────────────────────────────
    t0 = time.time()
    print(f"\n[STAGE 3/5] Uniform Spatial Tiled Matching ({args.grid_size[0]}x{args.grid_size[1]}) & ECC...")

    candidate_csv = out_dir / "candidate_matches.csv"
    subpixel_csv = out_dir / "subpixel_tie_points.csv"

    # Step 3.1: Streaming windowed tiled matching with coarse shift pre-positioning
    match_info = run_tiled_matching(
        src_input=source_cammap,
        ref_input=ref_cropped,
        coarse_result=coarse_res,
        coarse_dx=coarse_dx,
        coarse_dy=coarse_dy,
        output_csv=candidate_csv,
        tile_size=1600,
        step_size=1200,
        grid_size=(args.grid_size[0], args.grid_size[1]),
        method=args.method,
        structural_method=args.structural_method
    )

    # Step 3.2: Sub-Pixel Gauss-Newton ECC Refinement (< 0.2 px) via streaming windowed reads
    if match_info["total_matches"] > 0:
        matches_matrix = np.column_stack([match_info["src_pts"], match_info["ref_pts"]])
        ecc_info = refine_matches_subpixel(
            src_img=source_cammap,
            ref_img=ref_cropped,
            matches=matches_matrix,
            output_csv=subpixel_csv,
            patch_size=64
        )
        refined_matches = ecc_info["refined_matches"]
    else:
        refined_matches = np.empty((0, 5))

    # Step 3.3: Fit 3-Layer Physics-Grounded Hybrid Transformation
    model_json = out_dir / "hybrid_transform_model.json"
    hybrid_model = HybridTransform()
    
    with rasterio.open(source_cammap) as ds:
        image_shape = (ds.height, ds.width)

    if len(refined_matches) >= 6:
        if refined_matches.shape[1] >= 5:
            good_ecc = refined_matches[:, 4] >= 0.55
            if np.sum(good_ecc) >= 15:
                refined_matches = refined_matches[good_ecc]

        src_tie = refined_matches[:, :2]
        ref_tie = refined_matches[:, 2:4]
        print(f"  Fitting 3-Layer Hybrid Model on {len(src_tie)} sub-pixel tie points...")
        hybrid_model.fit(
            src_pts=src_tie,
            ref_pts=ref_tie,
            image_shape=image_shape,
            poly_degree=args.poly_degree,
            tps_smoothing=args.tps_smoothing,
            ransac_threshold=args.ransac_threshold
        )
        hybrid_model.save(model_json)
        if getattr(hybrid_model, "src_inliers", None) is not None:
            import pandas as pd
            inliers_csv = out_dir / "tie_points_inliers.csv"
            inlier_df = pd.DataFrame({
                "src_x": hybrid_model.src_inliers[:, 0],
                "src_y": hybrid_model.src_inliers[:, 1],
                "ref_x": hybrid_model.ref_inliers[:, 0],
                "ref_y": hybrid_model.ref_inliers[:, 1]
            })
            inlier_df.to_csv(inliers_csv, index=False)
    elif match_info["total_matches"] >= 6:
        print("  [WARNING] ECC yielded few points; fitting hybrid model on candidate matches...")
        hybrid_model.fit(
            src_pts=match_info["src_pts"],
            ref_pts=match_info["ref_pts"],
            image_shape=image_shape,
            poly_degree=args.poly_degree,
            tps_smoothing=args.tps_smoothing,
            ransac_threshold=args.ransac_threshold
        )
        hybrid_model.save(model_json)
        if getattr(hybrid_model, "src_inliers", None) is not None:
            import pandas as pd
            inliers_csv = out_dir / "tie_points_inliers.csv"
            inlier_df = pd.DataFrame({
                "src_x": hybrid_model.src_inliers[:, 0],
                "src_y": hybrid_model.src_inliers[:, 1],
                "ref_x": hybrid_model.ref_inliers[:, 0],
                "ref_y": hybrid_model.ref_inliers[:, 1]
            })
            inlier_df.to_csv(inliers_csv, index=False)
    else:
        print("  [WARNING] Sparse matches (<6); using coarse translation baseline as rigid model.")
        # Create pure translation affine matrix: [ [1, 0, coarse_dx], [0, 1, coarse_dy] ]
        pure_affine = np.array([
            [1.0, 0.0, coarse_dx],
            [0.0, 1.0, coarse_dy]
        ], dtype=np.float64)
        hybrid_model.affine_matrix = pure_affine
        hybrid_model.poly_coeffs_x = np.array([0.0])
        hybrid_model.poly_coeffs_y = np.array([0.0])
        hybrid_model.layer_params = {"inlier_count": 0, "total_points": 0, "rmse_layer1_affine": 0.0}
        hybrid_model.save(model_json)

    print(f"  [OK] Stage 3 finished in {time.time() - t0:.2f}s")

    # ──────────────────────────────────────────────────────────────────────────
    # STAGE 4: High-Precision Sub-Pixel Warping
    # ──────────────────────────────────────────────────────────────────────────
    t0 = time.time()
    print("\n[STAGE 4/5] High-Precision Sub-Pixel Image Warping...")

    registered_tif = out_dir / "registered_subpixel.tif"

    warp_image_subpixel(
        source_path=source_cammap,
        ref_path=ref_cropped,
        transform=hybrid_model,
        output_path=registered_tif,
        order=args.warp_order,
        block_rows=1024
    )
    print(f"  [OK] Stage 4 finished in {time.time() - t0:.2f}s")

    # ──────────────────────────────────────────────────────────────────────────
    # STAGE 5: Multi-Pillar Verification & Quality Control
    # ──────────────────────────────────────────────────────────────────────────
    t0 = time.time()
    print("\n[STAGE 5/5] Multi-Pillar Scientific Verification & Diagnostics...")

    if not args.skip_verify:
        metrics = run_verification(
            registered_path=registered_tif,
            ref_path=ref_cropped,
            output_dir=out_dir,
            tie_points_csv=subpixel_csv if subpixel_csv.exists() else candidate_csv,
            hybrid_model_json=model_json,
            grid_size=(args.grid_size[0], args.grid_size[1]),
            target_rmse_threshold=0.5
        )
    else:
        print("  [INFO] Verification stage skipped by user request.")
        metrics = {"verdict": "UNVERIFIED"}

    print(f"  [OK] Stage 5 finished in {time.time() - t0:.2f}s")

    total_elapsed = time.time() - start_total_time
    print(f"\n[SUCCESS] ALL PIPELINE PHASES COMPLETED SUCCESSFULLY IN {total_elapsed:.2f}s!")
    print(f"  Final Sub-Pixel Product: {registered_tif}")
    print(f"  Diagnostics Directory:   {diag_dir}")
    if isinstance(metrics, dict) and "verdict" in metrics:
        print(f"  Verification Verdict:    {metrics['verdict']}")
    print("=" * 75)
    return 0


def main():
    args = parse_args()
    try:
        sys.exit(run_pipeline(args))
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Pipeline execution failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
