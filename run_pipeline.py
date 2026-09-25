"""
run_pipeline.py
===============
ChandaShakti: Universal Sub-Pixel Multi-Modal Lunar Image Co-Registration Engine
ISRO Smart India Hackathon (SIH 2024) — Problem Statement SIH26166.

Features:
  - Phase 1: Robust scale harmonization, SPICE ray-tracing & structural enhancement
  - Phase 2: Dual-method coarse alignment (Structural FFT + Crater Rim Consensus Voting)
  - Phase 3: Dense LoFTR attention matching + Sub-Pixel continuous Gauss-Newton ECC
  - Phase 4: 3-Layer physics-grounded hybrid transformation (Affine + Drift + TPS)
  - Phase 5: Streaming block-wise bicubic warping & multi-pillar scientific verification

Usage Example:
  python run_pipeline.py --source <path> --reference <path> \
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
import pandas as pd
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
from src.registration.web_previews import export_raster_preview


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
    parser.add_argument("--method", choices=["loftr", "ensemble", "crater"], default="loftr",
                        help="Feature matching and pooling method (default: loftr)")
    parser.add_argument("--coarse_dx", type=float, default=None,
                        help="Optional manual/anchor initial coarse shift in X pixels")
    parser.add_argument("--coarse_dy", type=float, default=None,
                        help="Optional manual/anchor initial coarse shift in Y pixels")
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
    parser.add_argument("--export_native", action="store_true",
                        help="Export secondary un-decimated native-GSD registered raster (large multi-gigapixel output)")

    return parser.parse_args()


if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
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


def _emit_stage(progress_callback, stage_id, title, state, progress_pct, message, details=None):
    """Publish a compact stage event without changing the standalone CLI flow."""
    if progress_callback is None:
        return
    event = {
        "stage_id": stage_id,
        "title": title,
        "state": state,
        "progress_pct": progress_pct,
        "message": message,
        "details": details or {},
    }
    try:
        progress_callback(event)
    except Exception as exc:
        # Dashboard telemetry must not abort a scientific run.
        print(f"[PROGRESS] Could not publish stage update: {exc}")


def _try_export_web_preview(raster_path, output_png):
    """Best-effort dashboard quicklook; preview failure must not fail registration."""
    try:
        return export_raster_preview(raster_path, output_png)
    except Exception as exc:
        print(f"[PREVIEW] Could not create {Path(output_png).name}: {exc}")
        return None


def run_pipeline(args, progress_callback=None):
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
    _emit_stage(progress_callback, "stage_1", "Ingest and harmonize", "running", 5,
                "Inspecting products, preparing sensor data, and finding the common geographic overlap.")

    actual_source_path = args.source
    best_band_info = None

    # Hyperspectral band selection if source is IIRS
    if args.sensor_src == "IIRS":
        print("  [IIRS] Detected Hyperspectral cube. Performing solar-reflective SWIR band selection...")
        from src.preprocessing.band_selector import select_best_band_for_reference
        
        best_band_info = select_best_band_for_reference(
            iirs_path=args.source,
            reference_sensor=args.sensor_ref,
            max_swir_band=40
        )
        
        band_out = out_dir / f"iirs_selected_band_{best_band_info['selected_band']}.tif"
        actual_source_path = extract_or_synthesize_band(
            iirs_path=args.source,
            output_path=band_out,
            selected_band=best_band_info['selected_band'],
            synthesize_pan=False
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
        force_recompute=args.force,
        sensor_ref=args.sensor_ref,
        sensor_src=args.sensor_src
    )
    stage1_source_preview = diag_dir / "stage_01_source_harmonized.png"
    stage1_reference_preview = diag_dir / "stage_01_reference_harmonized.png"
    preview_info = {}
    stage1_previews = []
    if progress_callback is not None:
        for key, raster, preview_path in (
            ("source", source_cammap, stage1_source_preview),
            ("reference", ref_cropped, stage1_reference_preview),
        ):
            result = _try_export_web_preview(raster, preview_path)
            if result:
                preview_info[key] = result
                stage1_previews.append(preview_path.name)
    with rasterio.open(source_cammap) as harmonized_ds:
        harmonized_shape = [harmonized_ds.height, harmonized_ds.width]
    stage1_details = {
        "source_file": source_cammap.name,
        "reference_file": ref_cropped.name,
        "harmonized_shape_px": harmonized_shape,
        "gsd_m": harm_meta.get("gsd_harm"),
        "scale_ratio": harm_meta.get("scale_ratio"),
        "source_overlap_pct": harm_meta.get("src_overlap_pct"),
        "reference_overlap_pct": harm_meta.get("ref_overlap_pct"),
        "orientation": harm_meta.get("orientation"),
        "selected_iirs_band": best_band_info,
        "previews": stage1_previews,
        "preview_dimensions": preview_info,
        "files_created": [source_cammap.name, ref_cropped.name, "bbox_overlap_meta.json"],
    }
    if harm_meta.get("source_native_crop"):
        stage1_details["files_created"].append(Path(harm_meta["source_native_crop"]).name)
    _emit_stage(progress_callback, "stage_1", "Ingest and harmonize", "complete", 18,
                "Both products were georeferenced and prepared on the same overlap grid.", stage1_details)
    print(f"  [OK] Stage 1 finished in {time.time() - t0:.2f}s")

    # ──────────────────────────────────────────────────────────────────────────
    # STAGE 2: Modality-Invariant Structure & Dual-Method Coarse Alignment
    # ──────────────────────────────────────────────────────────────────────────
    t0 = time.time()
    print("\n[STAGE 2/5] Modality-Invariant Structural Extraction & Coarse Alignment...")
    _emit_stage(progress_callback, "stage_2", "Structural features and coarse alignment", "running", 20,
                "Estimating the global shift and along-track drift from the harmonized pair.")

    if args.coarse_dx is not None and args.coarse_dy is not None:
        coarse_dx = float(args.coarse_dx)
        coarse_dy = float(args.coarse_dy)
        coarse_conf = 1.0
        coarse_res = {
            "dx": coarse_dx,
            "dy": coarse_dy,
            "confidence": 1.0,
            "drift_model": {
                "dy_slope": 0.0, "dy_intercept": coarse_dy,
                "dx_slope": 0.0, "dx_intercept": coarse_dx
            }
        }
        print(f"  [COARSE-OVERRIDE] Using user-specified initial coarse shift: dx = {coarse_dx:.2f} px, dy = {coarse_dy:.2f} px")
    else:
        # Run dual-method coarse alignment
        coarse_res = run_coarse_alignment(
            source_harmonized_path=source_cammap,
            ref_cropped_path=ref_cropped,
            output_dir=out_dir,
            structural_method=args.structural_method,
            coarse_method=getattr(args, "coarse_method", "auto")
        )
        coarse_dx = coarse_res["dx"]
        coarse_dy = coarse_res["dy"]
        coarse_conf = coarse_res.get("confidence", 0.5)

    print(f"  Coarse Global Shift: dx = {coarse_dx:.2f} px, dy = {coarse_dy:.2f} px (Conf: {coarse_conf:.2f})")
    stage2_details = {
        "dx_px": float(coarse_dx),
        "dy_px": float(coarse_dy),
        "confidence": float(coarse_conf),
        "method_used": coarse_res.get("method_used", getattr(args, "coarse_method", "auto")),
        "drift_model": coarse_res.get("drift_model", {}),
        "strip_count": len(coarse_res.get("strip_profiles", [])),
        "files_created": ["coarse_alignment_result.json"],
    }
    _emit_stage(progress_callback, "stage_2", "Structural features and coarse alignment", "complete", 35,
                "Coarse shift and drift estimates are ready to position the tiled matcher.", stage2_details)
    print(f"  [OK] Stage 2 finished in {time.time() - t0:.2f}s")

    # ──────────────────────────────────────────────────────────────────────────
    # STAGE 3: Uniform Tiled Matching & Sub-Pixel Continuous ECC Refinement
    # ──────────────────────────────────────────────────────────────────────────
    t0 = time.time()
    print(f"\n[STAGE 3/5] Uniform Spatial Tiled Matching ({args.grid_size[0]}x{args.grid_size[1]}) & ECC...")
    _emit_stage(progress_callback, "stage_3", "Tiled matching and transform fit", "running", 38,
                "Finding correspondences in spatial tiles, refining them to subpixel positions, and fitting the deformation model.")

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
    ecc_info = {"success_count": 0}
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
        # Filter for high-confidence ECC converged points (ecc_rho >= 0.55) if 5th column exists
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

    stage3_details = {
        "candidate_matches": int(match_info.get("total_matches", 0)),
        "ecc_refined_matches": int(ecc_info.get("success_count", len(refined_matches))),
        "populated_cells": int(match_info.get("populated_cells", 0)),
        "spatial_entropy": match_info.get("spatial_entropy"),
        "inlier_matches": int(len(hybrid_model.src_inliers)) if getattr(hybrid_model, "src_inliers", None) is not None else 0,
        "transform_stats": getattr(hybrid_model, "layer_params", {}),
        "files_created": [candidate_csv.name, subpixel_csv.name, model_json.name],
    }
    if (out_dir / "tie_points_inliers.csv").exists():
        stage3_details["files_created"].append("tie_points_inliers.csv")
    _emit_stage(progress_callback, "stage_3", "Tiled matching and transform fit", "complete", 63,
                "Candidate matches, subpixel refinements, and the hybrid transform model are ready for warping.", stage3_details)
    print(f"  [OK] Stage 3 finished in {time.time() - t0:.2f}s")

    # ──────────────────────────────────────────────────────────────────────────
    # STAGE 4: High-Precision Sub-Pixel Warping
    # ──────────────────────────────────────────────────────────────────────────
    t0 = time.time()
    print("\n[STAGE 4/5] High-Precision Sub-Pixel Image Warping...")
    _emit_stage(progress_callback, "stage_4", "Warp registered product", "running", 65,
                "Applying the fitted transform to write the registered image on the harmonized reference grid.")

    registered_tif = out_dir / "registered_subpixel.tif"

    # Stage 4.1: Warp on harmonized reference grid for verified sub-pixel registration
    warp_image_subpixel(
        source_path=source_cammap,
        ref_path=ref_cropped,
        transform=hybrid_model,
        output_path=registered_tif,
        order=args.warp_order,
        block_rows=1024
    )

    # Stage 4.2: Dual-Resolution Native Export (Optional: enabled via --export_native)
    native_crop = harm_meta.get("source_native_crop")
    if getattr(args, "export_native", False) and native_crop and Path(native_crop).exists():
        try:
            with rasterio.open(native_crop) as src:
                native_gsd = abs(src.transform.a)
            native_out = out_dir / f"registered_native_{int(round(native_gsd))}m.tif"
            print(f"  [WARP] Dual-Resolution: Warping native resolution product -> {native_out.name} (GSD={native_gsd:.2f}m)...")
            warp_image_subpixel(
                source_path=native_crop,
                ref_path=ref_cropped,
                transform=hybrid_model,
                output_path=native_out,
                order=args.warp_order,
                block_rows=1024,
                target_gsd=native_gsd
            )
        except Exception as e:
            print(f"  [WARP] Warning: Native export encountered an issue: {e}")
    registered_preview = diag_dir / "registered_preview.png"
    registered_preview_info = None
    stage4_files = [registered_tif.name, "warp_composite_overlay.png"]
    native_tifs = sorted(out_dir.glob("registered_native_*m.tif"))
    if progress_callback is not None:
        registered_preview_info = _try_export_web_preview(registered_tif, registered_preview)
        if registered_preview_info:
            stage4_files.append(registered_preview.name)
        for native_tif in native_tifs:
            native_preview = diag_dir / f"{native_tif.stem}_preview.png"
            if _try_export_web_preview(native_tif, native_preview):
                stage4_files.append(native_preview.name)
            stage4_files.append(native_tif.name)
    with rasterio.open(registered_tif) as registered_ds:
        registered_details = {
            "width_px": registered_ds.width,
            "height_px": registered_ds.height,
            "crs": registered_ds.crs.to_string() if registered_ds.crs else None,
            "pixel_size_x": abs(registered_ds.transform.a),
            "pixel_size_y": abs(registered_ds.transform.e),
        }
    _emit_stage(progress_callback, "stage_4", "Warp registered product", "complete", 80,
                "Registered GeoTIFF and browser-sized quicklook are ready for verification.", {
                    **registered_details,
                    "preview": registered_preview.name if registered_preview_info else None,
                    "preview_dimensions": registered_preview_info,
                    "native_export_requested": bool(getattr(args, "export_native", False)),
                    "files_created": stage4_files,
                })
    print(f"  [OK] Stage 4 finished in {time.time() - t0:.2f}s")

    # ──────────────────────────────────────────────────────────────────────────
    # STAGE 5: Multi-Pillar Verification & Quality Control
    # ──────────────────────────────────────────────────────────────────────────
    t0 = time.time()
    print("\n[STAGE 5/5] Multi-Pillar Scientific Verification & Diagnostics...")
    _emit_stage(progress_callback, "stage_5", "Verify and summarize", "running", 82,
                "Computing registration quality metrics and compact visual diagnostics.")

    if not args.skip_verify:
        metrics = run_verification(
            registered_path=registered_tif,
            ref_path=ref_cropped,
            output_dir=out_dir,
            tie_points_csv=(out_dir / "tie_points_inliers.csv") if (out_dir / "tie_points_inliers.csv").exists() else (subpixel_csv if subpixel_csv.exists() else candidate_csv),
            hybrid_model_json=model_json,
            grid_size=(args.grid_size[0], args.grid_size[1]),
            target_rmse_threshold=0.5
        )
    else:
        print("  [INFO] Verification stage skipped by user request.")
        metrics = {"verdict": "UNVERIFIED"}

    stage5_details = {
        "metrics": {key: value for key, value in metrics.items() if key not in {"registered_raster", "reference_raster"}},
        "files_created": [
            "diagnostics/verification_metrics.json",
            "diagnostics/registration_verification.png",
            "diagnostics/difference_heatmap.png",
            "diagnostics/overview_side_by_side.png",
            "diagnostics/overview_false_color.png",
        ] if not args.skip_verify else [],
    }
    _emit_stage(progress_callback, "stage_5", "Verify and summarize", "complete", 97,
                "Verification finished; inspect the verdict, metrics, and diagnostic previews.", stage5_details)
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
