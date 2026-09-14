import argparse
import sys
import time
import json
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.registration.coarse_alignment import run_coarse_alignment
from src.registration.tiled_matching import run_tiled_matching
from src.registration.drift_profiler import run_drift_profile
from src.registration.hybrid_transform import fit_hybrid_transform
from src.registration.warp import run_warp
from src.registration.verifier import run_verification


def parse_args():
    parser = argparse.ArgumentParser(description="End-to-End Lunar Multi-Sensor Co-Registration Pipeline")
    parser.add_argument("--source_img", "-s", type=Path, required=True, help="Path to normalized source image (e.g. OHRC)")
    parser.add_argument("--ref_img", "-r", type=Path, required=True, help="Path to normalized reference image (e.g. NAC)")
    parser.add_argument("--output_dir", "-o", type=Path, default=Path("./projects/project_run"), help="Output directory for results")
    parser.add_argument("--config", "-c", type=Path, default=ROOT_DIR / "configs" / "default_config.json", help="Path to config JSON")
    parser.add_argument("--stages", type=str, default="0,1,2,3,4", help="Comma-separated stages to run (0=Coarse, 1=Tiled, 2=Fit, 3=Warp, 4=Verify)")
    parser.add_argument("--skip_verify", action="store_true", help="Skip stage 4 verification")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = {}
    if args.config and args.config.exists():
        with open(args.config, "r", encoding="utf-8") as f:
            config = json.load(f)
        print(f"Loaded config from: {args.config}")

    selected_stages = [int(s.strip()) for s in args.stages.split(",") if s.strip().isdigit()]

    print("=" * 80)
    print("LUNAR CO-REGISTRATION MASTER PIPELINE")
    print(f"Source Image: {args.source_img}")
    print(f"Ref Image   : {args.ref_img}")
    print(f"Output Dir  : {output_dir}")
    print(f"Stages      : {selected_stages}")
    print("=" * 80)

    start_total = time.perf_counter()

    # Stage 0: Coarse Alignment
    if 0 in selected_stages:
        print("\n>>> [STAGE 0] RUNNING COARSE ALIGNMENT...")
        run_coarse_alignment(args.source_img, args.ref_img, output_dir, config)

    # Stage 1: Fine Tiled Matching
    if 1 in selected_stages:
        print("\n>>> [STAGE 1] RUNNING TILED FINE MATCHING...")
        run_tiled_matching(args.source_img, args.ref_img, output_dir, config)

    # Stage 2: Drift Profiler & Hybrid Model Fitting
    if 2 in selected_stages:
        print("\n>>> [STAGE 2] RUNNING DRIFT ANALYSIS & HYBRID MODEL FITTING...")
        run_drift_profile(output_dir, config)
        fit_hybrid_transform(output_dir, config)

    # Stage 3: Coordinate Warping & Resampling
    warped_product = None
    if 3 in selected_stages:
        print("\n>>> [STAGE 3] RUNNING SUB-PIXEL INVERSE WARPING...")
        warped_product = run_warp(args.source_img, args.ref_img, output_dir, config)

    # Stage 4: Independent Verification
    if 4 in selected_stages and not args.skip_verify:
        print("\n>>> [STAGE 4] RUNNING REGISTRATION QUALITY VERIFICATION...")
        if warped_product is None:
            warped_product = output_dir / f"{args.source_img.stem}_registered.tif"
        if warped_product.exists():
            run_verification(warped_product, args.ref_img, output_dir, config)
        else:
            print(f"[WARNING] Registered product {warped_product} not found, skipping verification.")

    total_time = time.perf_counter() - start_total
    print("\n" + "=" * 80)
    print(f"PIPELINE COMPLETE in {total_time:.2f} seconds!")
    print(f"Results available in: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
