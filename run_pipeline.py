import argparse
import sys
import time
import json
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.preprocessing.pipeline import process_lro, process_ch2
from src.preprocessing.scale_harmonizer import harmonize_scale
from src.registration.coarse_alignment import run_coarse_alignment
from src.registration.tiled_matching import run_tiled_matching
from src.registration.drift_profiler import run_drift_profile
from src.registration.hybrid_transform import fit_hybrid_transform
from src.registration.warp import run_warp
from src.registration.verifier import run_verification


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "End-to-End Lunar Multi-Sensor Co-Registration Pipeline\n"
            "100%% ISIS-free, WSL-free, Cross-Platform.\n\n"
            "Quickstart (already-normalized images):\n"
            "  python run_pipeline.py -s normalized/OHRC.tif -r normalized/NAC.tif\n\n"
            "Quickstart (from raw PDS4/PDS3 inputs, full preprocessing):\n"
            "  python run_pipeline.py --preprocess \\\n"
            "    --raw_source data/raw/ch2_ohrc.xml --source_sensor OHRC \\\n"
            "    --raw_ref    data/raw/M162680801LE.IMG --ref_sensor NAC"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # ── Preprocessing args (optional — use if starting from raw data) ───────
    preproc = parser.add_argument_group("Preprocessing (Stage -1)")
    preproc.add_argument("--preprocess", action="store_true",
                         help="Run preprocessing (ingest + georeference) before registration")
    preproc.add_argument("--raw_source", type=Path,
                         help="Raw source image (.xml, .IMG, .h5)")
    preproc.add_argument("--source_sensor", choices=["OHRC", "TMC2", "IIRS", "NAC", "WAC"],
                         help="Sensor type of the source image")
    preproc.add_argument("--raw_ref", type=Path,
                         help="Raw reference image (.IMG, .tif)")
    preproc.add_argument("--ref_sensor", choices=["NAC", "WAC", "OHRC", "TMC2", "IIRS"],
                         help="Sensor type of the reference image")

    # ── Registration args ────────────────────────────────────────────────────
    reg = parser.add_argument_group("Registration (Stages 0–4)")
    reg.add_argument("--source_img", "-s", type=Path,
                     help="Path to normalized source image (skipped if --preprocess)")
    reg.add_argument("--ref_img", "-r", type=Path,
                     help="Path to normalized reference image (skipped if --preprocess)")
    reg.add_argument("--output_dir", "-o", type=Path, default=Path("./projects/project_run"),
                     help="Output directory for all results (default: ./projects/project_run)")
    reg.add_argument("--config", "-c", type=Path,
                     default=ROOT_DIR / "configs" / "default_config.json",
                     help="Path to config JSON")
    reg.add_argument("--stages", type=str, default="0,1,2,3,4",
                     help="Comma-separated stages to run (0=Coarse, 1=Tiled, 2=Fit, 3=Warp, 4=Verify)")
    reg.add_argument("--skip_verify", action="store_true",
                     help="Skip stage 4 verification")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    norm_dir   = output_dir / "normalized"
    output_dir.mkdir(parents=True, exist_ok=True)
    norm_dir.mkdir(parents=True, exist_ok=True)

    config = {}
    if args.config and args.config.exists():
        with open(args.config, "r", encoding="utf-8") as f:
            config = json.load(f)
        print(f"Loaded config from: {args.config}")

    selected_stages = [int(s.strip()) for s in args.stages.split(",") if s.strip().isdigit()]

    print("=" * 80)
    print("LUNAR CO-REGISTRATION MASTER PIPELINE  (ISIS-free)")
    print("=" * 80)

    start_total = time.perf_counter()

    # ── Stage -1: Native Preprocessing (ingest + georeference) ──────────────
    source_img = args.source_img
    ref_img    = args.ref_img

    if args.preprocess:
        if not args.raw_source or not args.source_sensor:
            print("[ERROR] --preprocess requires --raw_source and --source_sensor")
            sys.exit(1)
        if not args.raw_ref or not args.ref_sensor:
            print("[ERROR] --preprocess requires --raw_ref and --ref_sensor")
            sys.exit(1)

        print("\n>>> [STAGE -1] NATIVE PREPROCESSING (no ISIS, no WSL)...")

        if args.source_sensor in ("NAC", "WAC"):
            source_img = process_lro(args.raw_source, args.source_sensor, norm_dir)
        else:
            source_img = process_ch2(args.raw_source, args.source_sensor, norm_dir)

        if args.ref_sensor in ("NAC", "WAC"):
            ref_img = process_lro(args.raw_ref, args.ref_sensor, norm_dir)
        else:
            ref_img = process_ch2(args.raw_ref, args.ref_sensor, norm_dir)

        print(f"  Source normalized → {source_img}")
        print(f"  Ref    normalized → {ref_img}")

        # ── Scale Harmonization (SIH Scale Invariance requirement) ──────────
        coarse_source_path = norm_dir / f"{source_img.stem}_coarse.tif"
        print("\n>>> [SCALE HARMONIZER] Checking resolution mismatch...")
        _, _, scale_factor = harmonize_scale(
            source_path=source_img,
            ref_path=ref_img,
            out_coarse_path=coarse_source_path if not source_img == ref_img else None,
        )
        if scale_factor > 1.0:
            print(f"  Scale factor applied: {scale_factor:.2f}×  →  {coarse_source_path}")
            # Coarse stages use the downsampled image; fine stages use full-res
            config['coarse_source_override'] = str(coarse_source_path)
            config['scale_factor'] = scale_factor

    if source_img is None or ref_img is None:
        print("[ERROR] Must provide --source_img and --ref_img, "
              "or use --preprocess with raw inputs.")
        sys.exit(1)

    print(f"\n  Source Image: {source_img}")
    print(f"  Ref Image   : {ref_img}")
    print(f"  Output Dir  : {output_dir}")
    print(f"  Stages      : {selected_stages}")
    print("=" * 80)

    # ── Stage 0: Coarse Alignment ────────────────────────────────────────────
    if 0 in selected_stages:
        print("\n>>> [STAGE 0] RUNNING COARSE ALIGNMENT...")
        # If scale harmonization produced a coarse image, pass it here
        coarse_src = Path(config.get('coarse_source_override', source_img))
        run_coarse_alignment(coarse_src, ref_img, output_dir, config)

    # ── Stage 1: Fine Tiled Matching ─────────────────────────────────────────
    if 1 in selected_stages:
        print("\n>>> [STAGE 1] RUNNING TILED FINE MATCHING...")
        run_tiled_matching(source_img, ref_img, output_dir, config)

    # ── Stage 2: Drift Profiler & Hybrid Model Fitting ───────────────────────
    if 2 in selected_stages:
        print("\n>>> [STAGE 2] RUNNING DRIFT ANALYSIS & HYBRID MODEL FITTING...")
        run_drift_profile(output_dir, config)
        fit_hybrid_transform(output_dir, config)

    # ── Stage 3: Sub-pixel Inverse Warp ──────────────────────────────────────
    warped_product = None
    if 3 in selected_stages:
        print("\n>>> [STAGE 3] RUNNING SUB-PIXEL INVERSE WARPING...")
        warped_product = run_warp(source_img, ref_img, output_dir, config)

    # ── Stage 4: Independent Verification ────────────────────────────────────
    if 4 in selected_stages and not args.skip_verify:
        print("\n>>> [STAGE 4] RUNNING REGISTRATION QUALITY VERIFICATION...")
        if warped_product is None:
            warped_product = output_dir / f"{source_img.stem}_registered.tif"
        if warped_product.exists():
            run_verification(warped_product, ref_img, output_dir, config)
        else:
            print(f"[WARNING] Registered product {warped_product} not found, skipping verification.")

    total_time = time.perf_counter() - start_total
    print("\n" + "=" * 80)
    print(f"PIPELINE COMPLETE in {total_time:.2f} seconds!")
    print(f"Results available in: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
