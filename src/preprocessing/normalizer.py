import argparse
from pathlib import Path
import numpy as np
import rasterio


def normalize_tiff(input_tif: Path, output_tif: Path, is_ohrc: bool = False):
    """
    Normalize TIFF to uint8 [0, 255], preserving NoData=0 and spatial metadata.
    """
    input_tif = Path(input_tif)
    output_tif = Path(output_tif)
    output_tif.parent.mkdir(parents=True, exist_ok=True)

    print(f"[NORMALIZING]: {input_tif} -> {output_tif}")
    with rasterio.open(input_tif) as src:
        profile = src.profile.copy()
        data = src.read(1)

        if is_ohrc or data.dtype == np.uint8:
            # OHRC is typically already uint8
            data = data.astype(np.uint8)
            profile.update(driver="GTiff", dtype="uint8", count=1, nodata=0)
            with rasterio.open(output_tif, "w", **profile) as dst:
                dst.write(data, 1)
        else:
            # Float32 / Int16 to uint8 linear scaling
            data = data.astype(np.float32)
            src_nodata = src.nodata
            valid = np.isfinite(data)
            if src_nodata is not None:
                valid &= ~np.isclose(data, src_nodata)

            if not np.any(valid):
                raise RuntimeError(f"{input_tif} contains no valid pixels.")

            valid_values = data[valid]
            src_min = float(valid_values.min())
            src_max = float(valid_values.max())

            normalized = np.zeros(data.shape, dtype=np.uint8)
            if src_max > src_min:
                scaled = ((data[valid] - src_min) / (src_max - src_min)) * 255.0
                normalized[valid] = np.clip(np.rint(scaled), 0, 255).astype(np.uint8)

            profile.update(driver="GTiff", dtype="uint8", count=1, nodata=0)
            with rasterio.open(output_tif, "w", **profile) as dst:
                dst.write(normalized, 1)

    print(f"[SUCCESS] Saved normalized image to: {output_tif}")
    return output_tif


def main():
    parser = argparse.ArgumentParser(description="Linear Min-Max Normalizer for Lunar Satellite GeoTIFFs")
    parser.add_argument("--input", "-i", type=Path, required=True, help="Input GeoTIFF path")
    parser.add_argument("--output", "-o", type=Path, required=True, help="Output normalized GeoTIFF path")
    parser.add_argument("--is_ohrc", action="store_true", help="Set flag if input is already uint8 OHRC")
    args = parser.parse_args()

    normalize_tiff(args.input, args.output, is_ohrc=args.is_ohrc)


if __name__ == "__main__":
    main()
