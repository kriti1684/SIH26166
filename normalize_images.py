from pathlib import Path
import numpy as np
import rasterio


# ------------------------------------------------------------
# INPUT FILES
# ------------------------------------------------------------

OHRC = Path(
    r"C:\#Padhai\E\SIH1\ch2_ohr_ncp_20210401T2357376656_d_img_d18_NACscale.tif"
)

NAC = Path(
    r"C:\#Padhai\E\SIH1\M1350459544RE_map.tif"
)

OUTPUT_DIR = Path(r"C:\#Padhai\E\SIH1\normalized")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# OHRC
# Already uint8 / 0-255
# Just copy it while preserving all GeoTIFF metadata.
# ------------------------------------------------------------

ohrc_output = OUTPUT_DIR / "OHRC_normalized.tif"

with rasterio.open(OHRC) as src:
    profile = src.profile.copy()
    data = src.read(1)

    # Explicitly keep uint8
    data = data.astype(np.uint8)

    # Keep NoData = 0
    profile.update(
        dtype="uint8",
        count=1,
        nodata=0
    )

    with rasterio.open(ohrc_output, "w", **profile) as dst:
        dst.write(data, 1)

print(f"Saved: {ohrc_output}")


# ------------------------------------------------------------
# NAC
# Convert valid Float32 pixels to uint8 [0, 255]
# while preserving NoData.
# ------------------------------------------------------------

nac_output = OUTPUT_DIR / "NAC_normalized.tif"

with rasterio.open(NAC) as src:

    data = src.read(1).astype(np.float32)

    src_nodata = src.nodata

    # Build valid-pixel mask
    valid = np.isfinite(data)

    if src_nodata is not None:
        valid &= ~np.isclose(data, src_nodata)

    if not np.any(valid):
        raise RuntimeError("NAC contains no valid pixels.")

    valid_values = data[valid]

    src_min = float(valid_values.min())
    src_max = float(valid_values.max())

    print("\nNAC normalization:")
    print(f"Valid minimum = {src_min}")
    print(f"Valid maximum = {src_max}")

    if src_max <= src_min:
        raise RuntimeError("Invalid NAC range.")

    # Start with a uint8 array.
    # Use 0 as NoData in the output.
    normalized = np.zeros(data.shape, dtype=np.uint8)

    # Linear min-max scaling of VALID pixels only.
    scaled = (
        (data[valid] - src_min)
        / (src_max - src_min)
        * 255.0
    )

    normalized[valid] = np.clip(
        np.rint(scaled),
        0,
        255
    ).astype(np.uint8)

    profile = src.profile.copy()

    # Preserve spatial metadata, change only raster data type
    profile.update(
        dtype="uint8",
        count=1,
        nodata=0
    )

    with rasterio.open(nac_output, "w", **profile) as dst:
        dst.write(normalized, 1)

print(f"Saved: {nac_output}")