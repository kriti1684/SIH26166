from pathlib import Path
from typing import Dict, Any

import cv2
import numpy as np
import rasterio
from rasterio.enums import Resampling


def export_raster_preview(
    raster_path: Path,
    output_png: Path,
    max_dim: int = 1400,
) -> Dict[str, Any]:
    """Windowed/decimated single-band render; never loads a full raster in RAM."""
    raster_path = Path(raster_path)
    output_png = Path(output_png)
    with rasterio.open(raster_path) as src:
        scale = min(1.0, max_dim / max(src.width, src.height))
        width = max(1, int(round(src.width * scale)))
        height = max(1, int(round(src.height * scale)))
        data = src.read(
            1,
            out_shape=(height, width),
            resampling=Resampling.bilinear,
            masked=True,
        ).astype(np.float32)
        valid = ~np.ma.getmaskarray(data) & np.isfinite(np.asarray(data))
        if src.nodata is not None:
            valid &= np.asarray(data) != float(src.nodata)

    values = np.asarray(data)[valid]
    if values.size < 16:
        image = np.zeros((height, width), dtype=np.uint8)
    else:
        low, high = np.percentile(values, (2, 98))
        if high <= low:
            high = low + 1.0
        image = np.clip((np.asarray(data, dtype=np.float32) - low) / (high - low) * 255.0, 0, 255).astype(np.uint8)
        image[~valid] = 0

    output_png.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_png), image):
        raise RuntimeError(f"Could not write web preview: {output_png.name}")
    return {"width": width, "height": height, "max_dimension": max(width, height)}
