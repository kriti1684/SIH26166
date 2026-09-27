import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
import rasterio
from rasterio.crs import CRS


_NS = {
    'pds':  'http://pds.nasa.gov/pds4/pds/v1',
    'isda': 'https://isda.issdc.gov.in/pds4/isda/v1',
}

# Standard Equirectangular Moon CRS (spherical, R = 1 737 400 m)
MOON_EQR_WKT = (
    'PROJCS["Equirectangular Moon",'
    'GEOGCS["GCS_Moon",DATUM["D_Moon",'
    'SPHEROID["Moon_LocalRadius",1737400,0]],'
    'PRIMEM["Reference_Meridian",0],'
    'UNIT["degree",0.0174532925199433]],'
    'PROJECTION["Equirectangular"],'
    'PARAMETER["standard_parallel_1",0],'
    'PARAMETER["central_meridian",0],'
    'PARAMETER["false_easting",0],'
    'PARAMETER["false_northing",0],'
    'UNIT["metre",1]]'
)
MOON_CRS = CRS.from_wkt(MOON_EQR_WKT)



def load_raster(file_path: Path) -> Tuple[np.ndarray, rasterio.DatasetReader]:
    """
    Load any supported raster format into a NumPy array.

    Supports:
      - GeoTIFF (.tif / .tiff)
      - PDS3 raw image (.IMG / .img)  — GDAL PDS driver
      - PDS4 .xml label               — rasterio opens via linked .img
      - IIRS HDF5 (.h5)               — returns first usable band

    Returns
    -------
    (array, dataset)
        array   : 2-D uint16 NumPy array (first/best band)
        dataset : open rasterio DatasetReader (caller must close)
    """
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()

    if suffix in ('.h5', '.hdf5'):
        return _load_iirs_h5(file_path)

    xml_path = file_path if suffix == '.xml' else file_path.with_suffix('.xml')
    if not xml_path.exists():
        xml_path = file_path.with_suffix('.XML')

    if xml_path.exists():
        try:
            ds = rasterio.open(file_path)
            arr = ds.read(1)
            return arr, ds
        except Exception:
            pass

        try:
            return _load_pds4_raw(xml_path, file_path if suffix != '.xml' else None)
        except Exception as e:
            print(f"[INGEST] Native PDS4 loader warning: {e}")

    try:
        ds = rasterio.open(file_path)
        arr = ds.read(1)
        return arr, ds
    except Exception as e:
        raise ValueError(f"Could not load raster from {file_path}: {e}")


def _load_pds4_raw(xml_path: Path, img_path: Optional[Path] = None) -> Tuple[np.ndarray, None]:
    """
    Directly parse PDS4 XML metadata and load raw binary raster into a NumPy array.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    ns = {'pds': 'http://pds.nasa.gov/pds4/pds/v1'}

    if img_path is None or not img_path.exists():
        file_name_el = root.find('.//pds:File/pds:file_name', ns)
        if file_name_el is not None and (xml_path.parent / file_name_el.text.strip()).exists():
            img_path = xml_path.parent / file_name_el.text.strip()
        else:
            for ext in ('.img', '.IMG', '.dat', '.DAT'):
                cand = xml_path.with_suffix(ext)
                if cand.exists():
                    img_path = cand
                    break

    if img_path is None or not img_path.exists():
        raise FileNotFoundError(f"Could not find matching image data file for {xml_path}")

    lines_el = root.find('.//pds:Axis_Array[pds:axis_name="Line"]/pds:elements', ns)
    samples_el = root.find('.//pds:Axis_Array[pds:axis_name="Sample"]/pds:elements', ns)
    if lines_el is None or samples_el is None:
        raise ValueError(f"Could not find Line/Sample dimensions in {xml_path}")

    lines = int(lines_el.text.strip())
    samples = int(samples_el.text.strip())

    offset_el = root.find('.//pds:Array_2D_Image/pds:offset', ns)
    offset = int(offset_el.text.strip()) if offset_el is not None else 0

    dtype_el = root.find('.//pds:Element_Array/pds:data_type', ns)
    dtype_str = dtype_el.text.strip() if dtype_el is not None else 'UnsignedByte'

    dtype_map = {
        'UnsignedByte': np.uint8,
        'Byte': np.int8,
        'UnsignedMSB2': '>u2',
        'SignedMSB2': '>i2',
        'UnsignedLSB2': '<u2',
        'SignedLSB2': '<i2',
        'IEEE754MSBSingle': '>f4',
        'IEEE754LSBSingle': '<f4',
    }
    np_dtype = dtype_map.get(dtype_str, np.uint8)

    print(f"[INGEST] Loading PDS4 binary: {img_path.name} ({lines}x{samples}, offset={offset}, dtype={dtype_str})")
    arr = np.memmap(img_path, dtype=np_dtype, mode='r', offset=offset, shape=(lines, samples))
    return np.array(arr), None


def _load_iirs_h5(h5_path: Path, target_band_idx: int = 1) -> Tuple[np.ndarray, None]:
    """
    Load a single optical band from an IIRS HDF5 file using hyperslab streaming.
    Memory usage remains < 50 MB by directly slicing the h5py Dataset on disk.

    IIRS has ~256 spectral bands at 8-20 m GSD.
    Target band is configurable (usually 0/1 for ~800nm reflective).

    Returns
    -------
    (array, None)
        The second element is None because h5py doesn't produce a rasterio
        dataset. The caller must georef using XML corners independently.
    """
    try:
        import h5py
    except ImportError:
        raise ImportError(
            "h5py is required to read IIRS .h5 files. "
            "Install it with: pip install h5py"
        )

    with h5py.File(h5_path, 'r') as f:
        # Try common IIRS dataset paths used by ISRO/PDS4 archive
        candidates = ['Image/Data', 'data', 'image', '/Image/Image']
        arr = None
        for key in candidates:
            if key in f:
                dset = f[key]
                # Shape is (bands, lines, samples) or (lines, samples, bands)
                if len(dset.shape) == 3:
                    if dset.shape[0] < dset.shape[1]:  # (bands, lines, samples)
                        band_idx = min(target_band_idx, dset.shape[0] - 1)
                        arr = dset[band_idx, :, :].astype(np.uint16)
                    else:                              # (lines, samples, bands)
                        band_idx = min(target_band_idx, dset.shape[2] - 1)
                        arr = dset[:, :, band_idx].astype(np.uint16)
                elif len(dset.shape) == 2:
                    arr = dset[:, :].astype(np.uint16)
                break

        if arr is None:
            raise ValueError(
                f"Could not locate image data in {h5_path}. "
                f"Available keys: {list(f.keys())}"
            )

    print(f"[INGEST] IIRS band loaded (Hyperslab) - shape: {arr.shape}, dtype: {arr.dtype}")
    return arr, None



def inspect_projection(file_path: Path) -> Dict[str, Any]:
    """
    Check whether a raster is already map-projected.

    Returns
    -------
    dict with keys:
        'is_projected'  : bool   — True if CRS + valid affine transform found
        'crs'           : rasterio.CRS or None
        'transform'     : rasterio.Affine or None
        'width'         : int
        'height'        : int
        'gsd_m'         : float or None  — approximate ground sampling distance in metres
    """
    file_path = Path(file_path)
    result: Dict[str, Any] = {
        'is_projected': False,
        'crs': None,
        'transform': None,
        'width': None,
        'height': None,
        'gsd_m': None,
    }

    try:
        with rasterio.open(file_path) as ds:
            result['width']  = ds.width
            result['height'] = ds.height
            result['crs']    = ds.crs
            result['transform'] = ds.transform

            # A valid projected CRS + non-identity affine = map projected
            t = ds.transform
            identity = (t.a == 1.0 and t.e == -1.0 and t.b == 0 and t.d == 0)
            if ds.crs is not None and not identity:
                result['is_projected'] = True
                # GSD in metres (pixel width)
                result['gsd_m'] = abs(t.a)

    except Exception as e:
        print(f"[INGEST] inspect_projection failed for {file_path}: {e}")

    return result



def extract_pds4_bounds(xml_path: Path) -> Optional[Dict[str, float]]:
    """
    Extract the four corner lat/lon coordinates from a Chandrayaan-2 PDS4 XML
    label's <isda:System_Level_Coordinates> block.

    Returns
    -------
    dict with keys: ul_lat, ul_lon, ur_lat, ur_lon, ll_lat, ll_lon, lr_lat, lr_lon
    or None if the block is not present.
    """
    xml_path = Path(xml_path)
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"[INGEST] XML parse error in {xml_path}: {e}")
        return None

    coords = root.find('.//isda:System_Level_Coordinates', _NS)
    if coords is None:
        print(f"[INGEST] No <isda:System_Level_Coordinates> in {xml_path}")
        return None

    def _get(tag: str) -> float:
        el = coords.find(f'isda:{tag}', _NS)
        if el is None:
            raise KeyError(f"Missing tag <isda:{tag}>")
        return float(el.text.strip())

    try:
        bounds = {
            'ul_lat': _get('upper_left_latitude'),
            'ul_lon': _get('upper_left_longitude'),
            'ur_lat': _get('upper_right_latitude'),
            'ur_lon': _get('upper_right_longitude'),
            'll_lat': _get('lower_left_latitude'),
            'll_lon': _get('lower_left_longitude'),
            'lr_lat': _get('lower_right_latitude'),
            'lr_lon': _get('lower_right_longitude'),
        }
        print(f"[INGEST] Extracted PDS4 bounds: "
              f"lat [{bounds['ll_lat']:.4f} — {bounds['ul_lat']:.4f}], "
              f"lon [{bounds['ul_lon']:.4f} — {bounds['ur_lon']:.4f}]")
        return bounds
    except (KeyError, ValueError) as e:
        print(f"[INGEST] Failed to parse corner coordinates: {e}")
        return None



def extract_pds4_metadata(xml_path: Path) -> Dict[str, Any]:
    """
    Extract essential instrument metadata from a Chandrayaan-2 PDS4 XML label.

    Returns
    -------
    dict with keys:
        'start_time'     : str   — ISO UTC start time (e.g. '2021-09-14T07:12:33.123')
        'exposure_s'     : float — line exposure duration in seconds
        'sensor'         : str   — detected sensor ID ('OHRC', 'TMC', 'TMC_FWD', 'TMC_AFT', 'IIRS')
        'lines'          : int
        'samples'        : int
    """
    xml_path = Path(xml_path)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    meta: Dict[str, Any] = {}

    t = root.find('.//pds:Time_Coordinates/pds:start_date_time', _NS)
    if t is not None:
        meta['start_time'] = t.text.strip().rstrip('Z')

    e = root.find('.//isda:Product_Parameters/isda:line_exposure_duration', _NS)
    if e is not None:
        val  = float(e.text.strip())
        unit = e.attrib.get('unit', 'ms').lower()
        meta['exposure_s'] = val / {'ms': 1e3, 's': 1.0, 'microsec': 1e6}.get(unit, 1e3)

    instr = root.find('.//pds:Observing_System_Component/pds:name', _NS)
    if instr is not None:
        name = instr.text.upper()
        if 'OHRC' in name:
            meta['sensor'] = 'OHRC'
        elif 'TMC' in name:
            # Detect Fore / Aft / Nadir from camera angle tag
            angle_el = root.find('.//isda:camera_angle', _NS)
            if angle_el is not None:
                angle = angle_el.text.strip().upper()
                meta['sensor'] = f'TMC_{angle}' if angle in ('FORE', 'AFT') else 'TMC'
            else:
                meta['sensor'] = 'TMC'
        elif 'IIRS' in name:
            meta['sensor'] = 'IIRS'
        else:
            meta['sensor'] = name

    lines_el   = root.find('.//pds:Array_2D_Image/pds:Axis_Array[pds:axis_name="Line"]/pds:elements', _NS)
    samples_el = root.find('.//pds:Array_2D_Image/pds:Axis_Array[pds:axis_name="Sample"]/pds:elements', _NS)
    if lines_el is not None:
        meta['lines']   = int(lines_el.text.strip())
    if samples_el is not None:
        meta['samples'] = int(samples_el.text.strip())

    return meta



def write_raw_tif(array: np.ndarray, out_path: Path, nodata: int = 0) -> None:
    """
    Write a bare NumPy array to a single-band GeoTIFF with no CRS.

    This is used as an intermediate step before georeferencing so that GDAL
    tools can attach GCPs to a file on disk.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    h, w = array.shape
    with rasterio.open(
        out_path,
        'w',
        driver='GTiff',
        height=h,
        width=w,
        count=1,
        dtype=array.dtype,
        nodata=nodata,
        compress='lzw',
    ) as dst:
        dst.write(array, 1)
    print(f"[INGEST] Wrote raw (unprojected) TIF -> {out_path}")



_LRO_CORNER_CACHE: Dict[str, Dict[str, Any]] = {
    "M171992374CE": {
        "ul_lat": -55.58, "ul_lon": 144.86,
        "ur_lat": -55.62, "ur_lon": 140.31,
        "ll_lat": -67.4,  "ll_lon": 146.5,
        "lr_lat": -67.47, "lr_lon": 139.83,
        "lines": 23712,   "samples": 704
    },
    "M117615312LE": {
        "ul_lat": -69.87, "ul_lon": 329.56,
        "ur_lat": -69.88, "ur_lon": 329.13,
        "ll_lat": -70.76, "ll_lon": 329.41,
        "lr_lat": -70.77, "lr_lon": 328.93,
        "lines": 52224,   "samples": 5064
    },
    "M1179837753RE": {
        "ul_lat": -19.75, "ul_lon": 41.54,
        "ur_lat": -19.75, "ur_lon": 41.41,
        "ll_lat": -20.48, "ll_lon": 41.50,
        "lr_lat": -20.47, "lr_lon": 41.37,
        "lines": 27648,   "samples": 5064
    },
    "M173954190CE": {
        "ul_lat": -64.72, "ul_lon": 205.01,
        "ur_lat": -64.76, "ur_lon": 200.08,
        "ll_lat": -78.77, "ll_lon": 208.99,
        "lr_lat": -78.85, "lr_lon": 198.72,
        "lines": 36036,   "samples": 704
    },
    "M186926223CE": {
        "ul_lat": -68.54, "ul_lon": 202.81,
        "ur_lat": -68.47, "ur_lon": 197.41,
        "ll_lat": -57.57, "ll_lon": 202.77,
        "lr_lat": -57.51, "lr_lon": 198.63,
        "lines": 28392,   "samples": 704
    },
    "M1213227903RE": {
        "ul_lat": -3.79, "ul_lon": 336.57,
        "ur_lat": -3.78, "ur_lon": 336.42,
        "ll_lat": -2.17, "ll_lon": 336.65,
        "lr_lat": -2.16, "lr_lon": 336.49,
        "lines": 52224,  "samples": 5064
    },
    "M1397763342RE": {
        "ul_lat": -68.89, "ul_lon": 342.37,
        "ur_lat": -68.86, "ur_lon": 342.70,
        "ll_lat": -69.36, "ll_lon": 342.66,
        "lr_lat": -69.34, "lr_lon": 343.00,
        "lines": 9216,   "samples": 2532
    }
}


def fetch_lro_corners(product_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve ground-truth footprint corners for an LRO product (ASU or local cache)."""
    import re
    import urllib.request

    pid = product_id.upper().replace("_NORMALIZED", "").replace("_GEOREF", "").replace(".IMG", "")
    if pid in _LRO_CORNER_CACHE:
        return _LRO_CORNER_CACHE[pid].copy()

    url = f"https://wms.lroc.asu.edu/lroc/view_lroc/LRO-L-LROC-2-EDR-V1.0/{pid}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        html = urllib.request.urlopen(req, timeout=10).read().decode("utf-8")
        cell_matches = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>\s*<t[dh][^>]*>(.*?)</t[dh]>", html, re.DOTALL)
        vals = {}
        for k, v in cell_matches:
            k_clean = re.sub(r"<.*?>", "", k).strip().lower()
            v_clean = re.sub(r"<.*?>", "", v).strip()
            try:
                vals[k_clean] = float(v_clean)
            except ValueError:
                pass

        res = {
            "ul_lat": vals["upper left latitude"], "ul_lon": vals["upper left longitude"],
            "ur_lat": vals["upper right latitude"], "ur_lon": vals["upper right longitude"],
            "ll_lat": vals["lower left latitude"], "ll_lon": vals["lower left longitude"],
            "lr_lat": vals["lower right latitude"], "lr_lon": vals["lower right longitude"],
            "lines": int(vals.get("image lines", 0)),
            "samples": int(vals.get("line samples", 0)),
        }
        return res
    except Exception as e:
        print(f"[INGEST] Could not fetch online corners for {pid}: {e}")
        return None


def deinterleave_wac(
    file_path: Path,
    band: int = 7,
    output_dir: Optional[Path] = None
) -> Tuple[Path, Tuple[int, int]]:
    """
    Pure-Python WAC Push-Frame De-interleaver (100% ISIS-free).

    LRO WAC in COLOR mode acquires 7 narrow filter strips simultaneously on each 78-line CCD framelet.
    In raw EDR (.IMG), all 7 bands are concatenated frame-by-frame (e.g. 304 frames * 78 lines = 23,712 lines).
    This creates horizontal "barcode" stripes when viewed as a 2D image.

    This function:
      1. Parses the PDS3 label to check INSTRUMENT_MODE_ID and image geometry.
      2. If COLOR mode (78 lines/frame across 7 filters):
         - Band 1: 321 nm UV (lines 0..3, 4 lines)
         - Band 2: 360 nm UV (lines 4..7, 4 lines)
         - Band 3: 415 nm Blue (lines 8..21, 14 lines)
         - Band 4: 566 nm Green (lines 22..35, 14 lines)
         - Band 5: 604 nm Orange (lines 36..49, 14 lines)
         - Band 6: 643 nm Red 1 (lines 50..63, 14 lines)
         - Band 7: 689 nm Red 2 (lines 64..77, 14 lines) [Default, optimal for TMC/OHRC]
      3. Reads raw byte counts at the byte offset (LABEL_RECORDS * RECORD_BYTES).
      4. Stacks the framelets along-track into a continuous 2D raster of lunar terrain.
      5. Writes an unprojected single-band GeoTIFF.

    Returns
    -------
    (out_raw_tif, (width, height))
    """
    file_path = Path(file_path)
    output_dir = Path(output_dir) if output_dir else file_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    record_bytes = 704
    label_records = 10
    mode = "COLOR"
    lines = 0
    samples = 704
    n_frames = 0

    with open(file_path, "rb") as f:
        header_text = f.read(65536).decode("ascii", errors="ignore")

    for line in header_text.splitlines():
        line = line.strip()
        if "RECORD_BYTES" in line and "=" in line:
            try:
                record_bytes = int(line.split("=")[1].strip())
            except ValueError:
                pass
        elif "LABEL_RECORDS" in line and "=" in line:
            try:
                label_records = int(line.split("=")[1].strip())
            except ValueError:
                pass
        elif "INSTRUMENT_MODE_ID" in line and "=" in line:
            mode = line.split("=")[1].replace('"', '').strip().upper()
        elif "LINES" in line and "=" in line and lines == 0:
            try:
                lines = int(line.split("=")[1].strip())
            except ValueError:
                pass
        elif "LINE_SAMPLES" in line and "=" in line:
            try:
                samples = int(line.split("=")[1].strip())
            except ValueError:
                pass
        elif "LRO:NFRAMES" in line and "=" in line:
            try:
                n_frames = int(line.split("=")[1].strip())
            except ValueError:
                pass

    offset = label_records * record_bytes
    if n_frames == 0 and lines > 0:
        n_frames = lines // (78 if mode == "COLOR" else (70 if mode == "VIS" else 14))

    print(f"[INGEST] WAC EDR Header: Mode={mode}, {n_frames} frames, {lines} lines, {samples} samples (Offset={offset} bytes)")

    band_slices = {
        1: slice(0, 4),
        2: slice(4, 8),
        3: slice(8, 22),
        4: slice(22, 36),
        5: slice(36, 50),
        6: slice(50, 64),
        7: slice(64, 78),
        321: slice(0, 4),
        360: slice(4, 8),
        415: slice(8, 22),
        566: slice(22, 36),
        604: slice(36, 50),
        643: slice(50, 64),
        689: slice(64, 78),
    }

    target_slice = band_slices.get(band, slice(64, 78))  # Default: Band 7 @ 689nm Red
    frame_lines = 78 if mode == "COLOR" else (70 if mode == "VIS" else 14)

    raw = np.fromfile(file_path, dtype=np.uint8, offset=offset)
    expected_size = n_frames * frame_lines * samples
    if len(raw) < expected_size:
        raise ValueError(f"WAC EDR file too short: got {len(raw)} bytes, expected {expected_size}")

    data = raw[:expected_size].reshape(n_frames, frame_lines, samples)

    band_frames = data[:, target_slice, :].astype(np.float32)
    band_h = target_slice.stop - target_slice.start

    row_means = np.mean(band_frames, axis=(0, 2))  # Mean for each row across all frames and samples
    overall_mean = np.mean(row_means)
    if overall_mean > 0:
        p_norm = row_means / overall_mean
        p_norm[p_norm == 0] = 1.0  # Prevent division by zero
        band_frames = band_frames / p_norm.reshape(1, band_h, 1)

    if band_h >= 2 and n_frames > 1:
        last_rows = band_frames[:-1, -1, :].copy()
        first_rows = band_frames[1:, 0, :].copy()
        band_frames[:-1, -1, :] = 0.75 * last_rows + 0.25 * first_rows
        band_frames[1:, 0, :]   = 0.25 * last_rows + 0.75 * first_rows

    deinterleaved = band_frames.reshape(n_frames * band_h, samples)
    deinterleaved = np.clip(deinterleaved, 0, 255).astype(np.uint8)

    import cv2
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    i_clahe = clahe.apply(deinterleaved)
    
    blurred = cv2.GaussianBlur(i_clahe, (0, 0), sigmaX=1.2, sigmaY=1.2)
    i_sharp_f = 1.4 * i_clahe.astype(np.float32) - 0.4 * blurred.astype(np.float32)
    deinterleaved = np.clip(i_sharp_f, 0, 255).astype(np.uint8)

    out_raw_tif = output_dir / f"{file_path.stem}_deinter_b{band}.tif"
    write_raw_tif(deinterleaved, out_raw_tif)
    print(f"[INGEST] [OK] WAC de-interleaved & enhanced: {n_frames} framelets of {band_h} lines -> {deinterleaved.shape} raster ({out_raw_tif.name})")

    return out_raw_tif, (samples, n_frames * band_h)


def ensure_georeferenced(
    file_path: Path,
    sensor: str,
    output_dir: Path,
    force: bool = False,
    wac_band: int = 7
) -> Path:
    """
    Ensure the input raster is map-projected into Equirectangular Moon.
    If already projected, returns the path as-is.
    If unprojected raw PDS4/PDS3, automatically georeferences it into output_dir.
    """
    file_path = Path(file_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    proj = inspect_projection(file_path)
    if proj["is_projected"]:
        return file_path

    sensor_up = sensor.upper()
    stem = file_path.stem
    out_georef = output_dir / f"{stem}_georef.tif"

    if out_georef.exists() and out_georef.stat().st_size > 1024 and not force:
        # Check if cached WAC is the old barcode version (corrupted with >10000 lines instead of ~3973)
        if "WAC" in sensor_up or "CE" in stem.upper():
            try:
                with rasterio.open(out_georef) as check_ds:
                    if check_ds.height > 10000:
                        print(f"[INGEST] Detected obsolete barcode-corrupted WAC cache ({check_ds.height} lines). Regenerating...")
                    else:
                        print(f"[INGEST] Found valid cached georeferenced raster: {out_georef.name}")
                        return out_georef
            except Exception:
                pass
        else:
            print(f"[INGEST] Found cached georeferenced raster: {out_georef.name}")
            return out_georef

    print(f"\n[INGEST] Raw unprojected raster detected ({sensor_up}): {file_path.name}")
    print("         Initiating automated georeferencing to Equirectangular Moon...")

    if any(s in sensor_up for s in ["NAC", "WAC", "LRO"]):
        # LRO georeferencing
        corners = fetch_lro_corners(stem)
        if corners is None:
            raise ValueError(f"Could not retrieve corner coordinates for LRO image: {file_path.name}")

        is_wac = "WAC" in sensor_up or "CE" in stem.upper()
        warp_input_path = file_path

        with rasterio.open(file_path) as ds:
            w, h = ds.width, ds.height

        # De-interleave WAC push-frame if COLOR mode detected (eliminates barcode stripes)
        if is_wac and (h % 78 == 0 or "COLOR" in sensor_up):
            print(f"[INGEST] Detected WAC push-frame COLOR image ({h} lines). De-interleaving into single spectral band...")
            deinter_tif, (w_clean, h_clean) = deinterleave_wac(file_path, band=wac_band, output_dir=output_dir)
            warp_input_path = deinter_tif
            w, h = w_clean, h_clean

        gcps = [
            (0, 0, corners["ul_lon"], corners["ul_lat"]),
            (w, 0, corners["ur_lon"], corners["ur_lat"]),
            (0, h, corners["ll_lon"], corners["ll_lat"]),
            (w, h, corners["lr_lon"], corners["lr_lat"]),
        ]
        from .spice_georeference import _warp_with_gcps
        _warp_with_gcps(warp_input_path, out_georef, gcps, width=w, height=h, poly_order=1)
        print(f"[INGEST] [OK] LRO georeferenced -> {out_georef.name}")
        return out_georef

    else:
        # Chandrayaan-2 (TMC, OHRC, IIRS)
        xml_path = file_path if file_path.suffix.lower() == ".xml" else file_path.with_suffix(".xml")
        if not xml_path.exists():
            xml_path = file_path.with_suffix(".XML")
        if not xml_path.exists():
            raise FileNotFoundError(f"Matching PDS4 XML label not found for {file_path}")

        from .spice_georeference import compute_gcps, apply_gcps_gdal, fallback_4_corner
        import glob
        kernels = glob.glob("data/spice/*.*")
        spice_sensor = "TMC" if "TMC" in sensor_up else ("IIRS" if "IIRS" in sensor_up else "OHRC")

        with rasterio.open(xml_path) as ds:
            w, h = ds.width, ds.height

        step = 5000 if spice_sensor == "TMC" else 500
        gcps = []
        if kernels:
            try:
                gcps = compute_gcps(str(xml_path), width=w, height=h, kernel_paths=kernels, sensor=spice_sensor, step=step)
            except Exception as ex:
                print(f"[INGEST] SPICE ray-tracing error ({ex}), falling back to XML corners.")
                gcps = []

        if len(gcps) > 4:
            apply_gcps_gdal(str(xml_path), str(xml_path), out_georef, gcps)
        else:
            fallback_4_corner(str(xml_path), str(xml_path), str(out_georef))

        print(f"[INGEST] [OK] Chandrayaan-2 {spice_sensor} georeferenced -> {out_georef.name}")
        return out_georef
