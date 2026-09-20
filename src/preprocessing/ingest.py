"""
src/preprocessing/ingest.py
============================
Pure-Python Ingestion Engine — 100% ISIS-free.

Replaces:
  - lronac2isis / lrowac2isis / isisimport (ISIS3 commands)
  - process_lro() and process_ch2_optical() WSL-dependent functions in pipeline.py

Supports:
  - PDS4 format  : Chandrayaan-2 OHRC / TMC-2 / IIRS  (.xml + .img / .h5)
  - PDS3 format  : LRO NAC / WAC                       (.IMG)
  - GeoTIFF      : Any already map-projected image      (.tif / .tiff)

Usage:
  from src.preprocessing.ingest import load_raster, inspect_projection, extract_pds4_bounds
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
import rasterio
from rasterio.crs import CRS


# ── Namespaces used in ISRO PDS4 XML labels ─────────────────────────────────
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


# ── 1. Raster loader ─────────────────────────────────────────────────────────

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

    # HDF5 (IIRS hyperspectral) ─────────────────────────────────────────────
    if suffix in ('.h5', '.hdf5'):
        return _load_iirs_h5(file_path)

    # PDS4 XML / raw image binary loader ──────────────────────────────────
    xml_path = file_path if suffix == '.xml' else file_path.with_suffix('.xml')
    if not xml_path.exists():
        xml_path = file_path.with_suffix('.XML')

    if xml_path.exists():
        # Try direct rasterio open first
        try:
            ds = rasterio.open(file_path)
            arr = ds.read(1)
            return arr, ds
        except Exception:
            pass

        # Native PDS4 XML + binary loader
        try:
            return _load_pds4_raw(xml_path, file_path if suffix != '.xml' else None)
        except Exception as e:
            print(f"[INGEST] Native PDS4 loader warning: {e}")

    # PDS3 / standard raster loader via rasterio
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

    # If image path not provided, search in same folder
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


def _load_iirs_h5(h5_path: Path) -> Tuple[np.ndarray, None]:
    """
    Load a single high-contrast optical band from an IIRS HDF5 file.

    IIRS has ~256 spectral bands at 8–20 m GSD.
    We pick Band 50 (~750 nm — peak reflectance, good terrain contrast)
    for use in image co-registration.

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
                raw = f[key][()]
                # raw shape is (bands, lines, samples) or (lines, samples, bands)
                if raw.ndim == 3:
                    if raw.shape[0] < raw.shape[1]:  # (bands, lines, samples)
                        band_idx = min(50, raw.shape[0] - 1)
                        arr = raw[band_idx, :, :].astype(np.uint16)
                    else:                              # (lines, samples, bands)
                        band_idx = min(50, raw.shape[2] - 1)
                        arr = raw[:, :, band_idx].astype(np.uint16)
                elif raw.ndim == 2:
                    arr = raw.astype(np.uint16)
                break

        if arr is None:
            raise ValueError(
                f"Could not locate image data in {h5_path}. "
                f"Available keys: {list(f.keys())}"
            )

    print(f"[INGEST] IIRS band loaded — shape: {arr.shape}, dtype: {arr.dtype}")
    return arr, None


# ── 2. Projection inspector ──────────────────────────────────────────────────

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


# ── 3. PDS4 XML corner-coordinate extractor ──────────────────────────────────

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


# ── 4. PDS4 metadata parser ──────────────────────────────────────────────────

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

    # Start time
    t = root.find('.//pds:Time_Coordinates/pds:start_date_time', _NS)
    if t is not None:
        meta['start_time'] = t.text.strip().rstrip('Z')

    # Exposure duration
    e = root.find('.//isda:Product_Parameters/isda:line_exposure_duration', _NS)
    if e is not None:
        val  = float(e.text.strip())
        unit = e.attrib.get('unit', 'ms').lower()
        meta['exposure_s'] = val / {'ms': 1e3, 's': 1.0, 'microsec': 1e6}.get(unit, 1e3)

    # Sensor / instrument ID
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

    # Image dimensions
    lines_el   = root.find('.//pds:Array_2D_Image/pds:Axis_Array[pds:axis_name="Line"]/pds:elements', _NS)
    samples_el = root.find('.//pds:Array_2D_Image/pds:Axis_Array[pds:axis_name="Sample"]/pds:elements', _NS)
    if lines_el is not None:
        meta['lines']   = int(lines_el.text.strip())
    if samples_el is not None:
        meta['samples'] = int(samples_el.text.strip())

    return meta


# ── 5. Convenience: write an un-georeferenced raster to GeoTIFF ──────────────

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
