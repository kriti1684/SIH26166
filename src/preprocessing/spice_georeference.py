import os
import sys
import argparse
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import spiceypy as spice
import rasterio

# ---------------------------------------------------------------------------
# CRS definitions
# ---------------------------------------------------------------------------
# Geographic Moon CRS (lat/lon in degrees) — used as the SOURCE CRS when
# assigning corner GCPs.  All ground-truth coordinates (from XML, PDS, ODE)
# are in degrees, so this is the correct GCP SRS.
GEOG_MOON_WKT = (
    'GEOGCS["GCS_Moon",'
    'DATUM["D_Moon",SPHEROID["Moon_LocalRadius",1737400,0]],'
    'PRIMEM["Reference_Meridian",0],'
    'UNIT["degree",0.0174532925199433]]'
)

# Equirectangular Moon (projected, metres) — the standard target CRS for
# all output GeoTIFFs in this pipeline.
TARGET_CRS_WKT = (
    'PROJCS["Equirectangular Moon",'
    'GEOGCS["GCS_Moon",DATUM["D_Moon",SPHEROID["Moon_LocalRadius",1737400,0]],'
    'PRIMEM["Reference_Meridian",0],UNIT["degree",0.0174532925199433]],'
    'PROJECTION["Equirectangular"],'
    'PARAMETER["standard_parallel_1",0],'
    'PARAMETER["central_meridian",0],'
    'PARAMETER["false_easting",0],'
    'PARAMETER["false_northing",0],'
    'UNIT["metre",1]]'
)

INSTRUMENT_CONFIG = {
    'OHRC': {
        'frame_id':   -152270,
        'frame_name': 'CH2_OHRC',
        'pds_time_tag':     './/pds:Time_Coordinates/pds:start_date_time',
        'pds_exposure_tag': './/isda:Product_Parameters/isda:line_exposure_duration',
    },
    'TMC': {
        'frame_id':   -152210,
        'frame_name': 'CH2_TMC_NADIR',
        'pds_time_tag':     './/pds:Time_Coordinates/pds:start_date_time',
        'pds_exposure_tag': './/isda:Product_Parameters/isda:line_exposure_duration',
    },
    'IIRS': {
        'frame_id':   -152240,
        'frame_name': 'CH2_IIR_I',
        'pds_time_tag':     './/pds:Time_Coordinates/pds:start_date_time',
        'pds_exposure_tag': './/isda:Product_Parameters/isda:line_exposure_duration',
    },
}


def parse_label(xml_path, sensor):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    ns = {
        'pds':  'http://pds.nasa.gov/pds4/pds/v1',
        'isda': 'https://isda.issdc.gov.in/pds4/isda/v1',
    }
    config = INSTRUMENT_CONFIG[sensor]
    start_time_elem = root.find(config['pds_time_tag'], ns)
    if start_time_elem is None:
        raise ValueError(f"Could not find start time in {xml_path}")
    start_time_str = start_time_elem.text.strip().replace('Z', '')

    stop_time_elem = root.find('.//pds:Time_Coordinates/pds:stop_date_time', ns)
    stop_time_str  = stop_time_elem.text.strip().replace('Z', '') if stop_time_elem is not None else None

    exposure_elem = root.find(config['pds_exposure_tag'], ns)
    if exposure_elem is None:
        raise ValueError(f"Could not find exposure duration in {xml_path}")
    unit         = exposure_elem.attrib.get('unit', 'ms').lower()
    exposure_val = float(exposure_elem.text)
    if unit in ('microsec', 'us', '\u00b5s') or (unit == 'ms' and exposure_val > 10.0):
        exposure_s = exposure_val / 1_000_000.0
    elif unit == 'ms':
        exposure_s = exposure_val / 1000.0
    elif unit == 's':
        exposure_s = exposure_val
    else:
        exposure_s = exposure_val / 1_000_000.0

    return {'start_time': start_time_str, 'stop_time': stop_time_str, 'exposure_s': exposure_s}


# ---------------------------------------------------------------------------
# GCP-based warp helpers (shared by fallback AND SPICE paths)
# ---------------------------------------------------------------------------

def _warp_with_gcps(raw_tif: Path, out_tif: Path, gcps_geo: list, width: int, height: int,
                    poly_order: int = 1, num_threads: int = 0):
    """
    Assign geographic GCPs to raw_tif via a VRT, then warp to TARGET_CRS_WKT.

    gcps_geo : list of (pixel_col, pixel_row, lon_deg, lat_deg)
    num_threads : 0 = ALL_CPUS
    """
    from osgeo import gdal

    def _wrap_lon(l):
        l = float(l)
        while l > 180.0:
            l -= 360.0
        while l < -180.0:
            l += 360.0
        return l

    gdal_gcps = [
        gdal.GCP(_wrap_lon(lon), float(lat), 0.0, float(col), float(row))
        for (col, row, lon, lat) in gcps_geo
    ]

    temp_vrt = str(out_tif).replace('.tif', '_gcp.vrt')
    ds = gdal.Open(str(raw_tif))
    if ds is None:
        raise RuntimeError(f"GDAL could not open {raw_tif}")

    # Assign GCPs in GEOGRAPHIC Moon CRS (degrees).
    # This is the critical fix: the GCP coordinates are in lat/lon degrees,
    # so outputSRS must be the GEOGRAPHIC CRS, not the projected (metre) CRS.
    vrt_ds = gdal.Translate(
        temp_vrt, ds,
        format='VRT',
        outputSRS=GEOG_MOON_WKT,   # <-- FIXED: was TARGET_CRS_WKT (metres)
        GCPs=gdal_gcps,
    )
    vrt_ds = ds = None

    threads_opt = 'ALL_CPUS' if num_threads == 0 else str(num_threads)
    warp_opts = gdal.WarpOptions(
        format='GTiff',
        srcSRS=GEOG_MOON_WKT,       # source CRS: geographic (degrees)
        dstSRS=TARGET_CRS_WKT,      # dest CRS: equirectangular (metres)
        polynomialOrder=poly_order,
        resampleAlg=gdal.GRA_Bilinear,
        creationOptions=[
            'COMPRESS=DEFLATE', 'PREDICTOR=2',
            'TILED=YES', 'BLOCKXSIZE=512', 'BLOCKYSIZE=512',
            'BIGTIFF=YES',
        ],
        warpOptions=[f'NUM_THREADS={threads_opt}'],
        multithread=True,
    )
    warped_ds = gdal.Warp(str(out_tif), temp_vrt, options=warp_opts)
    warped_ds = None

    if os.path.exists(temp_vrt):
        os.remove(temp_vrt)

    if not out_tif.exists() or out_tif.stat().st_size == 0:
        raise RuntimeError(f"gdalwarp produced empty output: {out_tif}")


def fallback_4_corner(xml_file, raw_tif, out_tif):
    """
    Extract 4 corners from the ISDA XML and georeference using a first-order
    polynomial warp to Equirectangular Moon (metres).
    """
    print("[FALLBACK] Extracting 4 corners from XML...")
    tree = ET.parse(xml_file)
    root = tree.getroot()
    ns   = {'isda': 'https://isda.issdc.gov.in/pds4/isda/v1'}

    coords = root.find('.//isda:System_Level_Coordinates', ns)
    if coords is None:
        print("[ERROR] System_Level_Coordinates not found in XML.")
        return False

    try:
        ul_lat = float(coords.find('isda:upper_left_latitude',  ns).text)
        ul_lon = float(coords.find('isda:upper_left_longitude', ns).text)
        ur_lat = float(coords.find('isda:upper_right_latitude',  ns).text)
        ur_lon = float(coords.find('isda:upper_right_longitude', ns).text)
        ll_lat = float(coords.find('isda:lower_left_latitude',  ns).text)
        ll_lon = float(coords.find('isda:lower_left_longitude', ns).text)
        lr_lat = float(coords.find('isda:lower_right_latitude',  ns).text)
        lr_lon = float(coords.find('isda:lower_right_longitude', ns).text)
    except Exception as e:
        print(f"[ERROR] Failed to parse corner coordinates: {e}")
        return False

    try:
        with rasterio.open(raw_tif) as src:
            height, width = src.height, src.width
    except Exception:
        print(f"[ERROR] Cannot open {raw_tif} to get dimensions.")
        return False

    gcps = [
        (0,     0,      ul_lon, ul_lat),
        (width, 0,      ur_lon, ur_lat),
        (0,     height, ll_lon, ll_lat),
        (width, height, lr_lon, lr_lat),
    ]

    try:
        _warp_with_gcps(Path(raw_tif), Path(out_tif), gcps, width, height, poly_order=1)
        print(f"[FALLBACK] Wrote georeferenced file to {out_tif}")
        return True
    except Exception as e:
        print(f"[ERROR] Warp failed: {e}")
        return False


# ---------------------------------------------------------------------------
# SPICE ray-tracing
# ---------------------------------------------------------------------------

VALID_SPICE_EXTS = ('.tls', '.tpc', '.bsp', '.tsc', '.bc', '.tf', '.ti', '.tm', '.bpc')


def compute_gcps(xml_path, width, height, kernel_paths, sensor, step=100):
    label_info = parse_label(xml_path, sensor)
    print("\n[SPICE] Loading kernels...")
    loaded = 0
    for k in kernel_paths:
        if os.path.exists(k) and any(k.lower().endswith(ext) for ext in VALID_SPICE_EXTS):
            print(f"  -> {Path(k).name}")
            spice.furnsh(k)
            loaded += 1
        elif not os.path.exists(k):
            print(f"  -> [WARN] Not found: {k}")
    print(f"[SPICE] Loaded {loaded} kernels.")

    config    = INSTRUMENT_CONFIG[sensor]
    inst_id   = config['frame_id']
    inst_frame = config['frame_name']

    print(f"[SPICE] Getting FOV for {inst_frame} (ID: {inst_id})")
    try:
        shape, frame, bsight, n, bounds = spice.getfov(inst_id, 4)
    except spice.utils.support_types.SpiceyError as e:
        print(f"[ERROR] getfov failed for {inst_frame}: {e}")
        return []

    start_et = spice.str2et(label_info['start_time'])
    if label_info.get('stop_time') and height > 1:
        stop_et   = spice.str2et(label_info['stop_time'])
        exposure_s = (stop_et - start_et) / float(height - 1)
    else:
        exposure_s = label_info['exposure_s']

    v0, v1, v2, v3 = bounds
    dist01 = np.linalg.norm(v0 - v1)
    dist12 = np.linalg.norm(v1 - v2)
    if dist01 > dist12:
        left_vec  = (v0 + v3) / 2.0
        right_vec = (v1 + v2) / 2.0
    else:
        left_vec  = (v0 + v1) / 2.0
        right_vec = (v2 + v3) / 2.0
    left_vec  /= np.linalg.norm(left_vec)
    right_vec /= np.linalg.norm(right_vec)

    lines   = np.arange(0, height, step)
    if lines[-1] != height - 1:
        lines = np.append(lines, height - 1)
    samples = np.arange(0, width, step)
    if samples[-1] != width - 1:
        samples = np.append(samples, width - 1)

    fixref = "MOON_ME" if spice.namfrm("MOON_ME") != 0 else "IAU_MOON"
    print(f"[SPICE] Ray-tracing {len(lines)*len(samples)} grid points "
          f"(body frame: {fixref})...")
    gcps = []
    pts_found = pts_missed = 0
    first_error = None

    for line in lines:
        et = start_et + (line * exposure_s)
        for sample in samples:
            frac     = sample / float(width - 1)
            look_vec = (left_vec * (1 - frac) + right_vec * frac)
            look_vec /= np.linalg.norm(look_vec)
            try:
                point, _, _ = spice.sincpt(
                    "ELLIPSOID", "MOON", et, fixref,
                    "LT+S", "CHANDRAYAAN-2", inst_frame, look_vec,
                )
                _, lon, lat = spice.reclat(point)
                lon_deg = np.degrees(lon)
                lat_deg = np.degrees(lat)
                while lon_deg > 180.0:
                    lon_deg -= 360.0
                while lon_deg < -180.0:
                    lon_deg += 360.0
                gcps.append((sample, line, lon_deg, lat_deg))
                pts_found += 1
            except spice.utils.support_types.SpiceyError as err:
                pts_missed += 1
                if first_error is None:
                    first_error = str(err)
                    print(f"[SPICE] First intercept error: {first_error[:120]}")

    print(f"[SPICE] Done: {pts_found} found, {pts_missed} missed.")
    spice.kclear()
    return gcps


def apply_gcps_gdal(xml_file, raw_tif, out_tif, gcps):
    """
    Apply SPICE-computed GCPs (col, row, lon_deg, lat_deg) and warp to
    TARGET_CRS_WKT using TPS when >4 GCPs, poly-1 otherwise.
    """
    from osgeo import gdal

    # Use TPS for many GCPs (SPICE dense grid), poly-1 for only corners
    use_tps = len(gcps) > 6

    def _wrap_lon(l):
        l = float(l)
        while l > 180.0:
            l -= 360.0
        while l < -180.0:
            l += 360.0
        return l

    gdal_gcps = [
        gdal.GCP(_wrap_lon(lon), float(lat), 0.0, float(sample), float(line))
        for (sample, line, lon, lat) in gcps
    ]

    temp_vrt = str(out_tif).replace('.tif', '_gcp.vrt')
    print(f"[GDAL] Creating VRT with {len(gdal_gcps)} SPICE GCPs...")
    ds = gdal.Open(str(raw_tif))
    vrt_ds = gdal.Translate(
        temp_vrt, ds,
        format='VRT',
        outputSRS=GEOG_MOON_WKT,   # GCPs are in lat/lon degrees
        GCPs=gdal_gcps,
    )
    vrt_ds = ds = None

    print(f"[GDAL] Warping (TPS={use_tps})...")
    if use_tps:
        warp_opts = gdal.WarpOptions(
            format='GTiff',
            srcSRS=GEOG_MOON_WKT,
            dstSRS=TARGET_CRS_WKT,
            tps=True,
            resampleAlg=gdal.GRA_Bilinear,
            creationOptions=['COMPRESS=DEFLATE', 'TILED=YES', 'BIGTIFF=YES'],
            warpOptions=['NUM_THREADS=ALL_CPUS'],
            multithread=True,
        )
    else:
        warp_opts = gdal.WarpOptions(
            format='GTiff',
            srcSRS=GEOG_MOON_WKT,
            dstSRS=TARGET_CRS_WKT,
            polynomialOrder=1,
            resampleAlg=gdal.GRA_Bilinear,
            creationOptions=['COMPRESS=DEFLATE', 'TILED=YES', 'BIGTIFF=YES'],
            warpOptions=['NUM_THREADS=ALL_CPUS'],
            multithread=True,
        )

    warped_ds = gdal.Warp(str(out_tif), temp_vrt, options=warp_opts)
    warped_ds = None

    if os.path.exists(temp_vrt):
        os.remove(temp_vrt)
    print(f"[SUCCESS] Georeferenced -> {out_tif}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="SPICE Ray-Tracing Georeferencer for Chandrayaan-2 Optical Data"
    )
    parser.add_argument("--xml",     required=True, help="PDS4 XML label path")
    parser.add_argument("--raw_tif", required=True, help="Raw unprojected TIF")
    parser.add_argument("--out_tif", required=True, help="Output georeferenced TIF")
    parser.add_argument("--sensor",  required=True, choices=['OHRC', 'TMC', 'IIRS'])
    parser.add_argument("--kernels", nargs='*',     help="SPICE kernel paths")
    args = parser.parse_args()

    try:
        with rasterio.open(args.raw_tif) as src:
            width, height = src.width, src.height
    except Exception:
        with rasterio.open(args.xml) as src:
            width, height = src.width, src.height

    if not args.kernels:
        fallback_4_corner(args.xml, args.raw_tif, args.out_tif)
        return

    gcps = compute_gcps(args.xml, width, height, args.kernels, args.sensor)
    if len(gcps) == 0:
        print("[ERROR] SPICE ray-tracing failed. Falling back to 4-corner method...")
        fallback_4_corner(args.xml, args.raw_tif, args.out_tif)
    else:
        apply_gcps_gdal(args.xml, args.raw_tif, args.out_tif, gcps)


if __name__ == "__main__":
    main()
