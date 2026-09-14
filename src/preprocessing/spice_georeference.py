import os
import sys
import argparse
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import spiceypy as spice
import rasterio

# Standard Equirectangular WKT (Spherical Moon, R=1737400m)
TARGET_CRS_WKT = 'PROJCS["Equirectangular Moon",GEOGCS["GCS_Moon",DATUM["D_Moon",SPHEROID["Moon_LocalRadius",1737400,0]],PRIMEM["Reference_Meridian",0],UNIT["degree",0.0174532925199433]],PROJECTION["Equirectangular"],PARAMETER["standard_parallel_1",0],PARAMETER["central_meridian",0],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1]]'

INSTRUMENT_CONFIG = {
    'OHRC': {
        'frame_id': -152270,
        'frame_name': 'CH2_OHRC',
        'pds_time_tag': './/pds:Time_Coordinates/pds:start_date_time',
        'pds_exposure_tag': './/isda:Product_Parameters/isda:line_exposure_duration'
    },
    'TMC': {
        'frame_id': -152210,
        'frame_name': 'CH2_TMC_NADIR',
        'pds_time_tag': './/pds:Time_Coordinates/pds:start_date_time',
        'pds_exposure_tag': './/isda:Product_Parameters/isda:line_exposure_duration'
    },
    'IIRS': {
        'frame_id': -152240,
        'frame_name': 'CH2_IIR_I',
        'pds_time_tag': './/pds:Time_Coordinates/pds:start_date_time',
        'pds_exposure_tag': './/isda:Product_Parameters/isda:line_exposure_duration'
    }
}


def parse_label(xml_path, sensor):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    ns = {
        'pds': 'http://pds.nasa.gov/pds4/pds/v1',
        'isda': 'https://isda.issdc.gov.in/pds4/isda/v1'
    }
    
    config = INSTRUMENT_CONFIG[sensor]
    start_time_elem = root.find(config['pds_time_tag'], ns)
    if start_time_elem is None:
        raise ValueError(f"Could not find start time in {xml_path}")
    start_time_str = start_time_elem.text.strip().replace('Z', '')
    
    exposure_elem = root.find(config['pds_exposure_tag'], ns)
    if exposure_elem is None:
        raise ValueError(f"Could not find exposure duration in {xml_path}")
    
    unit = exposure_elem.attrib.get('unit', 'ms')
    exposure_val = float(exposure_elem.text)
    if unit == 'ms':
        exposure_s = exposure_val / 1000.0
    elif unit == 's':
        exposure_s = exposure_val
    elif unit == 'microsec':
        exposure_s = exposure_val / 1000000.0
    else:
        exposure_s = exposure_val / 1000.0
        
    return {
        'start_time': start_time_str,
        'exposure_s': exposure_s
    }


def fallback_4_corner(xml_file, raw_tif, out_tif):
    print("[FALLBACK] Extracting 4 corners from XML for Affine Georeferencing...")
    tree = ET.parse(xml_file)
    root = tree.getroot()
    ns = {'isda': 'https://isda.issdc.gov.in/pds4/isda/v1'}
    
    coords = root.find('.//isda:System_Level_Coordinates', ns)
    if coords is None:
        print("[ERROR] Fallback failed: System_Level_Coordinates not found in XML.")
        return False
        
    try:
        ul_lat = float(coords.find('isda:upper_left_latitude', ns).text)
        ul_lon = float(coords.find('isda:upper_left_longitude', ns).text)
        ur_lat = float(coords.find('isda:upper_right_latitude', ns).text)
        ur_lon = float(coords.find('isda:upper_right_longitude', ns).text)
        ll_lat = float(coords.find('isda:lower_left_latitude', ns).text)
        ll_lon = float(coords.find('isda:lower_left_longitude', ns).text)
        lr_lat = float(coords.find('isda:lower_right_latitude', ns).text)
        lr_lon = float(coords.find('isda:lower_right_longitude', ns).text)
    except Exception as e:
        print(f"[ERROR] Failed to parse corner coordinates: {e}")
        return False
        
    try:
        with rasterio.open(xml_file) as src:
            height, width = src.shape
    except Exception:
        with rasterio.open(raw_tif) as src:
            height, width = src.shape
        
    temp_gcp_tif = str(out_tif).replace(".tif", "_gcp.tif")
    
    cmd_translate = [
        'gdal_translate',
        '-a_srs', TARGET_CRS_WKT,
        '-gcp', '0', '0', str(ul_lon), str(ul_lat),
        '-gcp', str(width), '0', str(ur_lon), str(ur_lat),
        '-gcp', '0', str(height), str(ll_lon), str(ll_lat),
        '-gcp', str(width), str(height), str(lr_lon), str(lr_lat),
        str(xml_file), temp_gcp_tif
    ]
    print(f"Running: {' '.join(cmd_translate)}")
    subprocess.run(cmd_translate, check=True)
    
    cmd_warp = [
        'gdalwarp',
        '-order', '1',
        '-t_srs', TARGET_CRS_WKT,
        '-r', 'bilinear',
        temp_gcp_tif, str(out_tif)
    ]
    print(f"Running: {' '.join(cmd_warp)}")
    subprocess.run(cmd_warp, check=True)
    
    if os.path.exists(temp_gcp_tif):
        os.remove(temp_gcp_tif)
        
    print(f"[SUCCESS] Wrote fallback georeferenced file to {out_tif}")
    return True


def compute_gcps(xml_path, width, height, kernel_paths, sensor, step=100):
    label_info = parse_label(xml_path, sensor)
    print("\n[SPICE] Loading Kernels...")
    for k in kernel_paths:
        if os.path.exists(k):
            print(f"  -> {k}")
            spice.furnsh(k)
        else:
            print(f"  -> [WARNING] Kernel not found: {k}")
        
    config = INSTRUMENT_CONFIG[sensor]
    inst_id = config['frame_id']
    inst_frame = config['frame_name']
    
    print(f"[SPICE] Getting FOV for {inst_frame} (ID: {inst_id})")
    try:
        shape, frame, bsight, n, bounds = spice.getfov(inst_id, 4)
    except spice.utils.support_types.SpiceyError as e:
        print(f"[ERROR] Failed to get FOV for {inst_frame}: {e}")
        return []
        
    start_et = spice.str2et(label_info['start_time'])
    exposure_s = label_info['exposure_s']
    
    v0, v1, v2, v3 = bounds
    dist01 = np.linalg.norm(v0 - v1)
    dist12 = np.linalg.norm(v1 - v2)
    
    if dist01 > dist12:
        left_vec = (v0 + v3) / 2.0
        right_vec = (v1 + v2) / 2.0
    else:
        left_vec = (v0 + v1) / 2.0
        right_vec = (v2 + v3) / 2.0
        
    left_vec = left_vec / np.linalg.norm(left_vec)
    right_vec = right_vec / np.linalg.norm(right_vec)
    
    gcps = []
    lines = np.arange(0, height, step)
    if lines[-1] != height - 1:
        lines = np.append(lines, height - 1)
        
    samples = np.arange(0, width, step)
    if samples[-1] != width - 1:
        samples = np.append(samples, width - 1)
        
    print(f"[SPICE] Computing ray intersections for {len(lines)*len(samples)} grid points...")
    pts_found = 0
    pts_missed = 0
    
    for line in lines:
        et = start_et + (line * exposure_s)
        for sample in samples:
            frac = sample / float(width - 1)
            look_vec = left_vec * (1 - frac) + right_vec * frac
            look_vec = look_vec / np.linalg.norm(look_vec)
            
            try:
                point, trgepc, srfvec = spice.sincpt(
                    method="ELLIPSOID",
                    target="MOON",
                    et=et,
                    fixref="MOON_ME",
                    abcorr="LT+S",
                    obsrvr="CHANDRAYAAN-2",
                    dref=inst_frame,
                    dvec=look_vec
                )
                radii, lon, lat = spice.reclat(point)
                lon_deg = np.degrees(lon)
                lat_deg = np.degrees(lat)
                if lon_deg < 0:
                    lon_deg += 360
                gcps.append((sample, line, lon_deg, lat_deg))
                pts_found += 1
            except spice.utils.support_types.SpiceyError:
                pts_missed += 1
                
    print(f"[SPICE] Ray tracing complete. Found {pts_found} intercepts, {pts_missed} missed.")
    spice.kclear()
    return gcps


def apply_gcps_gdal(xml_file, raw_tif, out_tif, gcps):
    temp_gcp_tif = str(out_tif).replace(".tif", "_gcp.tif")
    gcp_args = []
    for (sample, line, lon, lat) in gcps:
        gcp_args.extend(["-gcp", str(sample), str(line), str(lon), str(lat)])
        
    cmd_translate = ["gdal_translate", "-a_srs", TARGET_CRS_WKT] + gcp_args + [str(raw_tif), temp_gcp_tif]
    print("[GDAL] Writing GCPs to temporary file...")
    subprocess.run(cmd_translate, check=True)
    
    print("[GDAL] Warping with Thin-Plate Splines (-tps)...")
    cmd_warp = ["gdalwarp", "-tps", "-t_srs", TARGET_CRS_WKT, "-r", "bilinear", temp_gcp_tif, str(out_tif)]
    subprocess.run(cmd_warp, check=True)
    
    if os.path.exists(temp_gcp_tif):
        os.remove(temp_gcp_tif)
    print(f"[SUCCESS] Wrote georeferenced file to {out_tif}")


def main():
    parser = argparse.ArgumentParser(description="SPICE Ray-Tracing Georeferencer for Chandrayaan-2 Optical Data")
    parser.add_argument("--xml", required=True, help="Path to PDS4 XML label")
    parser.add_argument("--raw_tif", required=True, help="Path to raw image TIF (unprojected)")
    parser.add_argument("--out_tif", required=True, help="Path to output map-projected TIF")
    parser.add_argument("--sensor", required=True, choices=['OHRC', 'TMC', 'IIRS'])
    parser.add_argument("--kernels", nargs='*', help="List of paths to SPICE kernels or meta-kernel")
    args = parser.parse_args()
    
    try:
        with rasterio.open(args.xml) as src:
            width, height = src.width, src.height
    except Exception:
        with rasterio.open(args.raw_tif) as src:
            width, height = src.width, src.height
        
    if not args.kernels:
        fallback_4_corner(args.xml, args.raw_tif, args.out_tif)
        return
        
    gcps = compute_gcps(args.xml, width, height, args.kernels, args.sensor)
    if len(gcps) == 0:
        print("[ERROR] SPICE Ray tracing failed entirely. Falling back to 4-corner method...")
        fallback_4_corner(args.xml, args.raw_tif, args.out_tif)
    else:
        apply_gcps_gdal(args.xml, args.raw_tif, args.out_tif, gcps)


if __name__ == "__main__":
    main()
