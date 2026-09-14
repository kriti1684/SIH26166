import argparse
import subprocess
import sys
from pathlib import Path
import os
import rasterio
import numpy as np


def to_wsl_path(win_path):
    path_str = str(win_path).replace("\\", "/")
    if path_str[1:3] == ":/":
        drive = path_str[0].lower()
        path_str = f"/mnt/{drive}/" + path_str[3:]
    return path_str


def extract_metadata(file_path):
    """Extract basic metadata (StartTime, InstrumentId, is_mapped) from .cub, .IMG, or .xml."""
    metadata = {}
    file_str = str(file_path).lower()
    
    try:
        if file_str.endswith(".xml"):
            import xml.etree.ElementTree as ET
            tree = ET.parse(file_path)
            root = tree.getroot()
            ns = {'pds': 'http://pds.nasa.gov/pds4/pds/v1'}
            time_elem = root.find('.//pds:start_date_time', ns)
            if time_elem is not None: metadata['STARTTIME'] = time_elem.text.strip().replace('Z', '')
        else:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                header = f.read(100000)
                metadata['is_mapped'] = 'Group = Mapping' in header or 'GROUP = MAPPING' in header
                for line in header.split('\n'):
                    line = line.strip()
                    if '=' in line:
                        key, val = line.split('=', 1)
                        key = key.strip().upper()
                        val = val.strip().split('<')[0].strip().replace('"', '')
                        if key in ['INSTRUMENT_ID', 'INSTRUMENTID', 'START_TIME', 'STARTTIME']:
                            metadata[key.replace('_', '')] = val
    except Exception as e:
        print(f"Error extracting metadata from {file_path}: {e}")
        
    return metadata


def run_command(cmd, shell=False):
    # Check if this is an ISIS/ALE command running on Windows
    isis_commands = {"isisimport", "spiceinit", "isd_generate", "csminit", "maptemplate", "cam2map", "lronac2isis", "lrowac2isis", "downloadIsisData", "gdal_translate"}
    if os.name == 'nt' and cmd[0] in isis_commands:
        wsl_cmd_parts = []
        for arg in cmd:
            if "=" in arg:
                key, val = arg.split("=", 1)
                wsl_cmd_parts.append(f"{key}='{to_wsl_path(val)}'")
            else:
                wsl_cmd_parts.append(f"'{to_wsl_path(arg)}'")
        
        inner_cmd = " ".join(wsl_cmd_parts)
        wsl_cmd = ["wsl", "-d", "Ubuntu-24.04", "-e", "bash", "-ic", f"mamba run -n ch2_isis_dev env ISISDATA=/home/kritikriti/isisdata_ch2 ALESPICEROOT=/home/kritikriti/isisdata_ch2 {inner_cmd}"]
        
        print(f"\n[RUNNING in WSL]: {' '.join(wsl_cmd)}")
        try:
            subprocess.run(wsl_cmd, check=True)
            return
        except subprocess.CalledProcessError as e:
            print(f"WSL Command failed with error code: {e.returncode}")
            sys.exit(e.returncode)

    print(f"\n[RUNNING]: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    try:
        subprocess.run(cmd, check=True, shell=shell)
    except FileNotFoundError as e:
        print(f"[WARNING] Command not found: {cmd[0]}")
        raise e
    except subprocess.CalledProcessError as e:
        print(f"Command failed with error code: {e.returncode}")
        sys.exit(e.returncode)


def normalize_tiff(input_tif, output_tif, is_ohrc=False):
    """Normalize TIFF to uint8 0-255, preserving NoData=0."""
    print(f"\n[NORMALIZING]: {input_tif} -> {output_tif}")
    with rasterio.open(input_tif) as src:
        profile = src.profile.copy()
        # Ensure we only have 1 band
        data = src.read(1)
        
        if is_ohrc:
            # OHRC is usually already uint8, just copy and set NoData
            data = data.astype(np.uint8)
            profile.update(driver="GTiff", dtype="uint8", count=1, nodata=0)
            with rasterio.open(output_tif, "w", **profile) as dst:
                dst.write(data, 1)
        else:
            # Float/16-bit to uint8 scaling
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
            
            if src_max <= src_min:
                # Flat image
                normalized = np.zeros(data.shape, dtype=np.uint8)
            else:
                normalized = np.zeros(data.shape, dtype=np.uint8)
                scaled = ((data[valid] - src_min) / (src_max - src_min) * 255.0)
                normalized[valid] = np.clip(np.rint(scaled), 0, 255).astype(np.uint8)
                
            profile.update(driver="GTiff", dtype="uint8", count=1, nodata=0)
            with rasterio.open(output_tif, "w", **profile) as dst:
                dst.write(normalized, 1)



def process_lro(input_file, sensor, reference_cub, output_dir):
    """LRO Branch: NAC or WAC"""
    stem = input_file.stem
    norm_tif = output_dir / f"{stem}_normalized.tif"
    temp_cub = output_dir / f"{stem}_temp.cub"
    map_cub = output_dir / f"{stem}_map.cub"
    raw_tif = output_dir / f"{stem}_raw.tif"
    
    ingest_cmd = "lronac2isis" if sensor == "NAC" else "lrowac2isis"
    
    try:
        run_command([ingest_cmd, f"from={input_file}", f"to={temp_cub}"])
        run_command(["spiceinit", f"from={temp_cub}", "web=true", "shape=ellipsoid"])
        
        if reference_cub:
            run_command(["cam2map", f"from={temp_cub}", f"to={map_cub}", f"map={reference_cub}", "matchmap=true"])
        else:
            run_command(["cam2map", f"from={temp_cub}", f"to={map_cub}", "pixres=mpp", "resolution=2.0"])
        run_command(["gdal_translate", str(map_cub), str(raw_tif)])
        
        normalize_tiff(raw_tif, norm_tif, is_ohrc=False)
        
        # Cache the results
        if 'cache_dir' in globals():
            import shutil
            cache_path_cub = cache_dir / f"{sensor}_{stem}_map.cub"
            cache_dir.mkdir(parents=True, exist_ok=True)
            if map_cub.exists():
                shutil.copy2(map_cub, cache_path_cub)
                print(f"[CACHE] Saved map cube to {cache_path_cub}")
                
        # Cleanup
        if temp_cub.exists(): temp_cub.unlink()
        if map_cub.exists(): map_cub.unlink()
        if raw_tif.exists(): raw_tif.unlink()
        print(f"Finished processing LRO {sensor}: {norm_tif}")

    except FileNotFoundError:
        print(f"\n[WARNING] ISIS3 tools ({ingest_cmd}) not found on this system.")
        print(f"[FALLBACK] Bypassing ISIS and directly reading {input_file.name} via rasterio...")
        normalize_tiff(input_file, norm_tif, is_ohrc=False)
        print(f"Finished processing LRO {sensor} (Fallback): {norm_tif}")


import json
import tempfile

def resolve_ch2_kernels(xml_file, isisdata_path):
    """Preflight resolver to find and atomically download missing SPICE kernels."""
    print("\n[KERNEL RESOLVER]: Assuming DBs are already synced by the user...")

    out_json = tempfile.mktemp(suffix=".json")
    
    # Needs to run in WSL
    wsl_resolver_script = to_wsl_path(Path(__file__).parent / "wsl_kernel_resolver.py")
    wsl_xml = to_wsl_path(xml_file)
    wsl_out = to_wsl_path(out_json)
    
    wsl_resolver_cmd = [
        "wsl", "-d", "Ubuntu-24.04", "-e", "bash", "-ic",
        f"mamba run -n ch2_isis_dev python '{wsl_resolver_script}' --xml '{wsl_xml}' --isisdata '{isisdata_path}' --output '{wsl_out}'"
    ]
    
    print("\n[KERNEL RESOLVER]: Calculating missing kernels...")
    run_command(wsl_resolver_cmd)
    
    with open(out_json, "r") as f:
        plan = json.load(f)
    os.remove(out_json)
    
    if plan.get("error"):
        print(f"[KERNEL RESOLUTION ERROR]: {plan['error']}")
        return
        
    missing = plan.get("missing", [])
    if not missing:
        print("[KERNEL RESOLVER]: All required kernels are present locally.")
        
    resolved_paths = []
    
    for r in plan.get("local", []):
        resolved_paths.append(r)
    for m in missing:
        if m.startswith("UNRESOLVED"):
            print(f"[KERNEL RESOLVER WARNING]: Could not resolve kernel type: {m}")
        else:
            print(f"\n[KERNEL RESOLVER]: Atomically downloading: {m}")
            try:
                rel_path = str(Path(m).relative_to(Path(isisdata_path) / "chandrayaan2")).replace("\\", "/")
            except ValueError:
                rel_path = Path(m).name
                
            dl_cmd = [
                "downloadIsisData", "chandrayaan2", isisdata_path,
                "--no-kernels",
                f"--include={rel_path}"
            ]
            run_command(dl_cmd)
            resolved_paths.append(m)
            
    return resolved_paths


def process_ch2_optical(input_file, sensor, reference_cub, output_dir):
    """CH-2 Optical Branch: OHRC or TMC-2"""
    stem = input_file.stem
    norm_tif = output_dir / f"{stem}_normalized.tif"
    temp_cub = output_dir / f"{stem}_temp.cub"
    isd_json = output_dir / f"{stem}.json"
    map_cub = output_dir / f"{stem}_map.cub"
    raw_tif = output_dir / f"{stem}_raw.tif"
    
    try:
        # Check if map-projected cube already exists and is valid
        map_cub_candidates = [map_cub]
        if 'cache_dir' in globals():
            map_cub_candidates.append(cache_dir / f"{sensor}_{stem}_map.cub")
            
        found_map_cub = None
        for cand in map_cub_candidates:
            if cand.exists():
                cand_meta = extract_metadata(cand)
                if cand_meta.get('is_mapped'):
                    found_map_cub = cand
                    break
                    
        if found_map_cub:
            print(f"\n[CACHE/RECOVERY HIT] Found matching projected map.cub: {found_map_cub}")
            if found_map_cub != map_cub:
                import shutil
                shutil.copy2(found_map_cub, map_cub)
            print("Resuming directly from gdal_translate...")
            run_command(["gdal_translate", str(map_cub), str(raw_tif)])
            normalize_tiff(raw_tif, norm_tif, is_ohrc=(sensor == "OHRC"))
            
        else:
            resolved_kernels = []
            if input_file.suffix.lower() == ".xml":
                isisdata_path = "/home/kritikriti/isisdata_ch2"
                resolved_kernels = resolve_ch2_kernels(input_file, isisdata_path)
            else:
                print("[KERNEL RESOLVER WARNING]: Input is not XML, skipping preflight resolution.")
                
            # Convert WSL paths to Windows paths for spiceypy natively
            win_kernels = []
            if resolved_kernels:
                # Add static kernels required by SPICE
                isisdata_win = r"\\wsl.localhost\Ubuntu-24.04\home\kritikriti\isisdata_ch2"
                base_lsk = os.path.join(isisdata_win, "base", "kernels", "lsk", "naif0012.tls")
                ch2_pck = os.path.join(isisdata_win, "chandrayaan2", "kernels", "pck", "pck00010.tpc")
                ch2_sclk = os.path.join(isisdata_win, "chandrayaan2", "kernels", "sclk", "ch2_sclk_v1.tsc")
                ch2_fk = os.path.join(isisdata_win, "chandrayaan2", "kernels", "fk", "ch2_v01.tf")
                ch2_tspk = os.path.join(isisdata_win, "chandrayaan2", "kernels", "tspk", "de430s.bsp")
                
                ik_name = "ch2_ohr_v01.ti" if sensor == "OHRC" else "ch2_tmc_v01.ti"
                ch2_ik = os.path.join(isisdata_win, "chandrayaan2", "kernels", "ik", ik_name)
                
                win_kernels = [base_lsk, ch2_pck, ch2_sclk, ch2_fk, ch2_tspk, ch2_ik]
                for k in resolved_kernels:
                    win_kernels.append(k.replace("/home/kritikriti/isisdata_ch2", isisdata_win).replace("/", "\\"))
            
            # Execute native Windows spice_georeference.py
            spice_georef_script = Path(__file__).parent / "spice_georeference.py"
            python_exe = r"C:\Users\podhu\.local\share\mamba\envs\lunar_reg10\python.exe"
            
            cmd = [python_exe, str(spice_georef_script), "--xml", str(input_file), "--raw_tif", str(input_file.with_suffix(".img")), "--out_tif", str(raw_tif), "--sensor", sensor]
            if win_kernels:
                cmd.extend(["--kernels"] + win_kernels)
                
            try:
                run_command(cmd)
            except Exception as e:
                print(f"[ERROR] Failed to run spice_georeference.py: {e}")
                
            if raw_tif.exists():
                normalize_tiff(raw_tif, norm_tif, is_ohrc=(sensor == "OHRC"))
            else:
                print("[WARNING] spice_georeference did not produce output. Generating unprojected normalized image...")
                normalize_tiff(input_file.with_suffix(".img"), norm_tif, is_ohrc=(sensor == "OHRC"))
        
        # Cleanup
        if temp_cub.exists(): temp_cub.unlink()
        if isd_json.exists(): isd_json.unlink()
        if map_cub.exists(): map_cub.unlink()
        if raw_tif.exists(): raw_tif.unlink()
        
        # Cache the results
        if 'cache_dir' in globals():
            import shutil
            cache_path_tif = cache_dir / f"{sensor}_{stem}_normalized.tif"
            cache_path_cub = cache_dir / f"{sensor}_{stem}_map.cub"
            cache_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(norm_tif, cache_path_tif)
            if map_cub.exists():
                shutil.copy2(map_cub, cache_path_cub)
            print(f"[CACHE] Saved results to {cache_dir}")
            
        print(f"Finished processing CH-2 {sensor}: {norm_tif}")

    except FileNotFoundError:
        print("\n[WARNING] ISIS3 tools (isisimport) not found on this system.")
        print(f"[FALLBACK] Bypassing ISIS and directly reading {input_file.name} via rasterio...")
        normalize_tiff(input_file, norm_tif, is_ohrc=(sensor == "OHRC"))
        print(f"Finished processing CH-2 {sensor} (Fallback): {norm_tif}")


from iirs_band_selector import select_best_band_for_wac

def process_ch2_iirs(input_file, nav_file, reference_cub, output_dir):
    """CH-2 IIRS Branch"""
    print("\n[IIRS BRANCH ACTIVATED]")
    
    stem = input_file.stem
    raw_tif = output_dir / f"{stem}_raw.tif"
    norm_tif = output_dir / f"{stem}_normalized.tif"
    
    resolved_kernels = []
    if input_file.suffix.lower() == ".xml":
        isisdata_path = "/home/kritikriti/isisdata_ch2"
        resolved_kernels = resolve_ch2_kernels(input_file, isisdata_path)
    else:
        print("[KERNEL RESOLVER WARNING]: Input is not XML, skipping preflight resolution.")
        
    win_kernels = []
    if resolved_kernels:
        isisdata_win = r"\\wsl.localhost\Ubuntu-24.04\home\kritikriti\isisdata_ch2"
        base_lsk = os.path.join(isisdata_win, "base", "kernels", "lsk", "naif0012.tls")
        ch2_pck = os.path.join(isisdata_win, "chandrayaan2", "kernels", "pck", "pck00010.tpc")
        ch2_sclk = os.path.join(isisdata_win, "chandrayaan2", "kernels", "sclk", "ch2_sclk_v1.tsc")
        ch2_fk = os.path.join(isisdata_win, "chandrayaan2", "kernels", "fk", "ch2_v01.tf")
        ch2_tspk = os.path.join(isisdata_win, "chandrayaan2", "kernels", "tspk", "de430s.bsp")
        ch2_ik = os.path.join(isisdata_win, "chandrayaan2", "kernels", "ik", "ch2_iir_v01.ti")
        
        win_kernels = [base_lsk, ch2_pck, ch2_sclk, ch2_fk, ch2_tspk, ch2_ik]
        for k in resolved_kernels:
            win_kernels.append(k.replace("/home/kritikriti/isisdata_ch2", isisdata_win).replace("/", "\\"))
            
    spice_georef_script = Path(__file__).parent / "spice_georeference.py"
    python_exe = r"C:\Users\podhu\.local\share\mamba\envs\lunar_reg10\python.exe"
    
    cmd = [python_exe, str(spice_georef_script), "--xml", str(input_file), "--raw_tif", str(input_file.with_suffix(".img")), "--out_tif", str(raw_tif), "--sensor", "IIRS"]
    if win_kernels:
        cmd.extend(["--kernels"] + win_kernels)
        
    try:
        run_command(cmd)
    except Exception as e:
        print(f"[ERROR] Failed to run spice_georeference.py for IIRS: {e}")
        
    if raw_tif.exists():
        normalize_tiff(raw_tif, norm_tif, is_ohrc=False)
    else:
        print("[WARNING] spice_georeference did not produce output. Generating unprojected normalized image...")
        normalize_tiff(input_file.with_suffix(".img"), norm_tif, is_ohrc=False)
        
    print(f"Finished processing CH-2 IIRS: {norm_tif}")


def main():
    parser = argparse.ArgumentParser(description="Multi-Sensor ISIS/GDAL Preprocessing Pipeline (Auto Map-Templating)")
    parser.add_argument("--input", required=True, type=Path, help="Input raw image file (.IMG, .xml, .h5)")
    parser.add_argument("--sensor", required=True, choices=["NAC", "WAC", "OHRC", "TMC2", "IIRS"])
    parser.add_argument("--reference_cub", type=Path, help="Optional: Reference ISIS cube to copy map projection from (Master image).")
    parser.add_argument("--output_dir", default=".", type=Path)
    parser.add_argument("--nav", type=Path, help="Nav/Geometry auxiliary file (Required for IIRS)")
    parser.add_argument("--cache_dir", type=Path, default=Path("C:/#Padhai/E/SIH1/preprocessed_cache"), help="Global cache directory for preprocessed images")
    
    args = parser.parse_args()
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Expose cache_dir globally so process functions can use it to store
    global cache_dir
    cache_dir = args.cache_dir
    
    stem = args.input.stem
    cached_tif = args.cache_dir / f"{args.sensor}_{stem}_normalized.tif"
    out_tif = args.output_dir / f"{stem}_normalized.tif"
    
    if cached_tif.exists():
        import shutil
        print(f"\n[CACHE HIT] Found existing normalized TIF in cache: {cached_tif}")
        shutil.copy2(cached_tif, out_tif)
        print(f"Copied to output directory: {out_tif}")
        return
        
    if args.sensor in ["NAC", "WAC"]:
        process_lro(args.input, args.sensor, args.reference_cub, args.output_dir)
        if out_tif.exists():
            import shutil
            shutil.copy2(out_tif, cached_tif)
            print(f"[CACHE] Saved to {cached_tif}")
    elif args.sensor in ["OHRC", "TMC2"]:
        if not args.reference_cub:
            parser.error(f"--reference_cub is strictly required for CH-2 {args.sensor} to avoid massive default projections!")
        process_ch2_optical(args.input, args.sensor, args.reference_cub, args.output_dir)
    elif args.sensor == "IIRS":
        if not args.nav:
            parser.error("--nav is required for IIRS processing")
        process_ch2_iirs(args.input, args.nav, args.reference_cub, args.output_dir)

if __name__ == "__main__":
    main()
