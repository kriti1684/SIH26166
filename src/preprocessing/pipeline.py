import argparse
import subprocess
import sys
from pathlib import Path
import os
import json
import tempfile
import rasterio
import numpy as np

from .normalizer import normalize_tiff
from .band_selector import select_best_band_for_wac

DEFAULT_WSL_DISTRO = os.environ.get("WSL_DISTRO", "Ubuntu-24.04")
DEFAULT_WSL_ISISDATA = os.environ.get("WSL_ISISDATA", "/home/kritikriti/isisdata_ch2")
DEFAULT_WIN_ISISDATA = os.environ.get("WIN_ISISDATA", r"\\wsl.localhost\Ubuntu-24.04\home\kritikriti\isisdata_ch2")


def to_wsl_path(win_path):
    path_str = str(win_path).replace("\\", "/")
    if len(path_str) >= 2 and path_str[1] == ":":
        drive = path_str[0].lower()
        path_str = f"/mnt/{drive}/" + path_str[3:]
    return path_str


def extract_metadata(file_path):
    metadata = {}
    file_str = str(file_path).lower()
    try:
        if file_str.endswith(".xml"):
            import xml.etree.ElementTree as ET
            tree = ET.parse(file_path)
            root = tree.getroot()
            ns = {'pds': 'http://pds.nasa.gov/pds4/pds/v1'}
            time_elem = root.find('.//pds:start_date_time', ns)
            if time_elem is not None:
                metadata['STARTTIME'] = time_elem.text.strip().replace('Z', '')
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
        wsl_cmd = [
            "wsl", "-d", DEFAULT_WSL_DISTRO, "-e", "bash", "-ic",
            f"mamba run -n ch2_isis_dev env ISISDATA={DEFAULT_WSL_ISISDATA} ALESPICEROOT={DEFAULT_WSL_ISISDATA} {inner_cmd}"
        ]
        
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


def resolve_ch2_kernels(xml_file, isisdata_path):
    print("\n[KERNEL RESOLVER]: Checking local SPICE databases...")
    out_json = tempfile.mktemp(suffix=".json")
    
    resolver_script = to_wsl_path(Path(__file__).parent / "kernel_resolver.py")
    wsl_xml = to_wsl_path(xml_file)
    wsl_out = to_wsl_path(out_json)
    
    wsl_resolver_cmd = [
        "wsl", "-d", DEFAULT_WSL_DISTRO, "-e", "bash", "-ic",
        f"mamba run -n ch2_isis_dev python '{resolver_script}' --xml '{wsl_xml}' --isisdata '{isisdata_path}' --output '{wsl_out}'"
    ]
    
    print("\n[KERNEL RESOLVER]: Calculating missing kernels...")
    run_command(wsl_resolver_cmd)
    
    with open(out_json, "r") as f:
        plan = json.load(f)
    os.remove(out_json)
    
    if plan.get("error"):
        print(f"[KERNEL RESOLUTION ERROR]: {plan['error']}")
        return []
        
    resolved_paths = list(plan.get("local", []))
    missing = plan.get("missing", [])
    
    for m in missing:
        if m.startswith("UNRESOLVED"):
            print(f"[KERNEL RESOLVER WARNING]: Could not resolve kernel type: {m}")
        else:
            print(f"\n[KERNEL RESOLVER]: Downloading: {m}")
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


def process_lro(input_file: Path, sensor: str, reference_cub: Path, output_dir: Path, cache_dir: Path):
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
        
        if cache_dir:
            import shutil
            cache_path_cub = cache_dir / f"{sensor}_{stem}_map.cub"
            cache_dir.mkdir(parents=True, exist_ok=True)
            if map_cub.exists():
                shutil.copy2(map_cub, cache_path_cub)
                
        if temp_cub.exists(): temp_cub.unlink()
        if map_cub.exists(): map_cub.unlink()
        if raw_tif.exists(): raw_tif.unlink()
        print(f"Finished processing LRO {sensor}: {norm_tif}")

    except FileNotFoundError:
        print(f"[FALLBACK] Bypassing ISIS and directly reading {input_file.name} via rasterio...")
        normalize_tiff(input_file, norm_tif, is_ohrc=False)


def process_ch2_optical(input_file: Path, sensor: str, reference_cub: Path, output_dir: Path, cache_dir: Path):
    stem = input_file.stem
    norm_tif = output_dir / f"{stem}_normalized.tif"
    raw_tif = output_dir / f"{stem}_raw.tif"
    
    resolved_kernels = []
    if input_file.suffix.lower() == ".xml":
        resolved_kernels = resolve_ch2_kernels(input_file, DEFAULT_WSL_ISISDATA)
    
    win_kernels = []
    if resolved_kernels:
        base_lsk = os.path.join(DEFAULT_WIN_ISISDATA, "base", "kernels", "lsk", "naif0012.tls")
        ch2_pck = os.path.join(DEFAULT_WIN_ISISDATA, "chandrayaan2", "kernels", "pck", "pck00010.tpc")
        ch2_sclk = os.path.join(DEFAULT_WIN_ISISDATA, "chandrayaan2", "kernels", "sclk", "ch2_sclk_v1.tsc")
        ch2_fk = os.path.join(DEFAULT_WIN_ISISDATA, "chandrayaan2", "kernels", "fk", "ch2_v01.tf")
        ch2_tspk = os.path.join(DEFAULT_WIN_ISISDATA, "chandrayaan2", "kernels", "tspk", "de430s.bsp")
        ik_name = "ch2_ohr_v01.ti" if sensor == "OHRC" else "ch2_tmc_v01.ti"
        ch2_ik = os.path.join(DEFAULT_WIN_ISISDATA, "chandrayaan2", "kernels", "ik", ik_name)
        
        win_kernels = [base_lsk, ch2_pck, ch2_sclk, ch2_fk, ch2_tspk, ch2_ik]
        for k in resolved_kernels:
            win_kernels.append(k.replace(DEFAULT_WSL_ISISDATA, DEFAULT_WIN_ISISDATA).replace("/", "\\"))
            
    spice_georef_script = Path(__file__).parent / "spice_georeference.py"
    python_exe = sys.executable
    
    cmd = [python_exe, str(spice_georef_script), "--xml", str(input_file), "--raw_tif", str(input_file.with_suffix(".img")), "--out_tif", str(raw_tif), "--sensor", sensor]
    if win_kernels:
        cmd.extend(["--kernels"] + win_kernels)
        
    try:
        run_command(cmd)
    except Exception as e:
        print(f"[ERROR] Failed to run spice_georeference.py: {e}")
        
    if raw_tif.exists():
        normalize_tiff(raw_tif, norm_tif, is_ohrc=(sensor == "OHRC"))
        raw_tif.unlink()
    else:
        print("[WARNING] spice_georeference did not produce output. Generating unprojected normalized image...")
        normalize_tiff(input_file.with_suffix(".img"), norm_tif, is_ohrc=(sensor == "OHRC"))


def main():
    parser = argparse.ArgumentParser(description="Multi-Sensor Ingestion & Preprocessing Pipeline")
    parser.add_argument("--input", required=True, type=Path, help="Input raw image file (.IMG, .xml, .h5)")
    parser.add_argument("--sensor", required=True, choices=["NAC", "WAC", "OHRC", "TMC2", "IIRS"])
    parser.add_argument("--reference_cub", type=Path, help="Reference ISIS cube to copy map projection from")
    parser.add_argument("--output_dir", default=Path("."), type=Path)
    parser.add_argument("--nav", type=Path, help="Nav/Geometry file for IIRS")
    parser.add_argument("--cache_dir", type=Path, default=Path("./preprocessed_cache"))
    args = parser.parse_args()
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.sensor in ["NAC", "WAC"]:
        process_lro(args.input, args.sensor, args.reference_cub, args.output_dir, args.cache_dir)
    elif args.sensor in ["OHRC", "TMC2"]:
        process_ch2_optical(args.input, args.sensor, args.reference_cub, args.output_dir, args.cache_dir)


if __name__ == "__main__":
    main()
