import pvl
import os
import glob
import re
from datetime import datetime
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import List, Optional, Tuple

@dataclass
class KernelCandidate:
    file_path: str
    kernel_type: str  # e.g., 'Reconstructed', 'Predicted', 'Nadir'
    start_time: datetime
    stop_time: datetime
    priority: int

class KernelResolver:
    def __init__(self, isis_data_dir: str):
        self.isis_data_dir = isis_data_dir
        self.ch2_data_dir = os.path.join(isis_data_dir, "chandrayaan2")
        
        # Priorities as defined by ISIS convention
        self.priorities = {
            "reconstructed": 3,
            "predicted": 2,
            "nadir": 1
        }

    def parse_time_tdb(self, time_str: str) -> datetime:
        """Parse TDB time string like '2024 APR 27 17:33:32.567381 TDB'"""
        time_str = time_str.replace(" TDB", "").strip()
        # Handle cases where microseconds might be missing
        if "." in time_str:
            return datetime.strptime(time_str, "%Y %b %d %H:%M:%S.%f")
        else:
            return datetime.strptime(time_str, "%Y %b %d %H:%M:%S")

    def parse_xml_time(self, xml_path: str) -> Tuple[datetime, datetime]:
        """Extract Start and Stop times from Chandrayaan-2 PDS4 XML label."""
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # PDS4 namespaces
        ns = {'pds': 'http://pds.nasa.gov/pds4/pds/v1'}
        
        time_coords = root.find('.//pds:Time_Coordinates', ns)
        if time_coords is None:
            raise ValueError(f"Time_Coordinates not found in {xml_path}")
            
        start_str = time_coords.find('pds:start_date_time', ns).text.replace('Z', '')
        stop_str = time_coords.find('pds:stop_date_time', ns).text.replace('Z', '')
        
        start_time = datetime.fromisoformat(start_str)
        stop_time = datetime.fromisoformat(stop_str)
        return start_time, stop_time

    def get_candidates(self, db_folder: str) -> List[KernelCandidate]:
        """Parse all .db files in a folder and extract all candidates."""
        candidates = []
        db_files = glob.glob(os.path.join(db_folder, "*.db"))
        
        for db_file in db_files:
            try:
                data = pvl.load(db_file)
            except Exception as e:
                print(f"Warning: Could not parse {db_file}: {e}")
                continue
                
            # Iterate through PVL structure to find Selection groups
            if hasattr(data, 'getall'):
                # pvl.collections.PVLModule supports getall for repeated keys
                s_pointing_list = data.getall("SpacecraftPointing")
                for sp in s_pointing_list:
                    if hasattr(sp, 'getall'):
                        selections = sp.getall("Selection")
                        for sel in selections:
                            if hasattr(sel, 'getall'):
                                file_path = sel.get("File")
                                k_type = sel.get("Type", "Unknown")
                                times = sel.getall("Time")
                                
                                if file_path:
                                    priority = self.priorities.get(str(k_type).lower(), 0)
                                    for t_range in times:
                                        if isinstance(t_range, list) and len(t_range) == 2:
                                            candidates.append(KernelCandidate(
                                                file_path=str(file_path),
                                                kernel_type=str(k_type),
                                                start_time=self.parse_time_tdb(str(t_range[0])),
                                                stop_time=self.parse_time_tdb(str(t_range[1])),
                                                priority=priority
                                            ))
        return candidates

    def resolve_kernel(self, ktype: str, img_start: datetime, img_stop: datetime) -> Optional[KernelCandidate]:
        """Resolve the best kernel for a specific time interval and kernel type (ck/spk)."""
        db_folder = os.path.join(self.ch2_data_dir, "kernels", ktype)
        if not os.path.exists(db_folder):
            print(f"Warning: DB folder {db_folder} does not exist.")
            return None
            
        candidates = self.get_candidates(db_folder)
        
        valid_matches = []
        for c in candidates:
            # Full interval coverage check
            if c.start_time <= img_start and c.stop_time >= img_stop:
                valid_matches.append(c)
                
        if not valid_matches:
            return None
            
        # Sort by priority descending
        valid_matches.sort(key=lambda x: x.priority, reverse=True)
        return valid_matches[0]

def resolve_and_plan(xml_path: str, isis_data_dir: str):
    print("=" * 80)
    print("KERNEL RESOLUTION PLAN (DRY RUN)")
    print("=" * 80)
    
    resolver = KernelResolver(isis_data_dir)
    
    print(f"Image XML:\n  {os.path.basename(xml_path)}")
    start, stop = resolver.parse_xml_time(xml_path)
    print(f"Acquisition Interval:\n  Start: {start.isoformat()}Z\n  Stop:  {stop.isoformat()}Z\n")
    
    plan = {"required": [], "missing": [], "local": []}
    
    for ktype in ["ck", "spk"]:
        print(f"--- Resolving {ktype.upper()} ---")
        best = resolver.resolve_kernel(ktype, start, stop)
        if not best:
            print(f"  Result: NO MATCHING KERNEL FOUND IN DATABASE.")
            plan["missing"].append(f"UNRESOLVED_{ktype.upper()}")
            continue
            
        print(f"  Selected: {best.file_path} (Type: {best.kernel_type})")
        
        # Translate ISIS variable to real path
        real_path = best.file_path.replace("$chandrayaan2", os.path.join(isis_data_dir, "chandrayaan2"))
        
        plan["required"].append(real_path)
        
        if os.path.exists(real_path):
            print(f"  Local Status: PRESENT")
            plan["local"].append(real_path)
        else:
            print(f"  Local Status: MISSING")
            plan["missing"].append(real_path)
        print()
            
    print("--- Download Plan ---")
    if plan["missing"]:
        for m in plan["missing"]:
            if m.startswith("UNRESOLVED"):
                print(f"  [ERROR] Cannot download, resolution failed for {m}")
            else:
                rel_path = m.replace(isis_data_dir + os.sep, "").replace("\\", "/")
                print(f"  Needs download: {rel_path}")
    else:
        print("  All required kernels are present locally.")
        
    return plan

if __name__ == "__main__":
    # Test with the OHRC XML
    xml_file = "/mnt/c/#Padhai/E/SIH1/runs/run_20260911_153214/uploads/ch2_ohr_ncp_20210331T2033243734_d_img_d18.xml"
    
    # We downloaded some temp DBs to Windows for testing
    temp_isis_data = "/mnt/c/#Padhai/E/SIH1/temp_db"
    
    resolve_and_plan(xml_file, temp_isis_data)
