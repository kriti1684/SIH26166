try:
    import pvl
except ImportError:
    pvl = None
import os
import glob
import json
import argparse
from datetime import datetime
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class KernelCandidate:
    file_path: str
    kernel_type: str
    start_time: datetime
    stop_time: datetime
    priority: int


class KernelResolver:
    def __init__(self, isis_data_dir: str):
        self.isis_data_dir = isis_data_dir
        self.ch2_data_dir = os.path.join(isis_data_dir, "chandrayaan2")
        
        self.priorities = {
            "reconstructed": 3,
            "predicted": 2,
            "nadir": 1
        }

    def parse_time_tdb(self, time_str: str) -> datetime:
        time_str = time_str.replace(" TDB", "").strip()
        if "." in time_str:
            return datetime.strptime(time_str, "%Y %b %d %H:%M:%S.%f")
        else:
            return datetime.strptime(time_str, "%Y %b %d %H:%M:%S")

    def parse_xml_time(self, xml_path: str) -> Tuple[datetime, datetime]:
        tree = ET.parse(xml_path)
        root = tree.getroot()
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
        candidates = []
        db_files = glob.glob(os.path.join(db_folder, "*.db"))
        
        for db_file in db_files:
            try:
                data = pvl.load(db_file)
            except Exception:
                continue
                
            if hasattr(data, 'getall'):
                s_pointing_list = []
                try:
                    s_pointing_list = data.getall("SpacecraftPointing")
                except KeyError:
                    pass
                
                if not s_pointing_list:
                    try:
                        s_pointing_list = data.getall("SpacecraftPosition")
                    except KeyError:
                        pass

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
        db_folder = os.path.join(self.ch2_data_dir, "kernels", ktype)
        if not os.path.exists(db_folder):
            return None
            
        candidates = self.get_candidates(db_folder)
        valid_matches = []
        for c in candidates:
            if c.start_time <= img_start and c.stop_time >= img_stop:
                valid_matches.append(c)
                
        if not valid_matches:
            return None
            
        valid_matches.sort(key=lambda x: x.priority, reverse=True)
        return valid_matches[0]


def main():
    parser = argparse.ArgumentParser(description="Preflight Kernel Resolver for ISIS/SPICE")
    parser.add_argument("--xml", required=True, help="Path to PDS4 XML file")
    parser.add_argument("--isisdata", required=True, help="Path to ISISDATA directory")
    parser.add_argument("--output", required=True, help="Path to output JSON plan file")
    args = parser.parse_args()

    resolver = KernelResolver(args.isisdata)
    plan = {"required": [], "missing": [], "local": [], "error": None}
    
    try:
        start, stop = resolver.parse_xml_time(args.xml)
    except Exception as e:
        plan["error"] = str(e)
        with open(args.output, "w") as f:
            json.dump(plan, f, indent=2)
        return

    for ktype in ["ck", "spk"]:
        best = resolver.resolve_kernel(ktype, start, stop)
        if not best:
            plan["missing"].append(f"UNRESOLVED_{ktype.upper()}")
            continue
            
        real_path = best.file_path.replace("$chandrayaan2", os.path.join(args.isisdata, "chandrayaan2"))
        real_path = os.path.normpath(real_path)
        plan["required"].append(real_path)
        
        if os.path.exists(real_path):
            plan["local"].append(real_path)
        else:
            plan["missing"].append(real_path)
            
    with open(args.output, "w") as f:
        json.dump(plan, f, indent=2)
    print(f"[SUCCESS] Kernel resolution plan written to: {args.output}")


if __name__ == "__main__":
    main()
