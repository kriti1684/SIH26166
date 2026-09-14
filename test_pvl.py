import pvl
import json

db_path = "/mnt/c/#Padhai/E/SIH1/temp_db/chandrayaan2/kernels/ck/kernels.0001.db"

def default_serializer(obj):
    if isinstance(obj, set):
        return list(obj)
    return str(obj)

try:
    # ISIS pvl library can load .db files directly
    data = pvl.load(db_path)
    
    # We convert the pvl structure to a dictionary for easier inspection
    # PVL modules behave like dicts, but might contain nested structures
    def pvl_to_list(node):
        result = []
        if hasattr(node, 'items'):
            for k, v in node.items():
                if hasattr(v, 'items') or isinstance(v, list):
                    result.append((k, pvl_to_list(v)))
                else:
                    result.append((k, v))
        elif isinstance(node, list):
            for item in node:
                result.append(pvl_to_list(item) if hasattr(item, 'items') else item)
        else:
            return node
        return result
            
    print("Successfully parsed DB file. Structure (as list of tuples to preserve duplicates):")
    # Print the first few items
    as_list = pvl_to_list(data)
    print(json.dumps(as_list, default=default_serializer, indent=2)[:3000])
    
except Exception as e:
    print(f"Error parsing PVL: {e}")
