import os
import glob
import spiceypy as spice

spice_dir = r"data\spice"
print("=== GENERIC & TEXT KERNELS ===")
for f in sorted(os.listdir(spice_dir)):
    if f.endswith((".tls", ".tpc", ".tf", ".ti", ".tsc")):
        print(f"  {f}")

print("\n=== CK (ATTITUDE) KERNELS ===")
tls = glob.glob(os.path.join(spice_dir, "*.tls"))[0]
sclk = glob.glob(os.path.join(spice_dir, "*.tsc"))[0]
spice.furnsh(tls)
spice.furnsh(sclk)

for f in sorted(glob.glob(os.path.join(spice_dir, "*.bc"))):
    fname = os.path.basename(f)
    size_mb = os.path.getsize(f) / (1024 * 1024)
    try:
        ids = spice.ckobj(f)
        coverages = []
        for obj in ids:
            cov = spice.ckcov(f, obj, False, "INTERVAL", 0.0, "TDB")
            n_int = spice.wncard(cov)
            if n_int > 0:
                s, _ = spice.wnfetd(cov, 0)
                _, e_end = spice.wnfetd(cov, n_int - 1)
                s_str = spice.et2utc(s, "C", 0)
                e_str = spice.et2utc(e_end, "C", 0)
                coverages.append(f"Obj {obj}: {s_str} -> {e_str} ({n_int} intervals)")
        print(f"  {fname} ({size_mb:.1f} MB):")
        for c in coverages:
            print(f"    {c}")
    except Exception as ex:
        print(f"  {fname} ({size_mb:.1f} MB): Error reading CK: {ex}")

print("\n=== SPK (ORBIT) KERNELS ===")
spks = sorted(glob.glob(os.path.join(spice_dir, "*.bsp")))
print(f"Total SPK files: {len(spks)}")
all_objs = set()
spk_intervals = []
for f in spks:
    fname = os.path.basename(f)
    try:
        ids = spice.spkobj(f)
        for obj in ids:
            all_objs.add(obj)
            cov = spice.spkcov(f, obj)
            n_int = spice.wncard(cov)
            if n_int > 0:
                s, _ = spice.wnfetd(cov, 0)
                _, e = spice.wnfetd(cov, n_int - 1)
                s_str = spice.et2utc(s, "C", 0)
                e_str = spice.et2utc(e, "C", 0)
                spk_intervals.append((s, e, s_str, e_str, fname, obj))
    except Exception as ex:
        print(f"Error {fname}: {ex}")

print(f"Detected Object IDs across SPK: {all_objs}")
ch2_spks = [x for x in spk_intervals if x[5] == -152 or "ch2" in x[4]]
print(f"Total Chandrayaan-2 SPK entries: {len(ch2_spks)}")
if ch2_spks:
    ch2_sorted = sorted(ch2_spks, key=lambda x: x[0])
    print(f"Earliest CH2 SPK: {ch2_sorted[0][2]} ({ch2_sorted[0][4]})")
    print(f"Latest CH2 SPK:   {ch2_sorted[-1][3]} ({ch2_sorted[-1][4]})")

spice.kclear()


