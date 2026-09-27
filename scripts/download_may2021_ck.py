import os
import sys
import time
import urllib.request

url = "https://asc-isisdata.s3.us-west-2.amazonaws.com/usgs_data/chandrayaan2/kernels/ck/ch2_att_27Apr2021_04Jun2021_v1.bc"
dest_dir = r"data\spice"
dest_path = os.path.join(dest_dir, "ch2_att_27Apr2021_04Jun2021_v1.bc")
temp_path = dest_path + ".part"

print(f"Target: {os.path.basename(dest_path)}")
print(f"URL   : {url}")

req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})

if os.path.exists(dest_path):
    sz = os.path.getsize(dest_path) / (1024 * 1024)
    print(f"File already exists! Size: {sz:.2f} MB")
    sys.exit(0)

resume_byte_pos = 0
if os.path.exists(temp_path):
    resume_byte_pos = os.path.getsize(temp_path)
    req.add_header("Range", f"bytes={resume_byte_pos}-")
    print(f"Resuming from byte position: {resume_byte_pos / (1024*1024):.2f} MB")

try:
    with urllib.request.urlopen(req, timeout=30) as resp:
        content_range = resp.headers.get("Content-Range")
        if content_range:
            total_size = int(content_range.split("/")[-1])
        else:
            total_size = int(resp.headers.get("Content-Length", 0)) + resume_byte_pos

        mode = "ab" if resume_byte_pos > 0 else "wb"
        downloaded = resume_byte_pos
        start_time = time.time()
        last_print = start_time

        with open(temp_path, mode) as f:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)

                now = time.time()
                if now - last_print >= 5.0 or downloaded >= total_size:
                    elapsed = now - start_time
                    speed = (downloaded - resume_byte_pos) / (elapsed + 1e-6) / (1024 * 1024)
                    pct = (downloaded / total_size) * 100 if total_size else 0
                    print(f"Progress: {downloaded / (1024*1024):.1f} / {total_size / (1024*1024):.1f} MB ({pct:.1f}%) | Speed: {speed:.2f} MB/s", flush=True)
                    last_print = now

    os.rename(temp_path, dest_path)
    print(f"\n[SUCCESS] Download completed and verified: {dest_path}")
    print(f"Final Size: {os.path.getsize(dest_path) / (1024*1024):.2f} MB")

except Exception as e:
    print(f"\n[ERROR] Download failed: {e}", file=sys.stderr)
    sys.exit(1)
