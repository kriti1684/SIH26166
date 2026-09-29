import os
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent

STORAGE_DIR   = PROJECT_ROOT / "data" / "storage"
UPLOADS_DIR   = STORAGE_DIR / "uploads"
OUTPUTS_DIR   = STORAGE_DIR / "outputs"
PREVIEWS_DIR  = STORAGE_DIR / "previews"

for _d in (UPLOADS_DIR, OUTPUTS_DIR, PREVIEWS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{STORAGE_DIR / 'lunar_reg.db'}")
MAX_JOB_UPLOAD_BYTES = int(os.getenv("MAX_JOB_UPLOAD_BYTES", str(20 * 1024**3)))

PROJECT_NAME = "ChandraShakti"
VERSION      = "2.0.0"
API_V1_PREFIX = "/api/v1"

CORS_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
]
