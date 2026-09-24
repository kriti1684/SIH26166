"""
FastAPI Entrypoint.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.config import PROJECT_NAME, VERSION, API_V1_PREFIX, CORS_ORIGINS, DATABASE_URL
from backend.database import init_db
from backend.routes import router

app = FastAPI(title=PROJECT_NAME, version=VERSION, docs_url="/docs", redoc_url="/redoc")

app.add_middleware(
    CORSMiddleware, allow_origins=CORS_ORIGINS,
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    init_db()

app.include_router(router, prefix=API_V1_PREFIX)

@app.get("/")
def read_root():
    return {"project": PROJECT_NAME, "version": VERSION, "docs": "/docs", "health": "/health"}

@app.get("/health")
def check_health():
    import torch
    try:
        from backend.worker import celery_app
        celery_app.control.ping(timeout=1.0)
        celery_status = "OK"
    except Exception:
        celery_status = "UNREACHABLE"
    return {
        "status": "online", "version": VERSION, "db": DATABASE_URL.split(":")[0],
        "celery_queue": celery_status,
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    }
