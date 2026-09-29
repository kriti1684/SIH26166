from sqlalchemy import create_engine, inspect, text, event
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from backend.config import DATABASE_URL

_connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, connect_args=_connect_args)

@event.listens_for(engine, 'connect')
def set_sqlite_pragma(dbapi_connection, connection_record):
    if DATABASE_URL.startswith('sqlite'):
        cursor = dbapi_connection.cursor()
        cursor.execute('PRAGMA journal_mode=WAL')
        cursor.execute('PRAGMA synchronous=NORMAL')
        cursor.close()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class Base(DeclarativeBase):
    pass

def init_db():
    import backend.models
    Base.metadata.create_all(bind=engine)

    # create_all does not add columns to a database created by an earlier app
    # version. Keep these small additive upgrades compatible with SQLite and
    # PostgreSQL so existing local job history remains usable.
    required_columns = {
        "run_config_json": "JSON",
        "stage_events_json": "JSON",
        "project_name": "VARCHAR(256)",
    }
    columns = {column["name"] for column in inspect(engine).get_columns("registration_jobs")}
    missing = [(name, sql_type) for name, sql_type in required_columns.items() if name not in columns]
    if missing:
        with engine.begin() as connection:
            for name, sql_type in missing:
                connection.execute(text(f"ALTER TABLE registration_jobs ADD COLUMN {name} {sql_type}"))

    reconcile_stale_jobs()


def reconcile_stale_jobs():
    from backend.models import RegistrationJob, JobStatus
    with SessionLocal() as db:
        stale = db.query(RegistrationJob).filter(RegistrationJob.status.in_([JobStatus.PENDING, JobStatus.PROCESSING])).all()
        for j in stale:
            j.status = JobStatus.FAILED
            j.current_stage = "FAILED"
            j.error_message = "Server process was restarted while registration was in progress."
        if stale:
            db.commit()
            print(f"[STARTUP] Reconciled {len(stale)} orphaned running/pending job(s) to FAILED.")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
