from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import logging

from research_monitor.config import DATABASE_URL
from research_monitor.database.models import Base, create_training_labels_view

logger = logging.getLogger(__name__)

# Resilient database engine configuration for hosted PostgreSQL & local environments
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=300,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db(target_engine=None):
    """Initializes the database schema, pgvector extension, and SQL views."""
    eng = target_engine or engine
    try:
        if eng.dialect.name == "postgresql":
            try:
                with eng.begin() as conn:
                    conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
                    conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS access_token VARCHAR;"))
                    conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS is_simulated BOOLEAN DEFAULT FALSE;"))
                    conn.execute(text("ALTER TABLE feedback_events ADD COLUMN IF NOT EXISTS decision_id INTEGER;"))
                    conn.execute(text("ALTER TABLE feedback_events ADD COLUMN IF NOT EXISTS is_audit BOOLEAN DEFAULT FALSE;"))
                    logger.info("Ensured pgvector extension and column migrations are applied.")
            except Exception as ext_err:
                logger.warning(
                    "Postgres pre-flight migration warning: %s",
                    ext_err,
                )
        Base.metadata.create_all(bind=eng)
        create_training_labels_view(eng)
        logger.info("Database schema and views initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise

def get_session():
    """Provides a database session context."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
