from contextlib import contextmanager
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

                    # Schema migration for validation_labels table (user_id, sampling_stratum, labelled_at, constraints)
                    conn.execute(text("""
                        DO $$
                        BEGIN
                            IF EXISTS (
                                SELECT 1 FROM information_schema.tables WHERE table_name = 'validation_labels'
                            ) THEN
                                -- 1. Ensure user_id column exists
                                IF NOT EXISTS (
                                    SELECT 1 FROM information_schema.columns
                                    WHERE table_name = 'validation_labels' AND column_name = 'user_id'
                                ) THEN
                                    ALTER TABLE validation_labels ADD COLUMN user_id VARCHAR DEFAULT 'developer';
                                    UPDATE validation_labels SET user_id = 'developer' WHERE user_id IS NULL;
                                    ALTER TABLE validation_labels ALTER COLUMN user_id SET NOT NULL;
                                END IF;

                                -- 2. Rename stratum -> sampling_stratum if needed
                                IF EXISTS (
                                    SELECT 1 FROM information_schema.columns
                                    WHERE table_name = 'validation_labels' AND column_name = 'stratum'
                                ) AND NOT EXISTS (
                                    SELECT 1 FROM information_schema.columns
                                    WHERE table_name = 'validation_labels' AND column_name = 'sampling_stratum'
                                ) THEN
                                    ALTER TABLE validation_labels RENAME COLUMN stratum TO sampling_stratum;
                                END IF;

                                -- 3. Rename created_at -> labelled_at if needed
                                IF EXISTS (
                                    SELECT 1 FROM information_schema.columns
                                    WHERE table_name = 'validation_labels' AND column_name = 'created_at'
                                ) AND NOT EXISTS (
                                    SELECT 1 FROM information_schema.columns
                                    WHERE table_name = 'validation_labels' AND column_name = 'labelled_at'
                                ) THEN
                                    ALTER TABLE validation_labels RENAME COLUMN created_at TO labelled_at;
                                END IF;

                                -- 4. Clean up any accidental duplicate (user_id, paper_id) before adding unique constraint
                                DELETE FROM validation_labels a USING validation_labels b
                                WHERE a.id < b.id AND a.user_id = b.user_id AND a.paper_id = b.paper_id;

                                -- 5. Ensure unique constraint on (user_id, paper_id)
                                IF NOT EXISTS (
                                    SELECT 1 FROM pg_constraint WHERE conname = 'uq_validation_labels_user_paper'
                                ) THEN
                                    ALTER TABLE validation_labels ADD CONSTRAINT uq_validation_labels_user_paper UNIQUE (user_id, paper_id);
                                END IF;
                            END IF;
                        END $$;
                    """))
                    logger.info("Ensured pgvector extension and column migrations are applied.")
            except Exception as ext_err:
                logger.warning(
                    "Postgres pre-flight migration warning: %s",
                    ext_err,
                )
        elif eng.dialect.name == "sqlite":
            try:
                with eng.begin() as conn:
                    tbl_check = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table' AND name='validation_labels'")).fetchone()
                    if tbl_check:
                        cols = [r[1] for r in conn.execute(text("PRAGMA table_info(validation_labels)")).fetchall()]
                        if "user_id" not in cols:
                            conn.execute(text("ALTER TABLE validation_labels ADD COLUMN user_id TEXT DEFAULT 'developer' NOT NULL"))
                        if "sampling_stratum" not in cols and "stratum" in cols:
                            conn.execute(text("ALTER TABLE validation_labels RENAME COLUMN stratum TO sampling_stratum"))
                        if "labelled_at" not in cols and "created_at" in cols:
                            conn.execute(text("ALTER TABLE validation_labels RENAME COLUMN created_at TO labelled_at"))
            except Exception as sql_err:
                logger.warning("SQLite pre-flight migration warning: %s", sql_err)
        Base.metadata.create_all(bind=eng)
        create_training_labels_view(eng)
        logger.info("Database schema and views initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise

@contextmanager
def get_session():
    """Provides a database session context."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def get_db_session():
    """Provides a resilient database session context for scripts and evaluations."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
