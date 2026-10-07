from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import logging

from research_monitor.config import DATABASE_URL
from research_monitor.database.models import Base

logger = logging.getLogger(__name__)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    """Initializes the database schema and pgvector extension."""
    try:
        with engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            logger.info("Ensured pgvector extension is enabled.")
        Base.metadata.create_all(bind=engine)
        logger.info("Database schema initialized.")
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
