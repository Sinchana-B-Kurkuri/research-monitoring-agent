import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import numpy as np

from research_monitor.database.models import Base, Paper
from research_monitor.config import DATABASE_URL

# For testing, we can use the main DB if it's running, or skip if it's not.
# We'll try to connect to the DB. If it fails, we skip these integration tests.
try:
    engine = create_engine(DATABASE_URL)
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    DB_AVAILABLE = True
except Exception:
    DB_AVAILABLE = False

@pytest.fixture(scope="module")
def test_db():
    if not DB_AVAILABLE:
        pytest.skip("Database not available for integration tests. Run docker-compose up first.")
    
    # Initialize schema
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
    Base.metadata.create_all(bind=engine)
    
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = SessionLocal()
    
    # Clean up test data before
    db.query(Paper).filter(Paper.id.like('test-%')).delete()
    db.commit()
    
    yield db
    
    # Clean up test data after
    db.query(Paper).filter(Paper.id.like('test-%')).delete()
    db.commit()
    db.close()

def test_paper_model_instantiation():
    """Unit test for Paper model attributes."""
    emb = np.random.rand(384).astype(np.float32).tolist()
    p = Paper(
        id="test-123",
        title="Test Paper",
        abstract="Test Abstract",
        submitted_date="2024-01-01",
        categories=["cs.AI"],
        embedding=emb,
        embedding_model="test-model",
        is_reference=True
    )
    assert p.id == "test-123"
    assert len(p.embedding) == 384
    assert p.is_reference is True

def test_db_insert_and_retrieve(test_db):
    """Integration test: Insert a paper and retrieve it."""
    emb = np.random.rand(384).astype(np.float32).tolist()
    
    p = Paper(
        id="test-integration-1",
        title="Integration Test Paper",
        abstract="Integration Abstract",
        submitted_date="2024-01-01",
        categories=["cs.AI"],
        embedding=emb,
        embedding_model="test-model",
        is_reference=False
    )
    
    test_db.add(p)
    test_db.commit()
    
    retrieved = test_db.query(Paper).filter(Paper.id == "test-integration-1").first()
    assert retrieved is not None
    assert retrieved.title == "Integration Test Paper"
    assert len(retrieved.embedding) == 384
    assert retrieved.is_reference is False
    assert retrieved.categories == ["cs.AI"]

def test_vector_similarity(test_db):
    """Integration test: Basic vector similarity functionality using pgvector L2 distance."""
    # Insert two papers
    emb1 = np.zeros(384, dtype=np.float32)
    emb1[0] = 1.0 # [1, 0, 0...]
    
    emb2 = np.zeros(384, dtype=np.float32)
    emb2[0] = 0.5 # [0.5, 0, 0...]
    
    p1 = Paper(
        id="test-vec-1", title="Vec 1", abstract="Abs 1", submitted_date="2024-01-01",
        categories=["cs.AI"], embedding=emb1.tolist(), embedding_model="test", is_reference=True
    )
    p2 = Paper(
        id="test-vec-2", title="Vec 2", abstract="Abs 2", submitted_date="2024-01-01",
        categories=["cs.AI"], embedding=emb2.tolist(), embedding_model="test", is_reference=True
    )
    test_db.add_all([p1, p2])
    test_db.commit()
    
    # Query ordering by L2 distance to emb1
    # <-> is the L2 distance operator in pgvector
    results = test_db.query(Paper).filter(Paper.id.in_(["test-vec-1", "test-vec-2"])) \
                     .order_by(Paper.embedding.l2_distance(emb1.tolist())) \
                     .all()
    
    assert len(results) == 2
    # The closest to emb1 should be emb1 (distance 0)
    assert results[0].id == "test-vec-1"
    # Next should be emb2
    assert results[1].id == "test-vec-2"
