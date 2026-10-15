import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import numpy as np

from datetime import datetime, timezone, timedelta
from pathlib import Path
import tempfile
import pandas as pd

from research_monitor.database.models import Base, Paper, ValidationLabel
from research_monitor.database.session import init_db
from research_monitor.evaluation.labeling import (
    export_labels_to_csv,
    get_labeled_paper_ids,
    load_latest_labels,
    normalize_paper_id,
    reconcile_csv_and_db_labels,
    record_validation_label,
)
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


def test_validation_label_postgres_insert_and_update(test_db):
    """Regression test: Saving and updating a ValidationLabel in PostgreSQL."""
    from sqlalchemy.exc import IntegrityError

    # Ensure schema migrations are applied
    init_db(engine)

    test_pid = "test-postgres-val-001"
    # Clean up any leftover test data
    test_db.query(ValidationLabel).filter(ValidationLabel.paper_id == test_pid).delete()
    test_db.commit()

    try:
        # 1. Insert new label via record_validation_label
        rec1 = record_validation_label(
            paper_id=test_pid,
            label="relevant",
            stratum="S1",
            sampling_weight=1.5,
            split="dev",
            notes="Initial ground truth note",
            session=test_db,
            user_id="developer",
        )
        assert rec1["label"] == "relevant"

        # Verify persisted record in PostgreSQL
        retrieved = (
            test_db.query(ValidationLabel)
            .filter_by(paper_id=test_pid, user_id="developer")
            .first()
        )
        assert retrieved is not None
        assert retrieved.user_id == "developer"
        assert retrieved.paper_id == test_pid
        assert retrieved.label == "relevant"
        assert retrieved.stratum == "S1"
        assert retrieved.sampling_stratum == "S1"
        assert retrieved.sampling_weight == pytest.approx(1.5)
        assert retrieved.split == "dev"
        assert retrieved.notes == "Initial ground truth note"
        assert retrieved.labelled_at is not None
        assert retrieved.created_at is not None

        # 2. Update existing label via record_validation_label
        rec2 = record_validation_label(
            paper_id=test_pid,
            label="somewhat",
            stratum="S1",
            sampling_weight=1.5,
            split="dev",
            notes="Updated review note",
            session=test_db,
            user_id="developer",
        )
        assert rec2["label"] == "somewhat"

        # Verify record was updated in-place without duplicate
        all_records = (
            test_db.query(ValidationLabel)
            .filter_by(paper_id=test_pid, user_id="developer")
            .all()
        )
        assert len(all_records) == 1
        updated = all_records[0]
        assert updated.label == "somewhat"
        assert updated.notes == "Updated review note"

        # 3. Verify unique constraint on (user_id, paper_id)
        duplicate_lbl = ValidationLabel(
            user_id="developer",
            paper_id=test_pid,
            label="not_relevant",
            sampling_stratum="S1",
            sampling_weight=1.5,
            split="dev",
        )
        test_db.add(duplicate_lbl)
        with pytest.raises(IntegrityError):
            test_db.commit()
        test_db.rollback()

    finally:
        # Clean up test label
        test_db.query(ValidationLabel).filter(ValidationLabel.paper_id == test_pid).delete()
        test_db.commit()


def test_csv_only_label_safely_reconciled_with_postgres(test_db):
    """Regression test: CSV-only label (e.g. from DB outage) safely reconciled with PostgreSQL."""
    test_pid = "test-csv-reconcile-001"
    test_db.query(ValidationLabel).filter(ValidationLabel.paper_id == test_pid).delete()
    test_db.commit()

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_records = [
                {
                    "user_id": "developer",
                    "paper_id": test_pid,
                    "label": "relevant",
                    "sampling_stratum": "S2",
                    "sampling_weight": 2.5,
                    "split": "dev",
                    "notes": "CSV only label note",
                    "labelled_at": "2026-10-15 10:00:00 UTC",
                }
            ]
            export_labels_to_csv(csv_records, output_dir=tmpdir)

            # Reconcile into PostgreSQL
            stats = reconcile_csv_and_db_labels(session=test_db, output_dir=tmpdir)
            assert stats["inserted"] >= 1

            # Verify persisted in PostgreSQL
            db_rec = (
                test_db.query(ValidationLabel)
                .filter_by(paper_id=test_pid, user_id="developer")
                .first()
            )
            assert db_rec is not None
            assert db_rec.label == "relevant"
            assert db_rec.sampling_stratum == "S2"
            assert db_rec.sampling_weight == pytest.approx(2.5)
            assert db_rec.notes == "CSV only label note"

    finally:
        test_db.query(ValidationLabel).filter(ValidationLabel.paper_id == test_pid).delete()
        test_db.commit()


def test_reconciliation_preserves_newer_db_label(test_db):
    """Regression test: Newer DB label is never overwritten by an older CSV label."""
    test_pid = "test-newer-db-001"
    test_db.query(ValidationLabel).filter(ValidationLabel.paper_id == test_pid).delete()
    test_db.commit()

    try:
        # DB has a newer label from 12:00:00 UTC
        now_dt = datetime(2026, 10, 15, 12, 0, 0, tzinfo=timezone.utc)
        older_dt_str = "2026-10-15 08:00:00 UTC"

        db_label = ValidationLabel(
            user_id="developer",
            paper_id=test_pid,
            label="relevant",
            sampling_stratum="S1",
            sampling_weight=1.0,
            split="test",
            notes="Newer DB label",
            labelled_at=now_dt,
            updated_at=now_dt,
        )
        test_db.add(db_label)
        test_db.commit()

        with tempfile.TemporaryDirectory() as tmpdir:
            # CSV has an older label from 08:00:00 UTC with a conflicting label
            csv_records = [
                {
                    "user_id": "developer",
                    "paper_id": test_pid,
                    "label": "not_relevant",
                    "sampling_stratum": "S1",
                    "sampling_weight": 1.0,
                    "split": "test",
                    "notes": "Older CSV label",
                    "labelled_at": older_dt_str,
                }
            ]
            export_labels_to_csv(csv_records, output_dir=tmpdir)

            stats = reconcile_csv_and_db_labels(session=test_db, output_dir=tmpdir)
            assert stats["preserved_db"] >= 1

            # Verify DB label was NOT overwritten
            refreshed = (
                test_db.query(ValidationLabel)
                .filter_by(paper_id=test_pid, user_id="developer")
                .first()
            )
            assert refreshed.label == "relevant"
            assert refreshed.notes == "Newer DB label"

    finally:
        test_db.query(ValidationLabel).filter(ValidationLabel.paper_id == test_pid).delete()
        test_db.commit()


def test_skip_labeled_paper_and_present_unlabeled_in_postgres(test_db):
    """Regression test: Check that DB-labeled paper is skipped and unlabeled is presented."""
    p_labeled = "test-skip-db-001"
    p_unlabeled = "test-unlabeled-db-002"
    test_db.query(ValidationLabel).filter(
        ValidationLabel.paper_id.in_([p_labeled, p_unlabeled])
    ).delete()
    test_db.commit()

    try:
        record_validation_label(
            paper_id=p_labeled,
            label="relevant",
            stratum="S1",
            sampling_weight=1.0,
            split="dev",
            session=test_db,
            user_id="developer",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            labeled_set = get_labeled_paper_ids(
                session=test_db, user_id="developer", output_dir=tmpdir
            )
            assert p_labeled in labeled_set
            assert p_unlabeled not in labeled_set

            sample_df = pd.DataFrame([
                {"paper_id": p_labeled, "title": "Labeled Paper", "stratum": "S1"},
                {"paper_id": p_unlabeled, "title": "Unlabeled Paper", "stratum": "S1"},
            ])
            sample_df["_norm_pid"] = sample_df["paper_id"].apply(normalize_paper_id)
            unlabeled_df = sample_df[~sample_df["_norm_pid"].isin(labeled_set)]

            assert len(unlabeled_df) == 1
            assert unlabeled_df.iloc[0]["paper_id"] == p_unlabeled

    finally:
        test_db.query(ValidationLabel).filter(
            ValidationLabel.paper_id.in_([p_labeled, p_unlabeled])
        ).delete()
        test_db.commit()


def test_multi_user_label_isolation_in_postgres(test_db):
    """Regression test: Multiple users' labels remain isolated in PostgreSQL."""
    test_pid = "test-multi-user-001"
    test_db.query(ValidationLabel).filter(ValidationLabel.paper_id == test_pid).delete()
    test_db.commit()

    try:
        # User Alpha labels as relevant
        record_validation_label(
            paper_id=test_pid,
            label="relevant",
            stratum="S1",
            sampling_weight=1.0,
            split="dev",
            session=test_db,
            user_id="user_alpha",
        )
        # User Beta labels as not_relevant
        record_validation_label(
            paper_id=test_pid,
            label="not_relevant",
            stratum="S1",
            sampling_weight=1.0,
            split="dev",
            session=test_db,
            user_id="user_beta",
        )

        records = (
            test_db.query(ValidationLabel)
            .filter_by(paper_id=test_pid)
            .order_by(ValidationLabel.user_id)
            .all()
        )
        assert len(records) == 2
        assert records[0].user_id == "user_alpha"
        assert records[0].label == "relevant"
        assert records[1].user_id == "user_beta"
        assert records[1].label == "not_relevant"

        # Check queue isolation
        with tempfile.TemporaryDirectory() as tmpdir:
            alpha_labeled = get_labeled_paper_ids(
                session=test_db, user_id="user_alpha", output_dir=tmpdir
            )
            beta_labeled = get_labeled_paper_ids(
                session=test_db, user_id="user_beta", output_dir=tmpdir
            )
            gamma_labeled = get_labeled_paper_ids(
                session=test_db, user_id="user_gamma", output_dir=tmpdir
            )

            assert test_pid in alpha_labeled
            assert test_pid in beta_labeled
            assert test_pid not in gamma_labeled

    finally:
        test_db.query(ValidationLabel).filter(ValidationLabel.paper_id == test_pid).delete()
        test_db.commit()
