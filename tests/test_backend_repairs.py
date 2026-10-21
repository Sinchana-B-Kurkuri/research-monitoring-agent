"""Regressions for failures at module boundaries, beyond individual stages."""
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import Session

from research_monitor.database.models import Base, Paper
from research_monitor.database.session import init_db
from scripts.run_phase3_loader import load_corpus


def test_schema_initialization_is_repeatable_and_training_labels_is_a_view():
    engine = create_engine("sqlite://")
    init_db(engine)
    init_db(engine)
    assert "training_labels" not in Base.metadata.tables
    assert "training_labels" not in inspect(engine).get_table_names()
    assert "training_labels" in inspect(engine).get_view_names()


def test_corpus_loader_is_idempotent_and_preserves_existing_papers():
    engine = create_engine("sqlite://")
    init_db(engine)
    frame = pd.DataFrame([dict(paper_id="2401.00001", title="Paper", abstract="Abstract",
        submitted_date="2024-01-01", categories="cs.AI,cs.CL", is_reference=True)])
    with Session(engine) as db:
        assert load_corpus(db, frame, np.ones((1, 384))) == 1
        assert load_corpus(db, frame, np.ones((1, 384))) == 0
        paper = db.get(Paper, "2401.00001")
        assert paper.categories == ["cs.AI", "cs.CL"]
        assert paper.is_reference
        assert np.isclose(np.linalg.norm(paper.embedding), 1.0, atol=1e-6)
