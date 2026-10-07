"""
tests/test_ingestion.py
~~~~~~~~~~~~~~~~~~~~~~~
Phase 2 unit tests for the ingestion pipeline.

Covers:
  - version-stripped ID deduplication
  - text preprocessing / normalization
  - withdrawn record filtering
  - empty-abstract filtering
  - embedding dimensionality (384)
  - L2 normalization
  - deterministic reference sampling
  - reference papers excluded from the labelling frame
  - batch processing correctness
"""

import numpy as np
import pandas as pd
import pytest

from research_monitor.ingestion.preprocessor import (
    strip_version,
    normalize_text,
    is_withdrawn,
    preprocess,
)
from research_monitor.ingestion.embedder import embed_texts, l2_normalize
from research_monitor.ingestion.reference_sample import (
    create_reference_sample,
    get_labelling_frame,
    save_reference_ids,
    load_reference_ids,
)


# ── Preprocessor tests ────────────────────────────────────────────────────────

class TestStripVersion:
    def test_url_with_version(self):
        assert strip_version("https://arxiv.org/abs/2401.12345v2") == "2401.12345"

    def test_plain_id_with_version(self):
        assert strip_version("2401.12345v3") == "2401.12345"

    def test_old_style_id(self):
        assert strip_version("cs/0612007v1") == "cs/0612007"

    def test_no_version(self):
        assert strip_version("2401.12345") == "2401.12345"

    def test_url_no_version(self):
        assert strip_version("https://arxiv.org/abs/2401.12345") == "2401.12345"


class TestNormalizeText:
    def test_strips_whitespace(self):
        assert normalize_text("  hello   world  ") == "hello world"

    def test_collapses_newlines(self):
        assert normalize_text("line one\n\nline two") == "line one line two"

    def test_collapses_tabs(self):
        assert normalize_text("a\t\tb") == "a b"

    def test_nfc_normalization(self):
        # é composed vs decomposed should both become "é"
        composed = "\u00e9"          # NFC é
        decomposed = "e\u0301"       # NFD é
        assert normalize_text(decomposed) == composed


class TestIsWithdrawn:
    def test_detects_withdrawn_uppercase(self):
        assert is_withdrawn("This paper has been [WITHDRAWN].")

    def test_detects_withdrawn_lowercase(self):
        assert is_withdrawn("[withdrawn] by authors.")

    def test_normal_abstract_not_withdrawn(self):
        assert not is_withdrawn("We propose a new method for NLP.")


class TestPreprocess:
    def _make_paper(self, paper_id, title="Title", abstract="Abstract text."):
        return {
            "paper_id": paper_id,
            "title": title,
            "abstract": abstract,
            "categories": "cs.CL",
            "submitted_date": "2023-01-01",
        }

    def test_version_stripped_deduplication(self):
        """Two papers with the same base ID but different versions → only one kept."""
        papers = [
            self._make_paper("https://arxiv.org/abs/2401.00001v1"),
            self._make_paper("https://arxiv.org/abs/2401.00001v2"),
        ]
        result = list(preprocess(papers))
        assert len(result) == 1
        assert result[0]["paper_id"] == "2401.00001"

    def test_withdrawn_removed(self):
        papers = [
            self._make_paper("2401.00001", abstract="This paper is [WITHDRAWN]."),
            self._make_paper("2401.00002", abstract="Valid abstract."),
        ]
        result = list(preprocess(papers))
        assert len(result) == 1
        assert result[0]["paper_id"] == "2401.00002"

    def test_empty_abstract_removed(self):
        papers = [
            self._make_paper("2401.00001", abstract=""),
            self._make_paper("2401.00002", abstract="Valid abstract."),
        ]
        result = list(preprocess(papers))
        assert len(result) == 1

    def test_whitespace_normalized(self):
        papers = [self._make_paper("2401.00001", title="  A   Title  ",
                                   abstract="Some  abstract\nwith newlines.")]
        result = list(preprocess(papers))
        assert result[0]["title"] == "A Title"
        assert result[0]["abstract"] == "Some abstract with newlines."

    def test_valid_papers_pass_through(self):
        papers = [self._make_paper(f"2401.{i:05d}") for i in range(5)]
        result = list(preprocess(papers))
        assert len(result) == 5

    def test_batch_of_mixed_papers(self):
        """Combined test: dedup + withdrawn + empty in one batch."""
        papers = [
            self._make_paper("2401.00001v1"),            # kept
            self._make_paper("2401.00001v2"),            # duplicate → removed
            self._make_paper("2401.00002", abstract="[WITHDRAWN]"),  # withdrawn
            self._make_paper("2401.00003", abstract=""),              # empty
            self._make_paper("2401.00004"),               # kept
        ]
        result = list(preprocess(papers))
        assert len(result) == 2
        ids = {r["paper_id"] for r in result}
        assert ids == {"2401.00001", "2401.00004"}


# ── Embedder tests ─────────────────────────────────────────────────────────────

class TestEmbedder:
    def test_embedding_dimension(self):
        """Embeddings must be exactly 384-dimensional."""
        texts = ["AI agents for autonomous planning.", "Deep learning for vision."]
        emb = embed_texts(texts)
        assert emb.shape == (2, 384)

    def test_l2_normalized(self):
        """Every embedding must have unit L2 norm."""
        texts = ["Reinforcement learning.", "Natural language processing."]
        emb = embed_texts(texts)
        norms = np.linalg.norm(emb, axis=1)
        np.testing.assert_allclose(norms, 1.0, rtol=1e-5)

    def test_l2_normalize_function(self):
        """l2_normalize on raw vectors should produce unit norms."""
        raw = np.array([[3.0, 4.0], [0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        normed = l2_normalize(raw)
        assert pytest.approx(np.linalg.norm(normed[0]), abs=1e-6) == 1.0
        # zero vector should survive unchanged (norm stays 0 to avoid NaN)
        np.testing.assert_array_equal(normed[1], [0.0, 0.0])

    def test_reproducibility(self):
        """Same text → same embedding every call."""
        text = ["Multi-agent reinforcement learning systems."]
        emb1 = embed_texts(text)
        emb2 = embed_texts(text)
        np.testing.assert_array_equal(emb1, emb2)


# ── Reference sample tests ─────────────────────────────────────────────────────

def _make_corpus(n: int) -> pd.DataFrame:
    return pd.DataFrame({
        "paper_id": [f"2401.{i:05d}" for i in range(n)],
        "title": [f"Paper {i}" for i in range(n)],
        "abstract": [f"Abstract for paper {i}." for i in range(n)],
    })


class TestReferenceSample:
    def test_exact_sample_size(self):
        df = _make_corpus(1000)
        _, ref = create_reference_sample(df, sample_size=100, seed=42)
        assert len(ref) == 100

    def test_deterministic_same_seed(self):
        """Same corpus + same seed → identical paper IDs."""
        df = _make_corpus(1000)
        _, ref1 = create_reference_sample(df, sample_size=100, seed=42)
        _, ref2 = create_reference_sample(df, sample_size=100, seed=42)
        assert set(ref1["paper_id"]) == set(ref2["paper_id"])

    def test_different_seeds_produce_different_samples(self):
        df = _make_corpus(1000)
        _, ref1 = create_reference_sample(df, sample_size=100, seed=42)
        _, ref2 = create_reference_sample(df, sample_size=100, seed=99)
        assert set(ref1["paper_id"]) != set(ref2["paper_id"])

    def test_is_reference_column_added(self):
        df = _make_corpus(1000)
        corpus, ref = create_reference_sample(df, sample_size=100, seed=42)
        assert "is_reference" in corpus.columns
        assert corpus["is_reference"].sum() == 100

    def test_reference_excluded_from_labelling_frame(self):
        df = _make_corpus(1000)
        corpus, ref = create_reference_sample(df, sample_size=100, seed=42)
        labelling = get_labelling_frame(corpus)
        ref_ids = set(ref["paper_id"])
        labelling_ids = set(labelling["paper_id"])
        assert ref_ids.isdisjoint(labelling_ids), \
            "Reference papers must not appear in the labelling frame."
        assert len(labelling) == 900

    def test_insufficient_corpus_raises(self):
        df = _make_corpus(50)
        with pytest.raises(ValueError, match="only 50 papers"):
            create_reference_sample(df, sample_size=100, seed=42)

    def test_save_and_load_reference_ids(self, tmp_path):
        df = _make_corpus(200)
        _, ref = create_reference_sample(df, sample_size=50, seed=42)
        out = tmp_path / "ref_ids.txt"
        save_reference_ids(ref, out)
        loaded = load_reference_ids(out)
        assert loaded == set(ref["paper_id"])
