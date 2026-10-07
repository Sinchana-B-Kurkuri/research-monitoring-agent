# Architectural Decisions

## Phase 0 Foundation

- **Language**: Python
- **Layout**: Modular `src/` layout to separate code from configuration and tests.
- **Configuration**: Environment variables are used for settings and secrets (loaded from `.env` via `python-dotenv`). No secrets are committed to the repository.
- **Database**: PostgreSQL/pgvector will be introduced in a later phase.
- **Testing**: Tests are required for important logic. `pytest` is the chosen framework.
- **Version Control**: Git is managed manually by the developer.

## Phase 2 — Real Corpus, Reference Sample, and Embedding Pipeline

- **arXiv source**: Papers fetched via the official `arxiv` Python client (OAI-style API).
  This is lightweight, reproducible, and requires no large local file download.
- **Corpus date range**: 2020-01-01 to 2024-12-31. This window captures the LLM and autonomous-agent research era. Override via `CORPUS_START_DATE` / `CORPUS_END_DATE` in `.env`.
- **Target categories**: `cs.CL`, `cs.LG`, `cs.AI`. Override via `CORPUS_CATEGORIES` in `.env`.
- **Incremental streaming**: The fetch and preprocess pipeline streams papers one-by-one so memory footprint stays flat at any corpus scale.
- **Deduplication**: Uses the version-stripped arXiv ID (e.g. `2401.12345` not `2401.12345v2`). First seen wins.
- **Parquet format**: The processed corpus is stored as Parquet (`data/corpus.parquet`) via PyArrow. Parquet is columnar, compressed, and efficient for the downstream embedding and threshold-calibration steps.
- **Embeddings**: Stored as NumPy `.npy` arrays (`data/embeddings.npy`). This avoids a database dependency until Phase 3 introduces pgvector.
- **Reference sample**: 5,000 papers selected with `numpy.random.default_rng(seed=42)`. The seed and size are configurable via `.env`. The same corpus + seed always produces the same IDs. Reference papers are marked `is_reference=True` in the corpus and their IDs are written to `data/reference_sample_ids.txt`.
- **Human-labelling frame**: Reference papers are explicitly excluded from `get_labelling_frame()` to prevent contamination of future threshold calibration.
- **Large artifacts not committed**: `*.parquet`, `*.npy`, `*.npz`, and `reference_sample_ids.txt` are gitignored.
