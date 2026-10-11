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

## Phase 3 — PostgreSQL + pgvector
- **Database**: Introduced local PostgreSQL + pgvector via Docker Compose (`pgvector/pgvector:pg16`).
- **ORM**: SQLAlchemy is used for basic model mapping and bulk insertion. No Alembic migrations yet as the schema is simple and initial.
- **Artifact Preservation**: The Phase 2 Parquet and NumPy artifacts are loaded into the database but preserved as the original source of truth. 
- **Vector Search**: The `embedding` column uses pgvector's `Vector(384)`. Native operators like `<->` (L2 distance) are supported, but formal vector indices (e.g. HNSW/IVFFlat) are omitted at this scale per roadmap constraints.

## Phase 4 — Ground Truth, Baselines, and Threshold Calibration
- **Evaluation Sampling Frame**: Evaluation papers are drawn strictly from the 5,000 non-reference corpus papers (`is_reference=False`), guaranteeing zero contamination of the reference sample.
- **Reference-Percentile Strata**: Evaluation papers are partitioned into five strata defined by the reference score percentiles:
  - S1: $\ge P98$ (100 papers, default region: HIGH)
  - S2: $P90 - P98$ (120 papers, default region: BORDERLINE)
  - S3: $P75 - P90$ (100 papers, default region: LOW)
  - S4: $P50 - P75$ (80 papers, default region: LOW)
  - S5: $< P50$ (100 papers, default region: LOW)
  Total sample = 500 papers. Sampling weight $w_h = N_h / n_h$ guarantees exact population reproduction ($\sum n_h w_h = 5,000$).
- **Dev / Test Split**: 70% development (350 papers) and 30% test (150 papers) split within each stratum with a fixed seed (`seed=42`).
- **Frozen Test Set & Audit Logging**: The test split is frozen before calibration. Every test set access is audited in `experiments/results/test_access.log`.
- **Three-Point Ground Truth Labels**: Human labels support three categories: `relevant`, `somewhat`, and `not_relevant`. Never fabricate artificial model predictions as ground truth.
- **Primary vs. Lenient Mappings**: Primary binary mapping treats `relevant=1`, `somewhat=0`, `not_relevant=0`. Lenient mapping treats `relevant=1`, `somewhat=1`, `not_relevant=0`.
- **Threshold Calibration**:
  - Calibrated strictly on the development split using the Section 7.2 objective:
    - Choose the smallest $P_{\text{high}}$ with weighted HIGH precision $\ge 0.80$.
    - Choose the largest $P_{\text{low}} < P_{\text{high}}$ with weighted LOW miss rate $\le 0.10$ (aiming for $\le 0.05$).
  - Provisional levels before calibration are preserved: $P_{\text{high}} = 98$, $P_{\text{low}} = 90$.
  - Calibrated levels are stored in configuration and documentation.
- **Baselines**:
  - B0: Random routing baseline at matched volume.
  - B1: Keyword/TF-IDF cosine similarity to seed-interest text.
  - B2: Embedding cosine similarity to fixed seed vector.
- **Data Protection**: `data/labels/` CSV backups and `experiments/results/*.log` are gitignored to prevent leaking human labels or access history.
