"""Load corpus artifacts without deleting existing papers or workflow records."""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from research_monitor.config import config
from research_monitor.database import Paper, get_session, init_db

logger = logging.getLogger(__name__)


def load_corpus(session, corpus, embeddings, batch_size=1000):
    """Insert missing canonical IDs; reruns preserve existing papers and references."""
    vectors = np.asarray(embeddings, dtype=np.float32)
    if vectors.shape != (len(corpus), 384):
        raise ValueError("Expected one 384-dimensional embedding per corpus row.")
    if not np.isfinite(vectors).all() or np.any(np.linalg.norm(vectors, axis=1) == 0):
        raise ValueError("Embeddings must be finite and nonzero.")
    if corpus["paper_id"].duplicated().any():
        raise ValueError("Corpus contains duplicate paper IDs.")
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    inserted = 0
    for start in range(0, len(corpus), batch_size):
        batch = corpus.iloc[start:start + batch_size]
        existing = {pid for (pid,) in session.query(Paper.id).filter(Paper.id.in_(batch["paper_id"].tolist())).all()}
        for offset, (_, row) in enumerate(batch.iterrows()):
            if row["paper_id"] in existing:
                continue
            categories = row["categories"]
            if isinstance(categories, str):
                categories = [c.strip() for c in categories.split(",") if c.strip()]
            session.add(Paper(
                id=str(row["paper_id"]), title=row["title"], abstract=row["abstract"],
                submitted_date=str(row["submitted_date"])[:10], categories=list(categories),
                embedding=vectors[start + offset].tolist(), embedding_model=config.EMBEDDING_MODEL,
                is_reference=bool(row["is_reference"]),
            ))
            inserted += 1
        session.flush()
    session.commit()
    return inserted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-path", default="data/corpus.parquet")
    parser.add_argument("--embeddings-path", default="data/embeddings.npy")
    args = parser.parse_args()
    init_db()
    corpus = pd.read_parquet(args.corpus_path)
    embeddings = np.load(args.embeddings_path, allow_pickle=False)
    with get_session() as session:
        inserted = load_corpus(session, corpus, embeddings)
        logger.info("Loaded %d new papers; existing records preserved.", inserted)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
