import os
import sys
import numpy as np
import pyarrow.parquet as pq
import logging
from tqdm import tqdm

# Add src to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from research_monitor.config import config
from research_monitor.database import init_db, get_session, Paper

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s")
logger = logging.getLogger(__name__)

def main():
    logger.info("Initializing database...")
    init_db()

    corpus_path = "data/corpus.parquet"
    embeddings_path = "data/embeddings.npy"

    if not os.path.exists(corpus_path) or not os.path.exists(embeddings_path):
        logger.error(f"Missing artifacts. Ensure {corpus_path} and {embeddings_path} exist.")
        sys.exit(1)

    logger.info(f"Loading corpus from {corpus_path}")
    table = pq.read_table(corpus_path)
    df = table.to_pandas()
    
    logger.info(f"Loading embeddings from {embeddings_path}")
    embeddings = np.load(embeddings_path)
    
    if len(df) != len(embeddings):
        logger.error(f"Mismatch: {len(df)} papers vs {len(embeddings)} embeddings")
        sys.exit(1)

    # Insert in batches
    batch_size = 1000
    
    session_gen = get_session()
    db = next(session_gen)
    
    try:
        # Check if already loaded
        existing = db.query(Paper).count()
        if existing > 0:
            logger.info(f"Database already contains {existing} papers. Dropping all existing papers for a clean load...")
            db.query(Paper).delete()
            db.commit()
            
        logger.info(f"Inserting {len(df)} papers into PostgreSQL...")
        
        for i in tqdm(range(0, len(df), batch_size)):
            batch_df = df.iloc[i:i+batch_size]
            batch_embeddings = embeddings[i:i+batch_size]
            
            papers_to_insert = []
            for (_, row), emb in zip(batch_df.iterrows(), batch_embeddings):
                paper = Paper(
                    id=row['paper_id'],
                    title=row['title'],
                    abstract=row['abstract'],
                    submitted_date=str(row['submitted_date'])[:10], # Keep just the YYYY-MM-DD
                    categories=list(row['categories']),
                    embedding=emb.tolist(),
                    embedding_model=config.EMBEDDING_MODEL,
                    is_reference=row['is_reference']
                )
                papers_to_insert.append(paper)
            
            db.bulk_save_objects(papers_to_insert)
            db.commit()
            
        final_count = db.query(Paper).count()
        logger.info(f"Successfully loaded {final_count} papers into the database.")
        
    except Exception as e:
        logger.error(f"Error during insertion: {e}")
        db.rollback()
        raise
    finally:
        db.close()

if __name__ == "__main__":
    main()
