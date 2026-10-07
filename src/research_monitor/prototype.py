import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from src.research_monitor.config import config

def load_data(filepath):
    return pd.read_csv(filepath)

def get_embeddings(texts, model_name='all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, convert_to_numpy=True)
    # L2 normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1 # Avoid division by zero
    return embeddings / norms

def assign_routes(df):
    n = len(df)
    high_cutoff = int(n * 0.1)
    borderline_cutoff = int(n * 0.4)
    
    def get_route(rank):
        if rank < high_cutoff:
            return 'HIGH'
        elif rank < borderline_cutoff:
            return 'BORDERLINE'
        else:
            return 'LOW'
            
    df['rank'] = np.arange(n)
    df['route'] = df['rank'].apply(get_route)
    return df

def run_prototype():
    print("Loading data...")
    df = load_data('data/sample_papers.csv')
    df['text'] = df['title'] + " " + df['abstract']
    
    # 1. Embed papers
    print("Embedding papers...")
    paper_embeddings = get_embeddings(df['text'].tolist())
    
    # 2. Embed interest
    print("Embedding seed interest...")
    interest_embedding = get_embeddings([config.SEED_INTEREST])[0]
    
    # 3. Calculate similarity (dot product of L2 normalized = cosine similarity)
    print("Calculating similarities...")
    similarities = np.dot(paper_embeddings, interest_embedding)
    df['similarity'] = similarities
    
    # 4. Sort
    df = df.sort_values(by='similarity', ascending=False).reset_index(drop=True)
    
    # 5. Routing
    df = assign_routes(df)
    
    # Outputs
    print(f"\n--- Prototype Run Summary ---")
    print(f"Processed {len(df)} papers.")
    print(f"Embedding dimension: {paper_embeddings.shape[1]}")
    print(f"Example embedding norm (should be 1.0): {np.linalg.norm(paper_embeddings[0]):.4f}")
    
    print("\n--- Route Counts ---")
    print(df['route'].value_counts().to_string())
    
    columns_to_display = ['rank', 'paper_id', 'title', 'similarity', 'route']
    
    print("\n--- Top 10 Papers (HIGH relevance) ---")
    print(df.head(10)[columns_to_display].to_string(index=False))
    
    print("\n--- Bottom 10 Papers (LOW relevance) ---")
    print(df.tail(10)[columns_to_display].to_string(index=False))
    
    return df, paper_embeddings

if __name__ == '__main__':
    run_prototype()
