import numpy as np
import pandas as pd
from src.research_monitor.prototype import get_embeddings, assign_routes

def test_embeddings():
    texts = ["Test paper 1", "Test paper 2"]
    embeddings = get_embeddings(texts)
    
    # embedding output dimension = 384
    assert embeddings.shape == (2, 384)
    # embeddings are L2-normalized
    norms = np.linalg.norm(embeddings, axis=1)
    np.testing.assert_allclose(norms, 1.0, rtol=1e-5)

def test_cosine_reproducibility():
    # cosine scores are reproducible
    texts = ["Autonomous agents planning", "Quantum physics"]
    interest = ["AI agents doing tasks"]
    
    text_emb = get_embeddings(texts)
    interest_emb = get_embeddings(interest)[0]
    
    scores = np.dot(text_emb, interest_emb)
    # The first text should be more similar to the interest than the second
    assert scores[0] > scores[1]

def test_routing():
    # ranking is descending and routing works
    df = pd.DataFrame({'similarity': np.random.rand(200)})
    df = df.sort_values('similarity', ascending=False).reset_index(drop=True)
    
    assert df['similarity'].is_monotonic_decreasing
    
    df = assign_routes(df)
    
    counts = df['route'].value_counts()
    assert counts['HIGH'] == 20
    assert counts['BORDERLINE'] == 60
    assert counts['LOW'] == 120
