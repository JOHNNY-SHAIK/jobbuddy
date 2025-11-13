# emb_recommender.py
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from preprocess import load_data, build_corpus

class EmbeddingRecommender:
    def __init__(self, jobs_df=None, model_name="all-MiniLM-L6-v2"):
        # If jobs_df is None, load from data
        if jobs_df is None:
            jobs, _ = load_data()
            jobs = build_corpus(jobs)
            self.jobs_df = jobs.reset_index(drop=True)
        else:
            self.jobs_df = jobs_df.reset_index(drop=True)

        # load SBERT model (will download on first run)
        print("Loading SentenceTransformer model (this may take a while if first run)...")
        self.model = SentenceTransformer(model_name)
        # build embeddings
        print("Encoding job texts (this may take 10-60s)...")
        texts = self.jobs_df['text'].tolist()
        emb = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        # normalize vectors for cosine similarity
        self.job_embeddings = normalize(emb, norm='l2')
        print("Embeddings ready. Number of jobs:", len(self.job_embeddings))

    def recommend(self, user_text, top_n=5):
        uemb = self.model.encode([user_text], convert_to_numpy=True)
        uemb = normalize(uemb, norm='l2')
        sims = cosine_similarity(uemb, self.job_embeddings).flatten()
        top_idx = np.argsort(-sims)[:top_n]
        results = self.jobs_df.iloc[top_idx].copy()
        results['score'] = sims[top_idx]
        return results
