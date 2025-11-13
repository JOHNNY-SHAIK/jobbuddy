# hybrid_recommender.py
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from preprocess import load_data, build_corpus, fit_tfidf
from emb_recommender import EmbeddingRecommender

class HybridRecommender:
    def __init__(self, alpha=0.0):
        # Load data
        jobs, _ = load_data()
        jobs = build_corpus(jobs)

        # Save dataset
        self.jobs = jobs.reset_index(drop=True)
        self.alpha = alpha

        # Build TF-IDF part
        self.tfidf_vec, self.job_tfidf = fit_tfidf(jobs)

        # Build SBERT embedding part
        self.emb_rec = EmbeddingRecommender(jobs_df=jobs)
        self.job_embeddings = self.emb_rec.job_embeddings

        print(f"Hybrid recommender ready. Alpha = {self.alpha}")

    def normalize(self, scores):
        s = np.array(scores, dtype=float)
        mn, mx = s.min(), s.max()
        if mx - mn == 0:
            return np.zeros_like(s)
        return (s - mn) / (mx - mn)

    def recommend(self, user_text, top_n=5):
        # TF-IDF similarity
        u_tfidf = self.tfidf_vec.transform([user_text])
        sim_tfidf = cosine_similarity(u_tfidf, self.job_tfidf).flatten()

        # Embedding similarity
        u_emb = self.emb_rec.model.encode([user_text], convert_to_numpy=True)
        sim_emb = cosine_similarity(u_emb, self.job_embeddings).flatten()

        # Normalize and combine
        s_t = self.normalize(sim_tfidf)
        s_e = self.normalize(sim_emb)

        final = self.alpha * s_e + (1 - self.alpha) * s_t
        top_idx = np.argsort(-final)[:top_n]

        results = self.jobs.iloc[top_idx].copy()
        results['score'] = final[top_idx]
        return results
