# src/hybrid_recommender.py

import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from preprocess import load_data, build_corpus, fit_tfidf, skill_set
from emb_recommender import EmbeddingRecommender


class HybridRecommender:
    def __init__(self, alpha=0.5):
        """
        alpha = 0 → TF-IDF only
        alpha = 1 → SBERT only
        """
        self.alpha = alpha

        # Load data
        jobs, _ = load_data()
        jobs = build_corpus(jobs)

        # Store jobs
        self.jobs = jobs

        # TF-IDF
        self.tfidf_vec, self.job_tfidf = fit_tfidf(jobs)

        # SBERT embeddings
        # EmbeddingRecommender will internally load the SBERT model
        self.emb_rec = EmbeddingRecommender()
        self.job_embeddings = self.emb_rec.model.encode(
            jobs["text"].tolist(), convert_to_numpy=True
        )

    @staticmethod
    def normalize(arr):
        arr = np.array(arr, dtype=float)
        if arr.max() - arr.min() == 0:
            return np.zeros_like(arr)
        return (arr - arr.min()) / (arr.max() - arr.min())

    def recommend(self, user_text, top_n=5, beta=0.5, user_skills_str=None):
        """
        Generate job recommendations using hybrid:
        TF-IDF + SBERT + skill-overlap boost.

        If `user_skills_str` is provided (comma-separated skills from the UI),
        we use that for overlap calculation. Otherwise we fall back to parsing user_text.
        """
        # -------- TF-IDF similarity --------
        u_tfidf = self.tfidf_vec.transform([user_text])
        sim_tfidf = cosine_similarity(u_tfidf, self.job_tfidf).flatten()

        # -------- SBERT similarity --------
        u_emb = self.emb_rec.model.encode([user_text], convert_to_numpy=True)
        sim_emb = cosine_similarity(u_emb, self.job_embeddings).flatten()

        # Normalize
        s_t = self.normalize(sim_tfidf)
        s_e = self.normalize(sim_emb)

        # Combined semantic/lexical score
        final = self.alpha * s_e + (1 - self.alpha) * s_t

        # -------- Skill-overlap boost --------
        # Prefer explicit skills string from UI (comma-separated)
        if user_skills_str and str(user_skills_str).strip():
            user_skills = skill_set(user_skills_str)
        else:
            user_skills = skill_set(user_text)

        overlaps = []
        for s in self.jobs["skill_set"]:
            try:
                # s should be a set; handle empty or malformed values
                if not s:
                    overlaps.append(0.0)
                else:
                    inter = user_skills.intersection(s)
                    overlaps.append(len(inter) / max(1, len(s)))
            except Exception:
                # fallback: parse string form
                parsed = skill_set(str(s))
                inter = user_skills.intersection(parsed)
                overlaps.append(len(inter) / max(1, len(parsed)))

        overlaps = np.array(overlaps, dtype=float)

        # Boost
        final = final + beta * overlaps

        # Normalize final score again for display
        if final.max() - final.min() > 0:
            final = (final - final.min()) / (final.max() - final.min())

        # -------- Top results --------
        idx = np.argsort(-final)[:top_n]
        results = self.jobs.iloc[idx].copy()
        results["score"] = final[idx]

        # compute matched skills for UI convenience (inside function)
        matched = []
        for j_sk in results["skill_set"]:
            try:
                intersection = user_skills.intersection(j_sk)
            except Exception:
                intersection = user_skills.intersection(skill_set(str(j_sk)))
            matched.append(", ".join(sorted(intersection)))
        results["matched_skills"] = matched

        return results
