# src/recommender.py
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from preprocess import load_data, build_corpus, fit_tfidf

class JobRecommender:
    def __init__(self, jobs_df, vectorizer, job_tfidf):
        self.jobs_df = jobs_df.reset_index(drop=True)
        self.vectorizer = vectorizer
        self.job_tfidf = job_tfidf

    def _user_vector(self, user_text):
        return self.vectorizer.transform([user_text])

    def recommend(self, user_text, top_n=5):
        uvec = self._user_vector(user_text)
        sims = cosine_similarity(uvec, self.job_tfidf).flatten()
        top_idx = np.argsort(-sims)[:top_n]
        results = self.jobs_df.iloc[top_idx].copy()
        results['score'] = sims[top_idx]
        return results

def build_recommender():
    jobs, _ = load_data()
    jobs = build_corpus(jobs)
    vectorizer, job_tfidf = fit_tfidf(jobs)
    return JobRecommender(jobs, vectorizer, job_tfidf)
