# tune_alpha.py
import numpy as np
import pandas as pd
from preprocess import load_data, build_corpus, fit_tfidf
from emb_recommender import EmbeddingRecommender
from sklearn.metrics.pairwise import cosine_similarity
from eval_metrics import precision_at_k

def normalize_scores(scores):
    s = np.array(scores, dtype=float)
    mn, mx = s.min(), s.max()
    if mx - mn == 0:
        return np.zeros_like(s)
    return (s - mn) / (mx - mn)

def main():
    # load jobs + users
    jobs, users = load_data()
    jobs = build_corpus(jobs)
    # TF-IDF
    tfidf_vec, job_tfidf = fit_tfidf(jobs)
    # Embeddings (loads model if needed)
    emb_rec = EmbeddingRecommender(jobs_df=jobs)
    job_embeddings = emb_rec.job_embeddings
    # labels
    labels = pd.read_csv("../data/labels.csv")
    user_ids = sorted(labels['user_id'].unique().tolist())

    alphas = [i/20 for i in range(21)]  # 0.0 .. 1.0
    results = []
    for alpha in alphas:
        p_list = []
        for uid in user_ids:
            user = users[users.user_id==uid].iloc[0]
            user_text = " ".join([str(user.get("title","")), str(user.get("skills","")), str(user.get("bio",""))])
            # TF-IDF sim
            u_tfidf = tfidf_vec.transform([user_text])
            sim_tfidf = cosine_similarity(u_tfidf, job_tfidf).flatten()
            # Emb sim
            u_emb = emb_rec.model.encode([user_text], convert_to_numpy=True)
            from sklearn.metrics.pairwise import cosine_similarity as cos2
            sim_emb = cos2(u_emb, job_embeddings).flatten()
            # normalize and combine
            s_t = normalize_scores(sim_tfidf)
            s_e = normalize_scores(sim_emb)
            final = alpha * s_e + (1.0 - alpha) * s_t
            top_idx = np.argsort(-final)[:5]
            rec_ids = jobs.iloc[top_idx]['job_id'].tolist()
            relevant = labels[(labels.user_id==uid) & (labels.relevance>0)]['job_id'].tolist()
            p5 = precision_at_k(rec_ids, relevant, k=5)
            p_list.append(p5)
        mean_p5 = np.mean(p_list)
        results.append((alpha, mean_p5))
        print(f"alpha={alpha:.2f}  Precision@5={mean_p5:.4f}")
    best = max(results, key=lambda x: x[1])
    print("BEST:", best)

if __name__ == "__main__":
    main()
