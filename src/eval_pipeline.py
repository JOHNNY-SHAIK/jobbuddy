# eval_pipeline.py
import pandas as pd
import numpy as np
from preprocess import load_data, build_corpus, fit_tfidf
from emb_recommender import EmbeddingRecommender
from hybrid_recommender import HybridRecommender
from eval_metrics import precision_at_k
from sklearn.model_selection import train_test_split

def load_labels(path="../data/labels.csv"):
    return pd.read_csv(path)

def build_splits(labels, test_size=0.2, random_state=42):
    # split by user-job pairs
    train, test = train_test_split(labels, test_size=test_size, random_state=random_state, stratify=labels['user_id'])
    return train, test

def evaluate_on_labels(hybrid, labels_df):
    jobs, users = hybrid.jobs, None
    # compute Precision@5 per user using labels_df
    ps = []
    for uid in sorted(labels_df['user_id'].unique()):
        user_row = pd.read_csv("../data/users_sample.csv")
        user = user_row[user_row.user_id==uid].iloc[0]
        user_text = " ".join([str(user.get("title","")), str(user.get("skills","")), str(user.get("bio",""))])
        res = hybrid.recommend(user_text, top_n=5)
        rec_ids = res['job_id'].tolist()
        rel_ids = labels_df[(labels_df.user_id==uid) & (labels_df.relevance>0)]['job_id'].tolist()
        p5 = precision_at_k(rec_ids, rel_ids, k=5)
        ps.append(p5)
    return np.mean(ps)

if __name__ == "__main__":
    labels = load_labels()
    train, test = build_splits(labels, test_size=0.25)
    print("Train rows:", len(train), "Test rows:", len(test))
    # pick alpha (start with 0.5, we will tune)
    hybrid = HybridRecommender(alpha=0.5)
    print("Eval on test (Precision@5):", evaluate_on_labels(hybrid, test))
