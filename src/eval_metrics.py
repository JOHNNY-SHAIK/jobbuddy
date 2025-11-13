# eval_metrics.py
import numpy as np

def precision_at_k(recommended_job_ids, relevant_job_ids, k=5):
    """Returns Precision@K."""
    recommended = recommended_job_ids[:k]
    hits = sum([1 for r in recommended if r in set(relevant_job_ids)])
    return hits / k

def average_precision_at_k(recommended_job_ids, relevant_job_ids, k=5):
    """Returns Average Precision@K."""
    relevant_set = set(relevant_job_ids)
    score = 0.0
    hits = 0
    for i, jid in enumerate(recommended_job_ids[:k], start=1):
        if jid in relevant_set:
            hits += 1
            score += hits / i
    return score / min(len(relevant_job_ids), k) if len(relevant_job_ids) > 0 else 0.0

def dcg_at_k(recommended_job_ids, gains, k=5):
    dcg = 0.0
    for i, jid in enumerate(recommended_job_ids[:k], start=1):
        rel = gains.get(jid, 0)
        dcg += (2**rel - 1) / np.log2(i + 1)
    return dcg

def ndcg_at_k(recommended_job_ids, gains, k=5):
    dcg = dcg_at_k(recommended_job_ids, gains, k)
    ideal_rels = sorted(gains.values(), reverse=True)[:k]
    ideal = 0.0
    for i, rel in enumerate(ideal_rels, start=1):
        ideal += (2**rel - 1) / np.log2(i + 1)
    return dcg / ideal if ideal > 0 else 0.0
