# create_weak_labels.py
import pandas as pd
from preprocess import load_data, build_corpus

def normalize_tokens(s):
    if pd.isna(s): return set()
    return set([t.strip().lower() for t in str(s).split(',') if t.strip()])

def main():
    jobs, users = load_data()
    jobs = build_corpus(jobs)
    rows = []
    for _, user in users.iterrows():
        user_skills = normalize_tokens(user.get('skills',''))
        for _, job in jobs.iterrows():
            job_skills = normalize_tokens(job.get('skills',''))
            relevance = 1 if len(user_skills.intersection(job_skills))>0 else 0
            rows.append({"user_id": int(user['user_id']), "job_id": int(job['job_id']), "relevance": int(relevance)})
    df = pd.DataFrame(rows)
    df.to_csv("../data/labels.csv", index=False)
    print("Weak labels written to ../data/labels.csv  (rows:", len(df), ")")

if __name__ == "__main__":
    main()
