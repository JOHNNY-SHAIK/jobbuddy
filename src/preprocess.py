# preprocess.py
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def load_data(jobs_path=None, users_path=None):
    src_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(src_dir, ".."))
    data_dir = os.path.join(project_root, "data")
    if jobs_path is None:
        jobs_path = os.path.join(data_dir, "jobs_sample.csv")
    if users_path is None:
        users_path = os.path.join(data_dir, "users_sample.csv")
    if not os.path.exists(jobs_path):
        raise FileNotFoundError(f"Jobs file not found at: {jobs_path}")
    if not os.path.exists(users_path):
        raise FileNotFoundError(f"Users file not found at: {users_path}")
    jobs = pd.read_csv(jobs_path)
    users = pd.read_csv(users_path)
    return jobs, users

def normalize_skills(skill_str):
    if pd.isna(skill_str):
        return ''
    tokens = [s.strip().lower() for s in str(skill_str).split(',') if s.strip()]
    mapping = {
        'js': 'javascript', 'py': 'python', 'ml': 'machine learning',
        'nlp': 'natural language processing', 'tf': 'tensorflow',
        'pytorch': 'pytorch', 'pd': 'pandas', 'reactjs': 'react',
        'ds': 'data science', 'dsa': 'data structures and algorithms',
        'sqlserver': 'sql', 'db': 'sql'
    }
    tokens = [mapping.get(t, t) for t in tokens]
    return ', '.join(sorted(set(tokens)))

def skill_set(skill_str):
    """
    Return a normalized set of skill tokens from a comma-separated string.
    Cleans common non-skill words like 'fresher', 'junior', 'developer'.
    """
    if pd.isna(skill_str) or not str(skill_str).strip():
        return set()

    # Use normalize_skills mapping first (handles abbreviations)
    s = normalize_skills(skill_str)

    # split on commas, then clean each token
    raw_tokens = [t.strip().lower() for t in s.split(',') if t.strip()]
    stop_words = set([
        'fresher','junior','senior','developer','engineer','backend','frontend',
        'lead','sr','jr','intern','student','developer','role'
    ])

    clean_tokens = set()
    for tok in raw_tokens:
        # split multi-word tokens and remove stop words
        parts = [p for p in tok.split() if p not in stop_words]
        if not parts:
            continue
        # join remaining words (e.g. 'machine learning' stays)
        cleaned = " ".join(parts).strip()
        if cleaned:
            clean_tokens.add(cleaned)
    return clean_tokens

def build_corpus(jobs_df):
    jobs_df = jobs_df.copy()
    jobs_df['skills'] = jobs_df['skills'].apply(normalize_skills)
    # keep a skill_set column for fast overlap computation
    jobs_df['skill_set'] = jobs_df['skills'].apply(skill_set)

    jobs_df['text'] = (
        (jobs_df['title'].fillna('') + ' ') * 2 +
        (jobs_df['skills'].fillna('') + ' ') * 3 +
        jobs_df['description'].fillna('')
    )
    return jobs_df

def fit_tfidf(jobs_df, max_features=5000, ngram=(1,2)):
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words='english',
        ngram_range=ngram
    )
    job_texts = jobs_df['text'].tolist()
    job_tfidf = vectorizer.fit_transform(job_texts)
    return vectorizer, job_tfidf
