# preprocess.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def load_data(jobs_path='../data/jobs_sample.csv', users_path='../data/users_sample.csv'):
    jobs = pd.read_csv(jobs_path)
    users = pd.read_csv(users_path)
    return jobs, users

def normalize_skills(skill_str):
    if pd.isna(skill_str):
        return ''
    tokens = [s.strip().lower() for s in str(skill_str).split(',') if s.strip()]
    
    mapping = {
        'js': 'javascript',
        'py': 'python',
        'ml': 'machine learning',
        'nlp': 'natural language processing',
        'tf': 'tensorflow',
        'pytorch': 'pytorch',
        'pd': 'pandas',
        'reactjs': 'react',
    }
    
    tokens = [mapping.get(t, t) for t in tokens]
    return ', '.join(sorted(set(tokens)))

def build_corpus(jobs_df):
    jobs_df = jobs_df.copy()
    jobs_df['skills'] = jobs_df['skills'].apply(normalize_skills)

    # Weighted text (better matching)
    jobs_df['text'] = (
        (jobs_df['title'] + ' ') * 2 +
        (jobs_df['skills'] + ' ') * 3 +
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
