# run_final.py
from hybrid_recommender import HybridRecommender
import pandas as pd

def main():
    # Best alpha from tuning
    rec = HybridRecommender(alpha=0.0)

    users = pd.read_csv("../data/users_sample.csv")

    for _, user in users.iterrows():
        user_text = " ".join([
            str(user.get("title", "")),
            str(user.get("skills", "")),
            str(user.get("bio", ""))
        ])

        res = rec.recommend(user_text, top_n=5)
        
        print("\nUser:", user['name'])
        print(res[['job_id', 'title', 'score']].to_string(index=False))
        print("-" * 40)

if __name__ == "__main__":
    main()
