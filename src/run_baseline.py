# src/run_baseline.py
import pandas as pd
from recommender import build_recommender

def main():
    rec = build_recommender()
    users = pd.read_csv("../data/users_sample.csv")
    for _, user in users.iterrows():
        user_text = " ".join([str(user.get("title","")), str(user.get("skills","")), str(user.get("bio",""))])
        res = rec.recommend(user_text, top_n=5)
        print("User:", user['name'])
        print(res[['job_id','title','score']].to_string(index=False))
        print("-"*40)

if __name__ == "__main__":
    main()
