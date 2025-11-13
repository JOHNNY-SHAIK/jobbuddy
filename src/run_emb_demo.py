# run_emb_demo.py
from emb_recommender import EmbeddingRecommender
import pandas as pd

def main():
    jobs, users = None, None  # emb class will load data itself
    emb_rec = EmbeddingRecommender()  # loads model and encodes jobs
    users = pd.read_csv("../data/users_sample.csv")  # run_emb_demo.py is in src/
    for _, user in users.iterrows():
        user_text = " ".join([str(user.get("title","")), str(user.get("skills","")), str(user.get("bio",""))])
        res = emb_rec.recommend(user_text, top_n=5)
        print("User:", user['name'])
        print(res[['job_id','title','score']].to_string(index=False))
        print("-"*40)

if __name__ == "__main__":
    main()
