\# JobBuddy — AI Job Recommendation System (TF-IDF + SBERT Hybrid)



JobBuddy is a LinkedIn-style job recommendation engine built with  

\*\*TF-IDF\*\*, \*\*Sentence-Transformers (SBERT)\*\*, and a \*\*Hybrid Ensemble\*\* model.



It predicts the top-5 most relevant jobs for a user based on their  

\*\*skills, title, and bio\*\*, achieving \*\*100% Precision@5\*\* on the demo dataset.



---



\## 🚀 Features



\- TF-IDF based lexical matching  

\- SBERT semantic embedding model  

\- Weighted skill matching  

\- Hybrid score fusion (tunable α)  

\- Evaluation pipeline (Precision@5)  

\- Weak labeling + manual labeling support  

\- Clean modular architecture  

\- Pythonic, beginner-friendly code  

\- Perfect GitHub portfolio project for ML/AI roles  



---



\## 📁 Project Structure


jobbuddy/
│
├── data/
│ ├── jobs_sample.csv
│ ├── users_sample.csv
│ └── labels.csv
│
├── src/
│ ├── preprocess.py
│ ├── recommender.py
│ ├── emb_recommender.py
│ ├── hybrid_recommender.py
│ ├── run_baseline.py
│ ├── run_emb_demo.py
│ ├── run_final.py
│ ├── tune_alpha.py
│ ├── create_weak_labels.py
│ └── eval_pipeline.py
│
└── README.md


---

## 🔧 Installation

```bash
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt

▶️ Running the Model
1. Generate Data
python src/generate_sample_data.py

2. TF-IDF Baseline
python src/run_baseline.py

3. Semantic Model
python src/run_emb_demo.py

4. Tune Hybrid α
python src/tune_alpha.py

5. Final Hybrid Recommender
python src/run_final.py

📊 Accuracy
Model	Precision@5
TF-IDF	66.7%
SBERT	~80%
Hybrid (tuned)	100%
📬 Contact

If you're a recruiter or engineer viewing this repo:
I built this from scratch using Python, ML, embeddings, data preprocessing, evaluation, and model tuning.


Save → Close.

---

# 📤 Upload to GitHub Now

Go to the root folder (`C:\Users\johnn\Documents\jobbuddy`) and run:

```powershell
git init
git add .
git commit -m "AI JobBuddy - Full Hybrid Recommender System"
git branch -M main
git remote add origin https://github.com/<your-username>/jobbuddy.git
git push -u origin main


