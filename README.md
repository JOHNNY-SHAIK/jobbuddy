---

# 🧠 JobBuddy — AI-Powered Job Recommendation System

A hybrid **TF-IDF + SBERT + Skill-Matching** job recommendation engine with **resume parsing**, **profile extraction**, and **Streamlit-based UI**.
Designed to provide **precise, skill-aligned job recommendations**, especially for freshers and early-career candidates.

---

## 🚀 Features

### ✅ **Hybrid Recommendation Engine**

* **TF-IDF lexical matching**
* **SBERT semantic matching**
* **Skill-overlap boosting (β parameter)**
* Adjustable **α (TF-IDF ↔ SBERT)** for hybrid control
* Highly accurate matching for **Java**, **Data Science**, **Cloud**, **Testing**, **Android**, etc.

---

### 📄 **Resume Upload (PDF Parsing)**

Upload a resume (PDF) and JobBuddy automatically extracts:

* **Title / Role**
* **Skills**
* **Short Bio / Summary**

Powered by the internal `resume_parser.py` module.

---

### 🎯 **Smart Skill Detection**

* Converts messy skills into normalized forms
  Example:
  `js → javascript`, `ml → machine learning`, `py → python`
* Skill extraction works even when the user provides:

  * free text
  * resume text
  * comma-separated skills

---

### 📊 **Top-N Job Recommendations**

* Easily choose **Top 1–10** recommendations
* Each result includes:

  * Job title
  * Required skills
  * Matched skills
  * Description
  * Similarity score

---

### 🎨 **Modern Streamlit UI**

* Left sidebar for settings
* Clean, dark theme
* Resume upload + interactive job cards
* Sample profiles you can load instantly

---

## 🛠️ Project Structure

```
jobbuddy/
│── data/
│     ├── jobs_sample.csv
│     ├── users_sample.csv
│     └── tmp_resume.pdf
│
│── src/
│     ├── hybrid_recommender.py
│     ├── preprocess.py
│     ├── emb_recommender.py
│     ├── resume_parser.py
│     ├── add_java_jobs.py
│     └── cli_test.py
│
│── streamlit_app.py
│── requirements.txt
│── README.md
│── venv/
```

---

## ⚙️ Installation & Setup

### **1. Clone the repository**

```
git clone https://github.com/your-username/jobbuddy.git
cd jobbuddy
```

### **2. Create virtual environment**

```
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows
```

### **3. Install dependencies**

```
pip install -r requirements.txt
```

### **4. Run Streamlit app**

```
streamlit run streamlit_app.py
```

App starts at:
🔗 **[http://localhost:8501](http://localhost:8501)**

---

## 🧩 Technical Architecture

### 🔹 **1. Preprocessing**

`preprocess.py`

* Cleans text
* Normalizes skills
* Builds TF-IDF corpus

### 🔹 **2. Embeddings (SBERT)**

`emb_recommender.py`

* Loads `"all-MiniLM-L6-v2"`
* Encodes all job descriptions
* Stores embeddings

### 🔹 **3. Hybrid Logic**

`hybrid_recommender.py` combines:

```
final_score = α * semantic_similarity 
            + (1 - α) * tfidf_similarity
            + β * skill_overlap
```

### 🔹 **4. Resume Parser**

Extracts structured data from PDFs.

### 🔹 **5. Streamlit UI**

Interactive app with:

* Resume upload
* Auto-filled fields
* Job cards
* Settings (β, α, Top-N)

---

## 🧪 CLI Testing (Optional)

Run internal accuracy tests:

```
python src/cli_test.py
```

---

## 📈 Example Output

**Input**:
`Skills: Java, DSA`
`Bio: I know Java and DSA`

**Top Recommendations**:

| Job Title             | Score | Matched Skills |
| --------------------- | ----- | -------------- |
| Junior Java Developer | 1.00  | java, dsa      |
| Java Backend Engineer | 0.57  | java, sql      |
| Android Developer     | 0.44  | java           |

---

## 📝 Notes & Tips

* Increase **α** if you want *semantic* matching.
* Increase **β** if you want *skill-based* matching.
* Add more jobs using `add_java_jobs.py`.
* For best results, fill at least 2 fields (skills + bio).

---

## 📌 Roadmap

* 🔜 Real-time job scraping (LinkedIn / Naukri / Indeed)
* 🔜 Fine-tuned domain-specific SBERT
* 🔜 Skill gap analysis
* 🔜 Resume scoring & feedback

---

## 🤝 Contributing

Pull Requests are welcome.
Ensure code follows existing module structure.

---

## 📄 License

MIT License — Free for personal & academic use.

---

## 🧑‍💻 Author

**JOHNNY (JobBuddy Developer)**
AI • ML • NLP • Python • Streamlit

---

