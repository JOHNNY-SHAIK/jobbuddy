# streamlit_app.py
import os
import sys
import time

import streamlit as st
import pandas as pd

# allow imports from src/
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from preprocess import skill_set
from resume_parser import extract_profile_from_pdf
from hybrid_recommender import HybridRecommender

st.set_page_config(
    page_title="JobBuddy — Job Recommendation Demo",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.title("JobBuddy — Job Recommendation Demo")
st.write("Enter a profile (title, skills, short bio) and get top job matches.")

# Sidebar controls
st.sidebar.header("Settings")
beta = st.sidebar.slider("Skill boost β (higher = skills dominate)", 0.0, 1.0, 0.5, 0.05)
alpha = st.sidebar.slider("Hybrid weight α (0 = TF-IDF only, 1 = SBERT only)", 0.0, 1.0, 0.0, 0.05)
top_n = st.sidebar.slider("Top N recommendations", 1, 10, 5)

# Load or initialize recommender (cached)
@st.cache_resource(show_spinner=False)
def load_recommender(alpha_value):
    # Create HybridRecommender with the given alpha
    return HybridRecommender(alpha=alpha_value)

with st.spinner("Loading recommender (this may take a few seconds)..."):
    rec = load_recommender(alpha)

# Example profiles from sample data
@st.cache_data
def load_sample_users():
    try:
        users = pd.read_csv(os.path.join("data", "users_sample.csv"))
    except Exception:
        users = pd.DataFrame([{
            "user_id": 1,
            "name": "Alice",
            "title": "Student - Data Science",
            "skills": "python, pandas, numpy, sql",
            "bio": "Final year student, built small ML projects and internships."
        }])
    return users

users = load_sample_users()

st.subheader("Try sample profiles")
cols = st.columns(min(len(users), 4))
for i, (_, u) in enumerate(users.iterrows()):
    with cols[i % len(cols)]:
        if st.button(f"Use: {u['name']}"):
            st.session_state["title"] = str(u.get("title", ""))
            st.session_state["skills"] = str(u.get("skills", ""))
            st.session_state["bio"] = str(u.get("bio", ""))

# Input form
st.subheader("Enter profile")

# Resume uploader (prefill fields if resume parsed)
uploaded_file = st.file_uploader("Upload resume (PDF) to auto-fill fields", type=["pdf"])
if uploaded_file is not None:
    tmp_path = os.path.join("data", "tmp_resume.pdf")
    with open(tmp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    profile = extract_profile_from_pdf(tmp_path)
    st.session_state['title'] = profile.get('title', "")
    st.session_state['skills'] = profile.get('skills', "")
    st.session_state['bio'] = profile.get('bio', "")
    st.success("Resume parsed and fields pre-filled. Edit fields if needed, then click Recommend jobs.")

# Text inputs (use session_state defaults when available)
title = st.text_input("Title / current role", value=st.session_state.get("title", ""))
skills = st.text_input("Skills (comma-separated)", value=st.session_state.get("skills", ""))
bio = st.text_area("Short bio (1-2 lines)", value=st.session_state.get("bio", ""))

# Recommend button: compute recommendations when clicked
if st.button("Recommend jobs"):
    user_text = " ".join([title, skills, bio]).strip()
    if not user_text:
        st.warning("Please enter some text (title / skills / bio).")
    else:
        with st.spinner("Generating recommendations..."):
            t0 = time.time()
            results = rec.recommend(user_text, top_n=top_n, beta=beta, user_skills_str=skills)
            time_taken = time.time() - t0

        # If recommender returned empty or invalid, handle gracefully
        if results is None or results.empty:
            st.error("No recommendations produced. Check your inputs or datasets.")
        else:
            # Compute matched_skills for display (results contains 'skill_set' column)
            user_skills_set = skill_set(skills) if skills else skill_set(user_text)
            matched = []
            for j_sk in results['skill_set']:
                try:
                    intersection = user_skills_set.intersection(j_sk)
                except Exception:
                    # If j_sk isn't a set (unexpected), attempt to parse
                    intersection = user_skills_set.intersection(skill_set(str(j_sk)))
                matched.append(", ".join(sorted(intersection)))
            results = results.copy()
            results['matched_skills'] = matched

            # Round score for nicer display
            results['score'] = results['score'].astype(float).round(4)

            st.success(f"Top {len(results)} recommendations (generated in {time_taken:.2f}s)")
            # show table with matched skills
            st.dataframe(results[['job_id', 'title', 'skills', 'matched_skills', 'description', 'score']].reset_index(drop=True))

            # show cards for the top 3
            st.markdown("### Top picks")
            for _, r in results.head(3).iterrows():
                st.markdown("---")
                st.markdown(f"#### {r['title']}  —  <small>score: {r['score']:.2f}</small>", unsafe_allow_html=True)
                st.markdown(f"**Skills:** {r['skills']}")
                st.markdown(f"**Matched skills:** {r['matched_skills']}")
                st.markdown(f"{r['description']}")

            # Notes & tips
            st.markdown("**Notes & tips:**")
            st.write("- Try adding more relevant skills (comma-separated).")
            st.write("- Increase α in sidebar to favor semantic matches (SBERT).")
            st.write("- To reproduce results, see `src/run_final.py` and `src/tune_alpha.py`.")
else:
    st.info("Fill the form and click **Recommend jobs** (or use a sample profile).")

st.markdown("---")
st.markdown("**About JobBuddy**: Content-based + semantic job recommender. Built with TF-IDF + SBERT and a hybrid ensemble. See the GitHub repo for code and evaluation scripts.")
