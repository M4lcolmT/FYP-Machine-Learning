import joblib
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from sentence_transformers import SentenceTransformer
import streamlit as st
import json
from collections import Counter
import matplotlib.pyplot as plt

# ------------------------------
# Load cached resources
# ------------------------------
model_path = "models/best_opt_svm_model_proba_class_weighted.pkl"

@st.cache_resource
def load_model_and_encoder():
    model = joblib.load(model_path)
    label_encoder = joblib.load("embeddings_labels/label_encoder.pkl")
    return model, label_encoder

# additional modification to allow numpy.load to handle pickled objects
@st.cache_resource
def load_fasttext():
    return KeyedVectors.load("models/fasttext_subword_300.kv", allow_pickle=True)

@st.cache_resource
def load_sentence_transformer():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_data
def load_skills_list():
    with open("data/skills_list.json", "r") as f:
        return sorted(json.load(f))

@st.cache_data
def load_job_skills_map():
    with open("data/job_skills_map.json", "r") as f:
        return json.load(f)

@st.cache_data
def load_job_role_descriptions():
    with open("data/job_role_descriptions.json", "r") as f:
        return json.load(f)

@st.cache_data
def load_top_skills_csv():
    return pd.read_csv("data/top_skills_per_title.csv")

# Load components
model, label_encoder = load_model_and_encoder()
fasttext_vectors = load_fasttext()
sentence_model = load_sentence_transformer()

available_skills = load_skills_list()
job_skills_map = load_job_skills_map()
job_role_descriptions = load_job_role_descriptions()
top_skills_df = load_top_skills_csv()

# ------------------------------
# Embedding user input
# ------------------------------
def embed_user_input(summary: str, skills_str: str, embedding_dim=300, summary_dim=384):
    skills = set(skill.strip().lower() for skill in skills_str.split(','))
    vectors = [fasttext_vectors[skill] for skill in skills if skill in fasttext_vectors]
    skill_embedding = np.mean(vectors, axis=0) if vectors else np.zeros(embedding_dim)
    summary_embedding = sentence_model.encode(summary)
    return np.hstack([skill_embedding, summary_embedding]).reshape(1, -1)

# ------------------------------
# Predict job
# ------------------------------
def predict_top_job(summary, skills):
    input_vector = embed_user_input(summary, skills)
    probs = model.predict_proba(input_vector)[0]
    top_index = np.argmax(probs)
    top_title = label_encoder.inverse_transform([top_index])[0]
    confidence = probs[top_index]
    return top_title, confidence

# ------------------------------
# Skill evaluation
# ------------------------------
def get_missing_skills(job_title, user_skills_list):
    required = job_skills_map.get(job_title, [])
    user_set = set(s.lower() for s in user_skills_list)
    return [s for s in required if s.lower() not in user_set]

def get_top_n_skills(job_title, n=15):
    df = top_skills_df[top_skills_df["Job Title"] == job_title]
    return df.sort_values("Frequency", ascending=False).head(n)["Skill"].tolist()

def get_top_skill_match_percentage(job_title, user_skills_list, top_n=15):
    top_skills = get_top_n_skills(job_title, top_n)
    user_set = set(s.lower() for s in user_skills_list)
    matched = [s for s in top_skills if s.lower() in user_set]
    return len(matched), len(top_skills)

def display_role_info(job_title, all_roles):
    match = next((r for r in all_roles if r["title"] == job_title), None)
    if match:
        st.markdown(f"### 📌 About: **{match['title']}**")
        st.info(match["description"])
        st.markdown("### 🧭 General Actionable Steps")
        for step in match["action_steps"]:
            st.markdown(f"✅ {step}")
    else:
        st.warning("No role info available.")

def plot_top_skills_bar(job_title, top_n):
    df = top_skills_df[top_skills_df["Job Title"] == job_title].sort_values("Frequency", ascending=False).head(top_n)
    if df.empty:
        st.warning("No skill data found for this role.")
        return
    fig, ax = plt.subplots(figsize=(10, top_n // 2 + 2))
    ax.barh(df["Skill"][::-1], df["Frequency"][::-1], color='skyblue')
    ax.set_xlabel("Frequency")
    ax.set_title(f"Top {top_n} Skills for '{job_title}'")
    st.pyplot(fig)

# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="Data Career Path Predictor", layout="centered")
st.title("🔮 Data Career Path Predictor")
st.markdown("""
Welcome! 🎓 This app helps you **identify your most suitable data career path** based on your skills and experience.

💼 Whether you're a student, job-seeker, or transitioning professional, we'll:
- Predict your best-fit job title in the data domain.
- Show how well your skills align with the role.
- Suggest missing skills and growth opportunities.
""")

# User description input
st.markdown("---")
st.markdown("## ✨ Step 1: Tell Us About Your Experience")
with st.container(border=True):
    user_summary = st.text_area(
        "🧠 **Your Background Summary**",
        height=180,
        placeholder="E.g. I've worked on data cleaning, EDA using Python and SQL, created dashboards in Tableau, and implemented machine learning models using scikit-learn.",
        key="user_summary",
    )

# Skill selection layout
st.markdown("---")
st.markdown("## 🧰 Step 2: Select Your Skills")
st.write("_You can select up to 15 skills to represent your strongest hard and soft skills._")

max_skills = 15

col1, col2 = st.columns([1, 4])
with col1:
    st.markdown("🔢 **Selected:**")
    st.metric(label="Skills Chosen", value=len(st.session_state.get("skill_multiselect", [])), delta=f"/ {max_skills}")

with col2:
    selected_skills = st.multiselect(
        "🛠️ **Your Skills**",
        options=available_skills,
        help="Choose your strongest technical and analytical skills.",
        max_selections=max_skills,
        # format_func=lambda x: x.capitalize(),
        key="skill_multiselect"
    )

user_skills = ", ".join(selected_skills)

st.markdown("---")
st.markdown("##### ✅ Ready? Let's see your best-fit data role:")

# --- Predict Button ---
if st.button("🚀 Predict Best Role"):
    if not user_summary.strip():
        st.error("Please enter your job summary above.")
    elif not selected_skills:
        st.error("Please select at least 1 skill.")
    else:
        with st.spinner("🔎 Analyzing your profile..."):
            job_title, confidence = predict_top_job(user_summary, user_skills)
            missing_skills = get_missing_skills(job_title, selected_skills)
            top_n_matched, total_top_n = get_top_skill_match_percentage(job_title, selected_skills, top_n=15)

        # Save results to session_state
        st.session_state.predicted = True
        st.session_state.job_title = job_title
        st.session_state.confidence = confidence
        st.session_state.missing_skills = missing_skills
        st.session_state.top_n_matched = top_n_matched
        st.session_state.total_top_n = total_top_n

# --- Display Results if Available ---
if st.session_state.get("predicted"):
    job_title = st.session_state.job_title
    confidence = st.session_state.confidence
    missing_skills = st.session_state.missing_skills
    top_n_matched = st.session_state.top_n_matched
    total_top_n = st.session_state.total_top_n

    st.success(f"🎯 **Predicted Role:** `{job_title}`")
    st.markdown(f"📊 **Confidence Score:** `{confidence:.2f}`")

    # Skill breakdown
    st.markdown("---")
    st.subheader("📋 Skills Evaluation")

    st.markdown("##### 🔎 Top 15 Skills Match")
    st.info(f"✅ You’re aligned with {top_n_matched}/{total_top_n} of the most critical skills for the {job_title} role.")

    if top_n_matched >= 13:
        st.success(f"🎉 Excellent! You possess most of the core skills for the `{job_title}` role.")
    else:
        st.warning(f"⚠️ Here are some additional skills from the top 50 required for the {job_title} role that you might not currently have.")
        if missing_skills:
            st.markdown("🔻 " + ", ".join(f"`{s}`" for s in missing_skills))

    # Top skill bar chart
    st.markdown("---")
    st.subheader("📊 Top Skills for This Role")
    top_n_choice = st.selectbox("🔢 Number of skills to display:", [10, 20, 30, 40, 50], index=0)
    plot_top_skills_bar(job_title, top_n=top_n_choice)

    st.markdown("---")
    display_role_info(job_title, job_role_descriptions)

    if job_title == "Others":
        with st.expander("🔎 Explore Other Potential Roles"):
            alt_roles = ["Business Intelligence Analyst", "Business Analyst", "Statistician"]
            alt = st.selectbox("Choose a role to explore:", alt_roles)
            display_role_info(alt, job_role_descriptions)

# Footer
st.markdown("---")
st.caption("🧠 Built by Wei Han for FYP project")
