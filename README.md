# 🔮 Data Career Path Predictor

An AI-powered app that predicts your most suitable **data career** based on your **skills** and **experience**.  
🎓 Final Year Project by **Wei Han**


## 🧩 Overview

**Data Career Path Predictor** helps users find which **data-related job role** best fits their background — such as **Data Analyst**, **Data Scientist**, or **Machine Learning Engineer**.

You’ll get:
- 🎯 A predicted career role  
- 📊 Confidence score  
- 💡 Skill match percentage  
- 🧠 Missing skills & suggestions  


## 🚀 Features

- **Career Prediction** – Trained **XGBoost** model suggests your best-fit data role  
- **Smart Embeddings** – Uses **FastText** (skills) + **SentenceTransformer** (summary)  
- **Skill Gap Analysis** – Compares your skills with top industry skills  
- **Visual Insights** – Displays confidence bars and top skills charts  
- **Career Guidance** – Provides job descriptions & learning steps  
- **Fast Loading** – Optimized with Streamlit caching  


## 🛠️ Tech Stack

| Layer | Tools |
|-------|-------|
| Frontend | Streamlit |
| ML / Backend | XGBoost, NumPy, Pandas, Joblib |
| NLP | FastText, SentenceTransformer |
| Visualization | Matplotlib |
| Data | JSON & CSV (skills, job mappings) |
| Caching | Streamlit `@st.cache_resource`, `@st.cache_data` |


## ⚙️ How It Works

1. **Input** your background summary & up to 10 skills  
2. **Vectorize** inputs using FastText + SentenceTransformer  
3. **Predict** best-fit data role using XGBoost  
4. **Evaluate** skill match %, missing skills, and confidence  
5. **Visualize** everything in an interactive Streamlit dashboard  


## 🧠 Example

**Input:**  
> “I’ve worked on data cleaning, EDA with Python and SQL, and created dashboards in Tableau.”

**Skills:**  
> Python, SQL, Tableau, Data Visualization, Machine Learning  

**Output:**  
> 🎯 **Predicted Role:** Data Analyst  
> 📊 **Confidence:** 0.89  
> 💡 **Skill Match:** 8/10 → 80% ✅


![FYP demo](https://github.com/M4lcolmT/FYP-Machine-Learning/blob/main/FYP_demo.gif?raw=true)
