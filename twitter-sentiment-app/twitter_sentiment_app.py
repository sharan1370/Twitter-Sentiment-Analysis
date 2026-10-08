import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from scipy.sparse import hstack
import joblib
import os

# ===================================
#  CONFIGURATION
# ===================================
st.set_page_config(
    page_title="Twitter Sentiment Analyzer",
    layout="wide",
    page_icon="💬"
)

# ===================================
# LOAD MODELS
# ===================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "model")

if not os.path.exists(MODEL_DIR):
    st.error("⚠️ Model folder not found. Please make sure the model folder exists.")
    st.stop()

tfidf = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
lr_model = joblib.load(os.path.join(MODEL_DIR, "logistic_model.pkl"))
le = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))

# ===================================
#  CUSTOM STYLE
# ===================================
st.markdown("""
    <style>
        /* Global background and text */
        body {
            background-color: #f8fafc;
            color: #222;
            font-family: 'Segoe UI', sans-serif;
        }
        /* Header */
        .main-header {
            background: linear-gradient(90deg, #1DA1F2, #007acc);
            padding: 1.5rem;
            border-radius: 12px;
            color: white;
            text-align: center;
            margin-bottom: 2rem;
        }
        .main-header h1 {
            font-size: 2.2rem;
            margin: 0;
        }
        .main-header p {
            font-size: 1rem;
            opacity: 0.9;
        }
        /* Text area */
        .stTextArea textarea {
            border-radius: 10px;
            border: 2px solid #1DA1F2;
            font-size: 1rem;
            padding: 10px;
        }
        /* Buttons */
        .stButton button {
            background-color: #1DA1F2;
            color: white;
            font-weight: 600;
            border-radius: 10px;
            padding: 0.6rem 1.5rem;
            border: none;
            transition: 0.3s;
        }
        .stButton button:hover {
            background-color: #007acc;
            transform: scale(1.03);
        }
        /* Result cards */
        .result-card {
            background-color: white;
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.08);
            margin-bottom: 1.5rem;
        }
    </style>
""", unsafe_allow_html=True)

# ===================================
#  HEADER
# ===================================
st.markdown("""
<div class='main-header'>
    <h1>💬 Twitter Sentiment Analyzer</h1>
    <p>Analyze tweets and comments using Machine Learning (Logistic Regression trained on 1.6M tweets)</p>
</div>
""", unsafe_allow_html=True)

# ===================================
#  SIDEBAR
# ===================================
st.sidebar.title("📘 Instructions")
st.sidebar.markdown("""
1. Type one or more tweets — one per line  
2. Click **Analyze Sentiment**  
3. View results and sentiment distribution chart  
---
**Legend:**  
🟢 Positive  
🔴 Negative  
""")
st.sidebar.info("Developed with ❤️ using Python, Scikit-learn & Streamlit")

# ===================================
#  MAIN INPUT AREA
# ===================================
st.markdown("<div class='result-card'>", unsafe_allow_html=True)
user_input = st.text_area(
    "✍️ Enter Tweets or Comments (one per line):",
    height=180,
    placeholder="Example:\nI love this app!\nThis is terrible!\nBest experience ever!"
)
st.markdown("</div>", unsafe_allow_html=True)

# ===================================
#  PREDICTION LOGIC
# ===================================
if st.button("🔍 Analyze Sentiment"):
    if not user_input.strip():
        st.warning("⚠️ Please enter at least one tweet or comment.")
    else:
        tweets = [t.strip() for t in user_input.split("\n") if t.strip()]
        results = []

        for tweet in tweets:
            exclamation_count = tweet.count('!')
            question_count = tweet.count('?')
            hashtag_count = len(re.findall(r'#\w+', tweet))
            seq = tfidf.transform([tweet])
            X_input = hstack([seq, np.array([[exclamation_count, question_count, hashtag_count]])])
            pred_class = lr_model.predict(X_input)[0]
            pred_prob = lr_model.predict_proba(X_input)[0][pred_class]
            label = le.inverse_transform([pred_class])[0]
            results.append({
                'Tweet': tweet,
                'Predicted Sentiment': label,
                'Confidence': round(float(pred_prob), 4)
            })

        df_results = pd.DataFrame(results)

        # Results Table
        st.markdown("<div class='result-card'>", unsafe_allow_html=True)
        st.markdown("### 🧾 Prediction Results")
        st.dataframe(df_results.style.background_gradient(subset=['Confidence'], cmap='RdYlGn'))
        st.markdown("</div>", unsafe_allow_html=True)

        # Chart Section
        st.markdown("<div class='result-card'>", unsafe_allow_html=True)
        st.markdown("### 📊 Sentiment Distribution")

        summary = df_results['Predicted Sentiment'].value_counts()
        fig, ax = plt.subplots(figsize=(6, 4))
        summary.plot(kind='bar', color=['#ff4d4d', '#00cc66'], ax=ax)
        ax.set_title("Sentiment Summary", fontsize=13, fontweight='bold')
        ax.set_ylabel("Number of Tweets")
        ax.set_xlabel("")
        st.pyplot(fig)
        st.markdown("</div>", unsafe_allow_html=True)

else:
    st.info("💡 Enter tweets above and click **Analyze Sentiment** to see the results.")


