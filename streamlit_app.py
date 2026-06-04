import streamlit as st
import joblib
import pandas as pd
import sqlite3
import random
import string
from transformers import pipeline
import plotly.express as px

# ---------------- DATABASE ----------------
conn = sqlite3.connect("grievance.db", check_same_thread=False)
c = conn.cursor()

c.execute("""
CREATE TABLE IF NOT EXISTS complaints (
    id TEXT PRIMARY KEY,
    complaint TEXT,
    department TEXT,
    sentiment TEXT,
    confidence REAL,
    priority INTEGER,
    status TEXT
)
""")
conn.commit()

# ---------------- MODELS ----------------
@st.cache_resource
def load_models():
    model = joblib.load("department_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    classifier = pipeline("sentiment-analysis")
    return model, vectorizer, classifier

model, vectorizer, classifier = load_models()

# ---------------- PAGE ----------------
st.set_page_config(page_title="AI Grievance System", layout="wide")
st.title("🏛️ AI Citizen Grievance Dashboard")

# ---------------- HELPERS ----------------
def generate_id():
    return "C" + ''.join(random.choices(string.digits, k=6))

def urgency_score(sentiment):
    sentiment = sentiment.upper()
    if "NEGATIVE" in sentiment:
        return 5
    elif "POSITIVE" in sentiment:
        return 1
    return 3

def insert_complaint(data):
    c.execute("""
        INSERT INTO complaints VALUES (?, ?, ?, ?, ?, ?, ?)
    """, data)
    conn.commit()

def fetch_data():
    return pd.read_sql("SELECT * FROM complaints", conn)

# ---------------- INPUT ----------------
complaint = st.text_area("Enter Complaint")

if st.button("Analyze & Submit"):

    if not complaint.strip():
        st.warning("Please enter complaint")

    else:
        vec = vectorizer.transform([complaint])
        department = model.predict(vec)[0]

        sentiment_result = classifier(complaint)[0]
        sentiment = sentiment_result["label"]
        confidence = float(sentiment_result["score"])

        priority = urgency_score(sentiment)
        cid = generate_id()

        status = "Pending"

        insert_complaint((
            cid,
            complaint,
            department,
            sentiment,
            confidence,
            priority,
            status
        ))

        st.success(f"Complaint Submitted! ID: {cid}")

# ---------------- DASHBOARD ----------------
df = fetch_data()

if len(df) > 0:

    st.subheader("📊 Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Complaints", len(df))
    col2.metric("Pending", (df["status"] == "Pending").sum())
    col3.metric("Resolved", (df["status"] == "Resolved").sum())

    st.subheader("📋 Complaint Records")
    st.dataframe(df, use_container_width=True)

    # ---------------- CHARTS ----------------
    col1, col2 = st.columns(2)

    with col1:
        fig1 = px.bar(df["department"].value_counts(),
                      title="Complaints by Department")
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig2 = px.pie(df, names="sentiment", title="Sentiment Distribution")
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("⚡ Priority Distribution")
    fig3 = px.histogram(df, x="priority", nbins=5)
    st.plotly_chart(fig3, use_container_width=True)

    # ---------------- STATUS UPDATE (ADMIN STYLE) ----------------
    st.subheader("🛠️ Update Complaint Status")

    cid_update = st.selectbox("Select Complaint ID", df["id"].tolist())
    new_status = st.selectbox("Set Status", ["Pending", "In Progress", "Resolved"])

    if st.button("Update Status"):
        c.execute("""
            UPDATE complaints SET status = ? WHERE id = ?
        """, (new_status, cid_update))
        conn.commit()
        st.success("Status Updated!")