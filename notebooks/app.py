from fastapi import FastAPI
import joblib
from transformers import pipeline

app = FastAPI()

# Load Department Prediction Model
model = joblib.load("notebooks/department_model.pkl")
vectorizer = joblib.load("notebooks/tfidf_vectorizer.pkl")

# Load BERT Sentiment Model
classifier = pipeline("sentiment-analysis")

# Priority Score Function
def urgency_score(sentiment):
    
    if sentiment == "NEGATIVE":
        return 5
    
    elif sentiment == "POSITIVE":
        return 1
    
    else:
        return 3

# Home Page
@app.get("/")
def home():
    
    return {
        "message": "AI Citizen Grievance System Running"
    }

# Prediction API
@app.get("/predict")
def predict(complaint: str):
    
    # Department Prediction
    complaint_vector = vectorizer.transform([complaint])
    
    department = model.predict(complaint_vector)[0]

    # Sentiment Prediction using BERT
    sentiment_result = classifier(complaint)

    sentiment = sentiment_result[0]['label']

    # Priority Score
    priority = urgency_score(sentiment)

    # Final Output
    return {

        "Complaint": complaint,

        "Predicted Department": str(department),

        "Sentiment": sentiment,

        "Priority Score": priority

    }