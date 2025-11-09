from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import joblib
import pandas as pd
import uvicorn
from src.preprocess.pipeline import SentimentPipeline, clean_text
from src.models.train_baseline import train_baseline  

class ReviewRequest(BaseModel):
  text: str

class ReviewResponse(BaseModel):
  prediction: str  # "Positive" or "Negative"
  confidence: float  # Prob of positive
  processed_text: str  # For debug

model = joblib.load('data/models/logreg_tuned.pkl')
vectorizer = joblib.load('data/models/vectorizer.pkl')

app = FastAPI(title="Sentiment Analysis MVP", version="1.0.0")

@app.get("/")
def root():
    return {"message": "Sentiment API ready! POST to /predict with {'text': 'your review'}."}

@app.post("/predict", response_model=ReviewResponse)
async def predict_sentiment(request: ReviewRequest) -> ReviewResponse:
  if not request.text.strip():
    raise HTTPException(status_code=400, detail="Text cannot be empty")
  
  # Preprocess: Clean + vectorize
  cleaned = clean_text(request.text)
  if not cleaned:
    raise HTTPException(status_code=400, detail="Text too short after cleaning")
  
  # Vectorize (use fitted vectorizer)
  X = vectorizer.transform([cleaned])
  X_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

  # Predict
  proba = model.predict_proba(X_df)[:, 1][0]  # Pos prob
  pred = 1 if proba > 0.5 else 0
  label = "Positive" if pred == 1 else "Negative"
  
  return ReviewResponse(
      prediction=label,
      confidence=round(proba, 3),
      processed_text=cleaned
  )

@app.get("/health")
def health():
  return {"status": "healthy", "model_loaded": True}

if __name__ == "__main__":  
  uvicorn.run(app, host="0.0.0.0", port=8000)   