import gradio as gr
import requests
from typing import Dict

# Gradio fn: Calls FastAPI (or direct model for simplicity—here, direct to keep standalone)
def predict_review(text: str) -> Dict[str, str]:
  if not text.strip():
    return {"error": "Enter a review!"}
  
  # Reuse same logic as API (copy for demo; in prod, call API)
  from src.preprocess.clean_text import clean_text
  import joblib
  import pandas as pd
  
  model = joblib.load('data/models/logreg_tuned.pkl')
  vectorizer = joblib.load('data/models/vectorizer.pkl')
  
  cleaned = clean_text(text)
  if not cleaned:
    return {"error": "Too short!"}
  
  X = vectorizer.transform([cleaned])
  X_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
  proba = model.predict_proba(X_df)[:, 1][0]
  pred = "Positive" if proba > 0.5 else "Negative"
  
  return pred, float(f"{proba:.3f}"), cleaned[:200] + "..." if len(cleaned) > 200 else cleaned


# Launch UI
demo = gr.Interface(
  fn=predict_review,
  inputs=gr.Textbox(label="Movie Review", placeholder="Enter a review..."),
  outputs=[
      gr.Textbox(label="Prediction"),
      gr.Slider(minimum=0, maximum=1, label="Confidence", interactive=False),
      gr.Textbox(label="Processed Text")
  ],
  title="Sentiment Analysis Demo",
  description="Paste a movie review and get instant sentiment!"
)

if __name__ == "__main__":
  demo.launch(share=True)  # Public link for 72h; remove for local-only