import os
import pickle
from src.preprocessing import preprocess_text

# Load model and vectorizer once at startup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "model.pkl")

LABEL_MAPPING = {
    "Anxiety": "Anxiety",
    "Normal": "Normal",
    "Depression": "Depression",
    "Stress": "Stress",
    "Personality disorder": "Personality disorder",
    "Bipolar":"Bipolar",
    "Suicidal":"Suicidal"
}
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

def predict_depression(text: str) -> dict:
    pred_label = model.predict([text])[0]
    confidence = max(model.predict_proba([text])[0])
    print(text)
    return {
        "label": pred_label,
        "meaning": LABEL_MAPPING[pred_label],
        "confidence": float(confidence)
    }
