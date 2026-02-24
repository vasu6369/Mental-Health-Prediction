import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
import sys
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from src.preprocessing import preprocess_text


# Load Dataset
DATA_PATH = "../dataset/Combined Data.csv"
df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=["text"])  # remove rows where text is NaN
df["text"] = df["text"].astype(str)  # ensure all text is string

# Shuffle dataset fully
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
print(df.head(10))
print(df['label'].value_counts())


# Required columns (text and label):
if "text" not in df.columns or "label" not in df.columns:
    raise ValueError("CSV must contain 'text' and 'label' columns")

X = df["text"]
y = df["label"]


# Train / Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=10000,
        stop_words="english",
        ngram_range=(1, 2),
        preprocessor=preprocess_text,
        sublinear_tf=True,
        min_df=2
    )),
    ("svm", SVC(kernel="linear", probability=True))
])

# Train

print("Model is being Trained...")
pipeline.fit(X_train, y_train)
print("Model Trained successfully !")

# Evaluate
y_pred = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy*100:.2f}%")

report = classification_report(y_test, y_pred, output_dict=True) 
report_df = pd.DataFrame(report).transpose() 
print("Classification Report:") 
print(report_df)

# Save file 
with open("model.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("Saved: model.pkl")
