import pandas as pd
import pickle
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    confusion_matrix, 
    classification_report
)

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# ----------------------------------------------------------
# IMPORT PREPROCESS FUNCTION
# ----------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from src.preprocessing import preprocess_text


# ----------------------------------------------------------
# LOAD DATASET
# ----------------------------------------------------------
DATA_PATH = "../dataset/Combined Data.csv"

df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=["text"])
df["text"] = df["text"].astype(str)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

X = df["text"]
y = df["label"]

# ----------------------------------------------------------
# SPLIT DATA
# ----------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ----------------------------------------------------------
# MODELS TO TEST
# ----------------------------------------------------------
models = {
    "SVM (Linear Kernel)": SVC(kernel="linear", probability=True),
    "LinearSVC": LinearSVC(),
    "Logistic Regression": LogisticRegression(max_iter=500),
    "Multinomial NB": MultinomialNB(),
    "Random Forest": RandomForestClassifier(n_estimators=200),

    # ⭐ Optimized Gradient Boosting Parameters
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        min_samples_split=4,
        min_samples_leaf=2,
        subsample=0.9,
        max_features='sqrt'
    )
}


results = {}
best_model = None
best_score = 0
labels = sorted(df["label"].unique())


# ----------------------------------------------------------
# SAVE CONFUSION MATRIX AS PLOT
# ----------------------------------------------------------
def save_confusion_matrix(y_true, y_pred, model_name, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    filename = f"confusion_{model_name.replace(' ', '_').replace('(', '').replace(')', '')}.png"
    plt.savefig(filename)
    plt.close()
    print(f"Saved confusion matrix: {filename}")


# ----------------------------------------------------------
# TRAIN AND EVALUATE ALL MODELS
# ----------------------------------------------------------
print("\n########## MODEL BENCHMARK STARTED ##########\n")

for model_name, model in models.items():
    print(f"\n🔹 Training: {model_name}")

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=10000,
            stop_words="english",
            ngram_range=(1, 2),
            preprocessor=preprocess_text
        )),
        ("clf", model)
    ])

    # Train
    pipeline.fit(X_train, y_train)

    # Predict
    y_pred = pipeline.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    # Save confusion matrix
    save_confusion_matrix(y_test, y_pred, model_name, labels)

    print(f"Accuracy :  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    # Store results
    results[model_name] = {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "report": classification_report(y_test, y_pred, output_dict=True)
    }

    # Select best model (based on F1)
    if f1 > best_score:
        best_score = f1
        best_model = pipeline
        best_model_name = model_name


# ----------------------------------------------------------
# SAVE BEST MODEL
# ----------------------------------------------------------
with open("best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

print("\n########## BENCHMARK SUMMARY ##########\n")
for name, metrics in results.items():
    print(f"{name}: Acc={metrics['accuracy']:.4f}, "
          f"P={metrics['precision']:.4f}, R={metrics['recall']:.4f}, "
          f"F1={metrics['f1']:.4f}")

print(f"\n🏆 BEST MODEL: {best_model_name}")
print("Saved as: best_model.pkl\n")
