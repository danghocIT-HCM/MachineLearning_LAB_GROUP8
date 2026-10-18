import re

import joblib
import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


# CẤU HÌNH
DATA_PATH = "emails.csv"
TEXT_COL = "Message"
LABEL_COL = "Category"
MODEL_PATH = "spam_model.joblib"
RANDOM_STATE = 42


# DATA PREPROCESSING
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_data(path: str):
    df = pd.read_csv(path, on_bad_lines='skip')
    df = df[[TEXT_COL, LABEL_COL]].dropna().drop_duplicates()
    df[TEXT_COL] = df[TEXT_COL].apply(clean_text)

    if pd.api.types.is_string_dtype(df[LABEL_COL]) or df[LABEL_COL].dtype == object:
        df[LABEL_COL] = df[LABEL_COL].str.strip().str.lower().map(
            {"ham": 0, "not spam": 0, "spam": 1}
        )
    return df[TEXT_COL], df[LABEL_COL].astype(int)


def make_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(stop_words="english", ngram_range=(1, 2),
                           min_df=2, max_features=50000)


# MODEL TRAINING
def build_models() -> dict:
    return {
        "Logistic Regression": Pipeline([
            ("tfidf", make_vectorizer()),
            ("clf", LogisticRegression(max_iter=1000, C=1.0)),
        ]),
        "SVM (LinearSVC)": Pipeline([
            ("tfidf", make_vectorizer()),
            ("clf", LinearSVC(C=1.0)),
        ]),
        "Naive Bayes": Pipeline([
            ("tfidf", make_vectorizer()),
            ("clf", MultinomialNB(alpha=1.0)),
        ]),
    }


# MODEL EVALUATION
def evaluate(name, model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    result = {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }
    print(f"\n=== {name} ===")
    print(classification_report(y_test, y_pred, target_names=["not spam", "spam"]))
    return result


# HYPERPARAMETER TUNING
def tune_model(pipeline, param_grid, X_train, y_train):
    grid = GridSearchCV(pipeline, param_grid, cv=5, scoring="f1", n_jobs=-1)
    grid.fit(X_train, y_train)
    print("Tham số tốt nhất:", grid.best_params_, "| F1 (CV):", round(grid.best_score_, 4))
    return grid.best_estimator_


# ENSEMBLE
def build_ensemble() -> Pipeline:
    voting = VotingClassifier(
        estimators=[
            ("lr", LogisticRegression(max_iter=1000)),
            ("svm", LinearSVC()),
            ("nb", MultinomialNB()),
        ],
        voting="hard",
    )
    return Pipeline([("tfidf", make_vectorizer()), ("clf", voting)])


# MODEL DEPLOYMENT
def predict_email(model, email_text: str) -> str:
    pred = model.predict([clean_text(email_text)])[0]
    return "SPAM" if pred == 1 else "NOT SPAM"


# MAIN
def main():
    X, y = load_data(DATA_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    models = build_models()
    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        results.append(evaluate(name, model, X_test, y_test))

    ensemble = build_ensemble().fit(X_train, y_train)
    results.append(evaluate("Ensemble (Voting)", ensemble, X_test, y_test))

    summary = pd.DataFrame(results).set_index("model").round(4)
    print("\n=== BẢNG SO SÁNH ===")
    print(summary)

    best_lr = tune_model(
        models["Logistic Regression"],
        {"clf__C": [0.1, 1, 10], "tfidf__ngram_range": [(1, 1), (1, 2)]},
        X_train, y_train,
    )
    evaluate("Logistic Regression (đã tuning)", best_lr, X_test, y_test)

    best_name = summary["f1"].idxmax()
    best_model = {**models, "Ensemble (Voting)": ensemble}[best_name]
    joblib.dump(best_model, MODEL_PATH)
    print(f"\nĐã lưu mô hình tốt nhất: {best_name} -> {MODEL_PATH}")

    loaded = joblib.load(MODEL_PATH)
    sample = "Congratulations! You won $1000 cash prize. Click here to claim FREE now!!!"
    print("Email mẫu:", predict_email(loaded, sample))


if __name__ == "__main__":
    main()