
import os
import joblib
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
EVIDENCE_DIR = os.path.join(BASE_DIR, "evidence")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(EVIDENCE_DIR, exist_ok=True)

RENAME_MAP = {
    "Installations": "Installs & Demo",
    "Consumables": "Filters & Consumables"
}

def prepare_text(df):
    columns = [
        "request_text",
        "product_family",
        "warranty_status",
        "channel",
        "source"
    ]

    result = []

    for _, row in df.iterrows():
        parts = []

        for col in columns:
            value = str(row.get(col, ""))

            if value.lower() != "nan":
                parts.append(value.lower())

        result.append(" ".join(parts))

    return result


def main():

    print("Loading datasets...")

    train = pd.read_csv(
        os.path.join(DATA_DIR, "train.csv")
    )

    test = pd.read_csv(
        os.path.join(DATA_DIR, "test_unlabelled.csv")
    )

    train["team_label"] = train["team_label"].replace(
        RENAME_MAP
    )

    print("Training records:", len(train))
    print("Test records:", len(test))

    train["created_at_ist"] = pd.to_datetime(
    train["created_at_ist"],
    errors="coerce"
)
    date_values = train["created_at_ist"].to_numpy()

    order = np.argsort(date_values, kind="stable")
    train = train.iloc[order].reset_index(drop=True)

# Chronological validation split
    split = int(len(train) * 0.80)

    train_part = train.iloc[:split].copy()
    valid_part = train.iloc[split:].copy()

    print("Chronological split completed.", flush=True)
    

    X_train_text = prepare_text(train_part)
    X_valid_text = prepare_text(valid_part)

    y_train = train_part["team_label"]
    y_valid = valid_part["team_label"]

    print("Building TF-IDF features...")

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=120000,
        sublinear_tf=True,
        strip_accents="unicode"
    )

    X_train = vectorizer.fit_transform(X_train_text)
    X_valid = vectorizer.transform(X_valid_text)

    print("Training Linear SVM...")

    model = LinearSVC(C=1.5)

    model.fit(X_train, y_train)

    predictions = model.predict(X_valid)

    accuracy = accuracy_score(y_valid, predictions)

    macro_f1 = f1_score(
        y_valid,
        predictions,
        average="macro"
    )

    print("\nVALIDATION RESULTS")

    print("Accuracy:", accuracy)
    print("Macro F1:", macro_f1)

    report = classification_report(
        y_valid,
        predictions,
        zero_division=0
    )

    print(report)

    with open(
        os.path.join(EVIDENCE_DIR, "evaluation_report.txt"),
        "w"
    ) as f:
        f.write(report)
        f.write(f"\nAccuracy: {accuracy:.5f}")
        f.write(f"\nMacro F1: {macro_f1:.5f}")

    # Confusion matrix
    import matplotlib.pyplot as plt
    import seaborn as sns

    labels = sorted(y_train.unique())

    cm = confusion_matrix(
        y_valid,
        predictions,
        labels=labels
    )

    plt.figure(figsize=(12, 9))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=labels,
        yticklabels=labels
    )

    plt.xlabel("Predicted Team")
    plt.ylabel("Actual Team")
    plt.title("Service Routing Confusion Matrix")

    plt.tight_layout()

    plt.savefig(
        os.path.join(EVIDENCE_DIR, "confusion_matrix.png")
    )

    plt.close()

    # Retrain on all historical data
    print("\nTraining final model on all records...")

    final_vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=120000,
        sublinear_tf=True,
        strip_accents="unicode"
    )

    X_all = final_vectorizer.fit_transform(
        prepare_text(train)
    )

    final_model = LinearSVC(C=1.5)

    final_model.fit(
        X_all,
        train["team_label"]
    )

    # Save model and vectorizer
    joblib.dump(
        {
            "model": final_model,
            "vectorizer": final_vectorizer,
            "team_names": sorted(train["team_label"].unique())
        },
        os.path.join(MODEL_DIR, "routing_model.pkl")
    )

    # Predict test records
    X_test = final_vectorizer.transform(
        prepare_text(test)
    )

    test_predictions = final_model.predict(X_test)

    output = pd.DataFrame({
        "request_id": test["request_id"],
        "team": test_predictions
    })

    sample = pd.read_csv(
        os.path.join(DATA_DIR, "sample_submission.csv")
    )

    assert output["request_id"].tolist() == sample["request_id"].tolist()

    assert output["team"].notna().all()

    assert set(output["team"]).issubset(
        set(train["team_label"].unique())
    )

    output.to_csv(
        os.path.join(BASE_DIR, "predictions.csv"),
        index=False
    )

    print("\nSUCCESS")
    print("Model saved.")
    print("Predictions saved.")
    print("Total predictions:", len(output))


if __name__ == "__main__":
    main()
