
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    average_precision_score
)

# 1. Load dataset
df = pd.read_csv("paypal_fraud_synthetic.csv")

df["timestamp"] = pd.to_datetime(
    df["timestamp"], errors="coerce"
)
df["transaction_hour"] = df["timestamp"].dt.hour
df["transaction_day"] = df["timestamp"].dt.dayofweek

features = [
    "amount_usd",
    "payment_method",
    "login_country",
    "time_since_last_login_min",
    "device_type",
    "transaction_hour",
    "transaction_day"
]

numeric_features = [
    "amount_usd",
    "time_since_last_login_min",
    "transaction_hour",
    "transaction_day"
]

categorical_features = [
    "payment_method",
    "login_country",
    "device_type"
]

X = df[features]
y = df["is_fraud"]

# 2. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 3. Preprocessing
preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), numeric_features),

    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical_features)
])

# 4. Models
models = {
    "Logistic Regression": LogisticRegression(
        class_weight="balanced",
        max_iter=1000
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )
}

thresholds = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
results = []

# 5. Train and test each model
for name, model in models.items():

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    probabilities = pipeline.predict_proba(X_test)[:, 1]

    print("\n", "=" * 55)
    print("MODEL:", name)
    print(
        "PR-AUC (Average Precision):",
        round(average_precision_score(y_test, probabilities), 4)
    )

    for threshold in thresholds:

        predictions = (probabilities >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_test, predictions, labels=[0, 1]
        ).ravel()

        precision = precision_score(
            y_test, predictions, zero_division=0
        )
        recall = recall_score(
            y_test, predictions, zero_division=0
        )
        f1 = f1_score(
            y_test, predictions, zero_division=0
        )

        results.append({
            "Model": name,
            "Threshold": threshold,
            "Precision": precision,
            "Recall": recall,
            "F1-score": f1,
            "True_Positive": tp,
            "False_Positive": fp,
            "False_Negative": fn,
            "True_Negative": tn
        })

        print(
            f"Threshold={threshold:.2f} | "
            f"Precision={precision:.3f} | "
            f"Recall={recall:.3f} | "
            f"F1={f1:.3f} | "
            f"Detected fraud={tp} | Missed fraud={fn}"
        )

# 6. Save all threshold results
results_df = pd.DataFrame(results)
results_df.to_csv("threshold_comparison.csv", index=False)

print("\nSaved: threshold_comparison.csv")

# 7. Plot recall against threshold for both models
for name in models:
    subset = results_df[results_df["Model"] == name]

    plt.plot(
        subset["Threshold"],
        subset["Recall"],
        marker="o",
        label=name
    )

plt.xlabel("Fraud Probability Threshold")
plt.ylabel("Fraud Recall")
plt.title("Fraud Detection Recall at Different Thresholds")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("threshold_recall.png")
plt.show()

print("Saved: threshold_recall.png")
