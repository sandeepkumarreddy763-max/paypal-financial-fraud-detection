
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
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    classification_report,
    ConfusionMatrixDisplay
)

# 1. Load dataset
df = pd.read_csv("paypal_fraud_synthetic.csv")

# 2. Prepare timestamp features
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df["transaction_hour"] = df["timestamp"].dt.hour
df["transaction_day"] = df["timestamp"].dt.dayofweek

# 3. Select features and target
features = [
    "amount_usd",
    "payment_method",
    "login_country",
    "time_since_last_login_min",
    "device_type",
    "transaction_hour",
    "transaction_day"
]

X = df[features]
y = df["is_fraud"]

# 4. Identify numerical and categorical columns
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

# 5. Preprocess numerical and categorical data
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# 6. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

# 7. Define two models
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

results = []
trained_models = {}

# 8. Train and evaluate each model
for name, model in models.items():

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    print("\n" + "=" * 50)
    print("MODEL:", name)

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        zero_division=0
    ))

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, zero_division=0
    )
    pr_auc = average_precision_score(y_test, probabilities)

    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1-score:", round(f1, 4))
    print("PR-AUC (Average Precision):", round(pr_auc, 4))

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "PR-AUC": pr_auc
    })

    trained_models[name] = pipeline

    # Plot confusion matrix
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        cmap="Blues",
        values_format="d"
    )
    plt.title(f"Confusion Matrix - {name}")
    plt.tight_layout()
    plt.show()

# 9. Compare model results
results_df = pd.DataFrame(results)

print("\nMODEL COMPARISON:")
print(results_df.round(4).to_string(index=False))

# 10. Save comparison results
results_df.to_csv("model_comparison.csv", index=False)

print("\nSaved model comparison to model_comparison.csv")
