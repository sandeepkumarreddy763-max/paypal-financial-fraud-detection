
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# Load dataset
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

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

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

model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(
        class_weight="balanced",
        max_iter=1000
    ))
])

model.fit(X_train, y_train)

# Save the complete trained pipeline
joblib.dump(model, "final_fraud_model.joblib")

# Save the selected probability threshold
with open("fraud_threshold.txt", "w") as file:
    file.write("0.20")

print("Final model saved successfully!")
print("Model: Logistic Regression")
print("Threshold: 0.20")
print("File: final_fraud_model.joblib")
