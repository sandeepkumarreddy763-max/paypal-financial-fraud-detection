
import pandas as pd

# 1. Load dataset
df = pd.read_csv("paypal_fraud_synthetic.csv")

print("Original dataset shape:", df.shape)

# 2. Convert timestamp into datetime
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

# 3. Create useful time-based features
df["transaction_hour"] = df["timestamp"].dt.hour
df["transaction_day"] = df["timestamp"].dt.dayofweek

# 4. Select input features
features = [
    "amount_usd",
    "payment_method",
    "login_country",
    "time_since_last_login_min",
    "device_type",
    "transaction_hour",
    "transaction_day"
]

# 5. Separate inputs (X) and target (y)
X = df[features].copy()
y = df["is_fraud"].copy()

# 6. Display input and target information
print("\nINPUT FEATURES (X):")
print(X.head())

print("\nTARGET (y):")
print(y.head())

print("\nFeature columns:")
print(X.columns.tolist())

print("\nTarget distribution:")
print(y.value_counts())

# 7. Check missing values
print("\nMissing values in X:")
print(X.isnull().sum())

print("\nMissing values in y:", y.isnull().sum())

print("\nPreprocessing preparation completed!")
print("X shape:", X.shape)
print("y shape:", y.shape)
