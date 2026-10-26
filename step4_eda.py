
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("paypal_fraud_synthetic.csv")

# Set graph style
sns.set_theme(style="whitegrid")

# 1. Basic dataset information
print("DATASET SHAPE:", df.shape)

print("\nFRAUD COUNTS:")
print(df["is_fraud"].value_counts())

print("\nFRAUD PERCENTAGE:")
print((df["is_fraud"].value_counts(normalize=True) * 100).round(2))

# 2. Fraud vs non-fraud transactions
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="is_fraud")
plt.title("Fraud vs Non-Fraud Transactions")
plt.xlabel("Is Fraud (0 = No, 1 = Yes)")
plt.ylabel("Number of Transactions")
plt.tight_layout()
plt.show()

# 3. Transaction amount distribution
plt.figure(figsize=(8, 4))
sns.histplot(data=df, x="amount_usd", bins=40)
plt.title("Transaction Amount Distribution")
plt.xlabel("Transaction Amount (USD)")
plt.ylabel("Number of Transactions")
plt.tight_layout()
plt.show()

# 4. Fraud by payment method
plt.figure(figsize=(8, 4))
sns.barplot(data=df, x="payment_method", y="is_fraud",
            errorbar=None)
plt.title("Fraud Rate by Payment Method")
plt.xlabel("Payment Method")
plt.ylabel("Fraud Rate")
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()

# 5. Fraud by device type
plt.figure(figsize=(7, 4))
sns.barplot(data=df, x="device_type", y="is_fraud",
            errorbar=None)
plt.title("Fraud Rate by Device Type")
plt.xlabel("Device Type")
plt.ylabel("Fraud Rate")
plt.tight_layout()
plt.show()

# 6. Fraud by login country
plt.figure(figsize=(8, 4))
sns.barplot(data=df, x="login_country", y="is_fraud",
            errorbar=None)
plt.title("Fraud Rate by Login Country")
plt.xlabel("Login Country")
plt.ylabel("Fraud Rate")
plt.tight_layout()
plt.show()

print("\nEDA completed successfully!")
