
import pandas as pd
import joblib

# Load the saved model and threshold
model = joblib.load("final_fraud_model.joblib")

with open("fraud_threshold.txt", "r") as file:
    threshold = float(file.read().strip())

print("PAYPAL TRANSACTION FRAUD DETECTION")
print("-" * 40)

# Enter transaction details
amount = float(input("Enter transaction amount (USD): "))
payment_method = input(
    "Payment method (Credit_Card, Debit_Card, Linked_Bank, PayPal_Balance): "
)
country = input("Enter login country (e.g., IN, US, GB): ")
login_time = float(
    input("Minutes since last login: ")
)
device = input(
    "Device type (Desktop_Web, Mobile_App, Mobile_Web): "
)
hour = int(input("Transaction hour (0-23): "))
day = int(input("Day of week (Monday=0, Sunday=6): "))

# Prepare transaction data
transaction = pd.DataFrame([{
    "amount_usd": amount,
    "payment_method": payment_method,
    "login_country": country,
    "time_since_last_login_min": login_time,
    "device_type": device,
    "transaction_hour": hour,
    "transaction_day": day
}])

# Predict probability
probability = model.predict_proba(transaction)[0][1]

# Classify using the selected threshold
prediction = int(probability >= threshold)

print("\n" + "=" * 40)
print(f"Estimated fraud probability: {probability:.2%}")
print(f"Threshold: {threshold:.2f}")

if prediction == 1:
    print("RESULT: FLAG FOR FRAUD REVIEW")
else:
    print("RESULT: LOWER MODEL-ESTIMATED FRAUD RISK")

print("=" * 40)
