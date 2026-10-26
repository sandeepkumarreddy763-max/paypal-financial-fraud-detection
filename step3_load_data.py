
import pandas as pd

# Load the transaction dataset
df = pd.read_csv("paypal_fraud_synthetic.csv")

# Display the first five rows
print("FIRST FIVE TRANSACTIONS:")
print(df.head())

# Display dataset dimensions
print("\nDATASET SHAPE:")
print(df.shape)

# Display column names
print("\nCOLUMN NAMES:")
print(df.columns.tolist())

# Display dataset information
print("\nDATASET INFORMATION:")
df.info()

# Check missing values
print("\nMISSING VALUES:")
print(df.isnull().sum())

# Count fraud and non-fraud transactions
print("\nFRAUD COUNTS:")
print(df["is_fraud"].value_counts())
