
import pandas as pd

# Load the results from Step 8
df = pd.read_csv("threshold_comparison.csv")

# Show the best F1 result for each model
best_f1 = (
    df.sort_values("F1-score", ascending=False)
      .groupby("Model")
      .head(1)
      .sort_values("F1-score", ascending=False)
)

print("\nBEST F1 RESULT FOR EACH MODEL")
print(
    best_f1[
        ["Model", "Threshold", "Precision", "Recall",
         "F1-score", "True_Positive", "False_Positive",
         "False_Negative"]
    ].round(4).to_string(index=False)
)

# Show high-recall Logistic Regression options
lr = df[df["Model"] == "Logistic Regression"]
high_recall = lr[lr["Recall"] >= 0.90]

print("\nLOGISTIC REGRESSION: THRESHOLDS WITH RECALL >= 90%")
print(
    high_recall[
        ["Threshold", "Precision", "Recall", "F1-score",
         "True_Positive", "False_Positive", "False_Negative"]
    ].round(4).to_string(index=False)
)

# Save the report
best_f1.to_csv("best_model_results.csv", index=False)

print("\nSaved: best_model_results.csv")
