
import pandas as pd

df = pd.read_csv("credit_card_fraud_dataset.csv")

print("=== FRAUD PATTERN ANALYSIS ===")

print("\n7. Fraud rate by transaction hour:")

df["TransactionDate"] = pd.to_datetime(df["TransactionDate"])
df["Hour"] = df["TransactionDate"].dt.hour

hourly_fraud = df.groupby("Hour")["IsFraud"].agg(
    ["count", "sum", "mean"]
)

hourly_fraud["FraudRate%"] = hourly_fraud["mean"] * 100

print(hourly_fraud[["count", "sum", "FraudRate%"]].round(2))
print("\n8. Transaction date-time sample:")
print(df["TransactionDate"].head(10).to_list())

print("\n9. Unique transaction timestamps:")
print(df["TransactionDate"].nunique())

print("\n10. Transactions per timestamp:")
print(df["TransactionDate"].value_counts().head(10))
print("\n11. Fraud rate by merchant:")

merchant_stats = df.groupby("MerchantID")["IsFraud"].agg(
    Transactions="count",
    FraudCases="sum",
    FraudRate="mean"
)

merchant_stats["FraudRate%"] = merchant_stats["FraudRate"] * 100

print(
    merchant_stats
    .sort_values("Transactions", ascending=False)
    .head(10)[["Transactions", "FraudCases", "FraudRate%"]]
    .round(2)
)
print("\n12. Merchant ID data type:")
print(df["MerchantID"].dtype)

print("\n13. Number of unique merchants:")
print(df["MerchantID"].nunique())

print("\n14. Fraud cases by merchant:")
print(
    df[df["IsFraud"] == 1]["MerchantID"]
    .value_counts()
    .head(10)
)
