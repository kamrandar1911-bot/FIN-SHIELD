
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

df = pd.read_csv("credit_card_fraud_dataset.csv")

date = pd.to_datetime(df["TransactionDate"])
df["Hour"] = date.dt.hour
df["DayOfWeek"] = date.dt.dayofweek
df["Month"] = date.dt.month

features = [
    "Amount", "MerchantID", "TransactionType",
    "Location", "Hour", "DayOfWeek", "Month"
]
X = df[features].copy()
X["MerchantID"] = X["MerchantID"].astype(str)
y = df["IsFraud"]

numeric = ["Amount", "Hour", "DayOfWeek", "Month"]
categorical = ["MerchantID", "TransactionType", "Location"]

preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())
    ]), numeric),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = Pipeline([
    ("prep", preprocessor),
    ("model", LogisticRegression(
        class_weight="balanced", max_iter=1000
    ))
])

print("Training Logistic Regression...")
model.fit(X_train, y_train)
scores = model.predict_proba(X_test)[:, 1]

print("\nROC-AUC:", round(roc_auc_score(y_test, scores), 4))
print("Average Precision:",
      round(average_precision_score(y_test, scores), 4))

print("\nThreshold | TP | FP | FN | TN | Precision% | Recall%")

actual = y_test.to_numpy()

for threshold in [0.01, 0.02, 0.03, 0.05, 0.10, 0.20, 0.30, 0.50]:
    predicted = scores >= threshold
    tp = int(((predicted) & (actual == 1)).sum())
    fp = int(((predicted) & (actual == 0)).sum())
    fn = int(((~predicted) & (actual == 1)).sum())
    tn = int(((~predicted) & (actual == 0)).sum())

    precision = 100 * tp / max(1, tp + fp)
    recall = 100 * tp / max(1, tp + fn)

    print(
        f"{threshold:.2f} | {tp} | {fp} | {fn} | {tn} "
        f"| {precision:.2f} | {recall:.2f}"
    )
