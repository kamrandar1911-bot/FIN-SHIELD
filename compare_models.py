
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

# Load existing dataset
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
y = df["IsFraud"]
X["MerchantID"] = X["MerchantID"].astype(str)

numeric = ["Amount", "Hour", "DayOfWeek", "Month"]
categorical = ["MerchantID", "TransactionType", "Location"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),
    "Extra Trees": ExtraTreesClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),
    "Hist Gradient Boosting": HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.1,
        random_state=42
    )
}

# HistGradientBoosting needs numeric input, so use ordinal encoding
from sklearn.preprocessing import OrdinalEncoder

results = []

for name, classifier in models.items():
    if name == "Hist Gradient Boosting":
        preprocessor = ColumnTransformer([
            ("num", SimpleImputer(strategy="median"), numeric),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1
                ))
            ]), categorical)
        ])
    else:
        preprocessor = ColumnTransformer([
            ("num", SimpleImputer(strategy="median"), numeric),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ]), categorical)
        ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])

    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)

    scores = model.predict_proba(X_test)[:, 1]

    ap = average_precision_score(y_test, scores)
    auc = roc_auc_score(y_test, scores)

    results.append({
        "Model": name,
        "Average Precision": round(ap, 4),
        "ROC-AUC": round(auc, 4)
    })

print("\n=== MODEL COMPARISON ===")
print(pd.DataFrame(results).sort_values(
    "Average Precision", ascending=False
).to_string(index=False))

print("\nHigher Average Precision is better for this imbalanced dataset.")
