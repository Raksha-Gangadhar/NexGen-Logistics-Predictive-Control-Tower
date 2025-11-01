# train.py
import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

DATA_PATH = Path("merged_all.csv")
MODEL_PATH = Path("model_bundle.pkl")

# 1) Load data ---------------------------------------------------------------
if not DATA_PATH.exists():
    raise FileNotFoundError("merged_all.csv not found. Run merge_data.py first.")
df = pd.read_csv(DATA_PATH)

# 2) Target handling ---------------------------------------------------------
if "Is_Delayed" not in df.columns:
    raise RuntimeError("Column 'Is_Delayed' missing. Ensure merge_data.py created it.")

# Fill missing target with 0 (assume not delayed) and cast to int
df["Is_Delayed"] = df["Is_Delayed"].fillna(0).astype(int)
y = df["Is_Delayed"]

# 3) Candidate features (only keep those that exist) -------------------------
candidates = [
    "Priority","Product_Category","Origin","Destination","Route",
    "Weather_Impact","Special_Handling","Customer_Segment",
    "Assigned_Vehicle_Type","Assigned_Vehicle_FuelEff","Assigned_Vehicle_Age",
    "Distance_KM","Traffic_Delay_Minutes","Toll_Charges_INR",
    "Order_Value_INR","Total_Cost_INR","CO2_Est_kg"
]
features = [c for c in candidates if c in df.columns]
if not features:
    raise RuntimeError("No model features found. Check merged_all.csv columns.")

X = df[features].copy()

# 4) Basic cleaning on numerics ----------------------------------------------
# Coerce obvious numeric columns to numeric (errors -> NaN)
for col in features:
    if any(tok in col.lower() for tok in ["km", "cost", "age", "value", "toll", "fuel", "traffic", "co2", "distance"]):
        X[col] = pd.to_numeric(X[col], errors="coerce")

# Split into numeric & categorical by dtype after coercion
num_cols = [c for c in X.columns if X[c].dtype.kind in "iufc"]
cat_cols = [c for c in X.columns if c not in num_cols]

# 5) Preprocess & model pipeline ---------------------------------------------
numeric_pipe = Pipeline(steps=[
    ("impute", SimpleImputer(strategy="median"))
])
categorical_pipe = Pipeline(steps=[
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(handle_unknown="ignore"))
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_pipe, num_cols),
        ("cat", categorical_pipe, cat_cols),
    ],
    remainder="drop"
)

model = RandomForestClassifier(
    n_estimators=400,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

clf = Pipeline(steps=[
    ("prep", preprocess),
    ("rf", model)
])

# 6) Train / test split ------------------------------------------------------
# If all targets are a single class, stratify will fail; handle gracefully
stratify_arg = y if y.nunique() > 1 else None

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=stratify_arg
)

# 7) Fit ---------------------------------------------------------------------
clf.fit(X_train, y_train)

# 8) Evaluate ----------------------------------------------------------------
try:
    proba = clf.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    auc = roc_auc_score(y_test, proba) if y_test.nunique() > 1 else float("nan")
    print("AUC:", round(auc, 3) if np.isfinite(auc) else "NA (single-class test)")
    print(classification_report(y_test, pred, digits=3))
except Exception as e:
    # Some classifiers/configs might not expose predict_proba if changed; fallback
    pred = clf.predict(X_test)
    print("Note:", type(e).__name__, "-", str(e))
    print(classification_report(y_test, pred, digits=3))

# 9) Save model bundle -------------------------------------------------------
bundle = {
    "model": clf,
    "features": features,
    "numeric_cols": num_cols,
    "categorical_cols": cat_cols,
}
joblib.dump(bundle, MODEL_PATH)
print(f"✅ Saved {MODEL_PATH}")
print("Features used:", features)

# 10) Optional: small training summary file ----------------------------------
summary = {
    "rows_total": int(len(df)),
    "rows_train": int(len(X_train)),
    "rows_test": int(len(X_test)),
    "features_used": features,
    "num_cols": num_cols,
    "cat_cols": cat_cols,
}
Path("training_summary.json").write_text(json.dumps(summary, indent=2))
print("Wrote training_summary.json")
