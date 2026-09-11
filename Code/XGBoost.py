#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, roc_curve
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
 
pd.set_option("display.max_columns", None)

train = pd.read_csv("train.csv")
 
TARGET = "Will_Buy_EV"
y = train[TARGET]
if not pd.api.types.is_numeric_dtype(y):
    y = y.map({"Yes": 1, "No": 0})

n_pos = y.sum()
n_neg = len(y) - n_pos
scale_pos_weight = n_neg / n_pos
print(f"scale_pos_weight (ratio negativos/positivos): {scale_pos_weight:.2f}")
 
X_raw = train.drop(columns=[TARGET, "id"], errors="ignore")

X = X_raw.copy()
binary_cols = ["Home_Charging_Possible", "Subsidy_Available"]
for col in binary_cols:
    X[col] = X[col].map({"Yes": 1, "No": 0})
 
ordinal_map = {"Low": 0, "Medium": 1, "High": 2}
X["Range_Anxiety_Level"] = X["Range_Anxiety_Level"].map(ordinal_map)
 
nominal_cols = ["Gender", "City_Type", "Current_Car_Type"]
X = pd.get_dummies(X, columns=nominal_cols, drop_first=False)
X["Concern_x_Subsidy"] = X["Environmental_Concern_Level"] * X["Subsidy_Available"]
 
assert X.select_dtypes(include=["object", "string"]).empty
 
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

X_cat = X_raw.copy()
cat_feature_names = ["Gender", "City_Type", "Current_Car_Type",
                      "Home_Charging_Possible", "Subsidy_Available", "Range_Anxiety_Level"]
for col in cat_feature_names:
    X_cat[col] = X_cat[col].astype(str)
cat_feature_idx = [X_cat.columns.get_loc(c) for c in cat_feature_names]

gb_best = HistGradientBoostingClassifier(
    learning_rate=0.2443549100736809,
    max_depth=3,
    max_iter=233,
    max_leaf_nodes=42,
    min_samples_leaf=37,
    l2_regularization=0.9296976523425731,
    class_weight="balanced",
    random_state=42,
)
proba_gb = cross_val_predict(gb_best, X, y, cv=skf, method="predict_proba")[:, 1]
auc_gb = roc_auc_score(y, proba_gb)
print(f"\nAUC HistGradientBoosting (afinado, referencia): {auc_gb:.4f}")

xgb = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    scale_pos_weight=scale_pos_weight,
    eval_metric="auc",
    random_state=42,
    n_jobs=-1,
)
proba_xgb = cross_val_predict(xgb, X, y, cv=skf, method="predict_proba")[:, 1]
auc_xgb = roc_auc_score(y, proba_xgb)
print(f"AUC XGBoost: {auc_xgb:.4f}")


proba_cb = np.zeros(len(y))

for train_idx, val_idx in skf.split(X_cat, y):
    X_tr, X_val = X_cat.iloc[train_idx], X_cat.iloc[val_idx]
    y_tr = y.iloc[train_idx]

    cb = CatBoostClassifier(
        iterations=300,
        learning_rate=0.05,
        depth=4,
        auto_class_weights="Balanced",
        random_state=42,
        verbose=False,
    )
    cb.fit(X_tr, y_tr, cat_features=cat_feature_idx)
    proba_cb[val_idx] = cb.predict_proba(X_val)[:, 1]

auc_cb = roc_auc_score(y, proba_cb)
print(f"AUC CatBoost (categóricas nativas): {auc_cb:.4f}")
print("\n--- Resumen ---")
print(f"HistGradientBoosting (afinado): {auc_gb:.4f}")
print(f"XGBoost:                        {auc_xgb:.4f}")
print(f"CatBoost:                       {auc_cb:.4f}")

fpr_gb, tpr_gb, _ = roc_curve(y, proba_gb)
fpr_xgb, tpr_xgb, _ = roc_curve(y, proba_xgb)
fpr_cb, tpr_cb, _ = roc_curve(y, proba_cb)
 
plt.figure(figsize=(5, 5))
plt.plot(fpr_gb, tpr_gb, label=f"HistGradientBoosting (AUC = {auc_gb:.3f})")
plt.plot(fpr_xgb, tpr_xgb, label=f"XGBoost (AUC = {auc_xgb:.3f})")
plt.plot(fpr_cb, tpr_cb, label=f"CatBoost (AUC = {auc_cb:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Azar (AUC = 0.5)")
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("Curva ROC - Comparación de librerías de boosting")
plt.legend()
plt.tight_layout()
plt.savefig("roc_libraries_comparison.png")
plt.close()
 
print("\nListo. Revisa 'roc_libraries_comparison.png'.")
 
# %%
