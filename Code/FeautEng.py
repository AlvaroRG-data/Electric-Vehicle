#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.inspection import permutation_importance
 
pd.set_option("display.max_columns", None)
train = pd.read_csv("train.csv")
 
TARGET = "Will_Buy_EV"
y = train[TARGET]
if not pd.api.types.is_numeric_dtype(y):
    y = y.map({"Yes": 1, "No": 0})
 
X = train.drop(columns=[TARGET, "id"], errors="ignore")
 
binary_cols = ["Home_Charging_Possible", "Subsidy_Available"]
for col in binary_cols:
    X[col] = X[col].map({"Yes": 1, "No": 0})
 
ordinal_map = {"Low": 0, "Medium": 1, "High": 2}
X["Range_Anxiety_Level"] = X["Range_Anxiety_Level"].map(ordinal_map)
 
nominal_cols = ["Gender", "City_Type", "Current_Car_Type"]
X = pd.get_dummies(X, columns=nominal_cols, drop_first=False)
 
assert X.select_dtypes(include=["object", "string"]).empty, \
    "Quedan columnas sin codificar - revisa el preprocesamiento"
 
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
 
gb_params = dict(
    max_iter=300,
    learning_rate=0.05,
    max_depth=4,
    class_weight="balanced",
    random_state=42,
)
 
# %%
gb_base = HistGradientBoostingClassifier(**gb_params)
proba_base = cross_val_predict(gb_base, X, y, cv=skf, method="predict_proba")[:, 1]
auc_base = roc_auc_score(y, proba_base)
print(f"AUC Gradient Boosting SIN interacciones: {auc_base:.4f}")
#%%
#modelo con iteracciones
X_fe = X.copy()
 
X_fe["Concern_x_Subsidy"] = X_fe["Environmental_Concern_Level"] * X_fe["Subsidy_Available"]
X_fe["Concern_x_Income"] = X_fe["Environmental_Concern_Level"] * X_fe["Annual_Income_USD"]
X_fe["Subsidy_x_Income"] = X_fe["Subsidy_Available"] * X_fe["Annual_Income_USD"]
X_fe["All_Three"] = (
    X_fe["Environmental_Concern_Level"]
    * X_fe["Subsidy_Available"]
    * X_fe["Annual_Income_USD"]
)
 
print(f"\nColumnas nuevas añadidas: {['Concern_x_Subsidy', 'Concern_x_Income', 'Subsidy_x_Income', 'All_Three']}")
 
# %%
gb_fe = HistGradientBoostingClassifier(**gb_params)
proba_fe = cross_val_predict(gb_fe, X_fe, y, cv=skf, method="predict_proba")[:, 1]
auc_fe = roc_auc_score(y, proba_fe)
print(f"AUC Gradient Boosting CON interacciones: {auc_fe:.4f}")
 
print(f"\nMejora por feature engineering: {auc_fe - auc_base:+.4f}")
 #%%
 #roc
fpr_base, tpr_base, _ = roc_curve(y, proba_base)
fpr_fe, tpr_fe, _ = roc_curve(y, proba_fe)
 
plt.figure(figsize=(5, 5))
plt.plot(fpr_base, tpr_base, label=f"Sin interacciones (AUC = {auc_base:.3f})")
plt.plot(fpr_fe, tpr_fe, label=f"Con interacciones (AUC = {auc_fe:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Azar (AUC = 0.5)")
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("Curva ROC - Efecto del feature engineering")
plt.legend()
plt.tight_layout()
plt.savefig("roc_feature_engineering.png")
plt.close()
 #%%
 #importancia
gb_fe.fit(X_fe, y)
perm = permutation_importance(
    gb_fe, X_fe, y, scoring="roc_auc", n_repeats=5, random_state=42, n_jobs=-1
)
importances_fe = pd.Series(perm.importances_mean, index=X_fe.columns).sort_values(ascending=False)
 
print("\nImportancia de variables CON interacciones (permutation importance sobre AUC):")
print(importances_fe)
 
print("\nListo. Revisa 'roc_feature_engineering.png' para ver el efecto visual.")
 