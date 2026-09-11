#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, roc_curve
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
 
pd.set_option("display.max_columns", None)
 
# --- 1. Carga y preprocesamiento (igual que en el script de comparación de librerías) ---
train = pd.read_csv("train.csv")
 
TARGET = "Will_Buy_EV"
y = train[TARGET]
if not pd.api.types.is_numeric_dtype(y):
    y = y.map({"Yes": 1, "No": 0})
 
n_pos = y.sum()
n_neg = len(y) - n_pos
scale_pos_weight = n_neg / n_pos
 
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
 
X_cat = X_raw.copy()
cat_feature_names = ["Gender", "City_Type", "Current_Car_Type",
                      "Home_Charging_Possible", "Subsidy_Available", "Range_Anxiety_Level"]
for col in cat_feature_names:
    X_cat[col] = X_cat[col].astype(str)
cat_feature_idx = [X_cat.columns.get_loc(c) for c in cat_feature_names]
 
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

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
 
xgb = XGBClassifier(
    n_estimators=300, learning_rate=0.05, max_depth=4,
    scale_pos_weight=scale_pos_weight, eval_metric="auc",
    random_state=42, n_jobs=-1,
)
proba_xgb = cross_val_predict(xgb, X, y, cv=skf, method="predict_proba")[:, 1]
 
# CatBoost con bucle manual (cat_features en fit, no en el constructor)
proba_cb = np.zeros(len(y))
for train_idx, val_idx in skf.split(X_cat, y):
    X_tr, X_val = X_cat.iloc[train_idx], X_cat.iloc[val_idx]
    y_tr = y.iloc[train_idx]
    cb = CatBoostClassifier(
        iterations=300, learning_rate=0.05, depth=4,
        auto_class_weights="Balanced", random_state=42, verbose=False,
    )
    cb.fit(X_tr, y_tr, cat_features=cat_feature_idx)
    proba_cb[val_idx] = cb.predict_proba(X_val)[:, 1]
 
auc_gb = roc_auc_score(y, proba_gb)
auc_xgb = roc_auc_score(y, proba_xgb)
auc_cb = roc_auc_score(y, proba_cb)
print(f"AUC HistGradientBoosting: {auc_gb:.4f}")
print(f"AUC XGBoost:              {auc_xgb:.4f}")
print(f"AUC CatBoost:             {auc_cb:.4f}")
 
# --- 3. Correlación entre los errores/predicciones de los tres modelos ---
# Si están muy correlacionados, el ensembling aportará poco (como sospechamos).
oof_df = pd.DataFrame({"GB": proba_gb, "XGB": proba_xgb, "CB": proba_cb})
print("\nCorrelación entre las predicciones OOF de los tres modelos:")
print(oof_df.corr())
 
# --- 4. Ensembling por promedio simple ---
proba_avg = (proba_gb + proba_xgb + proba_cb) / 3
auc_avg = roc_auc_score(y, proba_avg)
print(f"\nAUC Promedio simple (GB+XGB+CB)/3: {auc_avg:.4f}")
 
# --- 5. Stacking: meta-modelo (regresión logística) sobre las predicciones OOF ---
# Usamos las mismas predicciones OOF como features de entrada a un meta-modelo simple.
# Esto es válido porque son "out-of-fold": ningún modelo base vio esos datos al predecirlos.
meta_model = LogisticRegression()
proba_stack = cross_val_predict(
    meta_model, oof_df, y, cv=skf, method="predict_proba"
)[:, 1]
auc_stack = roc_auc_score(y, proba_stack)
print(f"AUC Stacking (meta-modelo logístico sobre las 3 predicciones): {auc_stack:.4f}")
 
# --- 6. Resumen final ---
mejor_individual = max(auc_gb, auc_xgb, auc_cb)
print("\n--- Resumen ---")
print(f"Mejor modelo individual:  {mejor_individual:.4f}")
print(f"Promedio simple:          {auc_avg:.4f}  ({auc_avg - mejor_individual:+.4f})")
print(f"Stacking (meta-modelo):   {auc_stack:.4f}  ({auc_stack - mejor_individual:+.4f})")
 
# --- 7. Curvas ROC comparadas ---
fpr_best, tpr_best, _ = roc_curve(y, proba_gb if auc_gb == mejor_individual else
                                    (proba_xgb if auc_xgb == mejor_individual else proba_cb))
fpr_avg, tpr_avg, _ = roc_curve(y, proba_avg)
fpr_stack, tpr_stack, _ = roc_curve(y, proba_stack)
 
plt.figure(figsize=(5, 5))
plt.plot(fpr_best, tpr_best, label=f"Mejor individual (AUC = {mejor_individual:.3f})")
plt.plot(fpr_avg, tpr_avg, label=f"Promedio simple (AUC = {auc_avg:.3f})")
plt.plot(fpr_stack, tpr_stack, label=f"Stacking (AUC = {auc_stack:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Azar (AUC = 0.5)")
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("Curva ROC - Ensembling")
plt.legend()
plt.tight_layout()
plt.savefig("roc_ensembling.png")
plt.close()
 
print("\nListo. Revisa 'roc_ensembling.png'.")
# %%
