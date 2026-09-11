#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, roc_curve
 
pd.set_option("display.max_columns", None)
 
# %%
#procesamiento
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
 
# %%
#el random forest
rf = RandomForestClassifier(
    n_estimators=300,       
    max_depth=5,           
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,              
)
proba_oof_rf = cross_val_predict(
    rf, X, y, cv=skf, method="predict_proba"
)[:, 1]
 
auc_rf = roc_auc_score(y, proba_oof_rf)
print(f"AUC Random Forest (out-of-fold, 5-fold CV): {auc_rf:.4f}")
#%% 
#Calculo del AUC, pero del arbol individual
tree = DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=42)
proba_oof_tree = cross_val_predict(
    tree, X, y, cv=skf, method="predict_proba"
)[:, 1]
auc_tree = roc_auc_score(y, proba_oof_tree)
print(f"AUC Árbol de decisión (referencia, misma CV): {auc_tree:.4f}")
 
print(f"\nMejora de Random Forest sobre árbol individual: {auc_rf - auc_tree:+.4f}")
#%%
fpr_tree, tpr_tree, _ = roc_curve(y, proba_oof_tree)
fpr_rf, tpr_rf, _ = roc_curve(y, proba_oof_rf)
 
plt.figure(figsize=(5, 5))
plt.plot(fpr_tree, tpr_tree, label=f"Árbol de decisión (AUC = {auc_tree:.3f})")
plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC = {auc_rf:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Azar (AUC = 0.5)")
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("Curva ROC - Árbol vs Random Forest")
plt.legend()
plt.tight_layout()
plt.show("roc_tree_vs_rf.png")
plt.close()
#%%
#importancia de las cosas
rf.fit(X, y)  # ajuste final sobre todos los datos, solo para inspeccionar importancias
importances_rf = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
 
print("\nImportancia de variables (Random Forest):")
print(importances_rf)
 
print("\nListo. Revisa 'roc_tree_vs_rf.png' para comparar visualmente ambas curvas ROC.")
 
# %%
