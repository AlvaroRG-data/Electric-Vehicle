#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform, randint
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, cross_val_predict
from sklearn.metrics import roc_auc_score, roc_curve
 
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

X["Concern_x_Subsidy"] = X["Environmental_Concern_Level"] * X["Subsidy_Available"]
 
assert X.select_dtypes(include=["object", "string"]).empty, \
    "Quedan columnas sin codificar - revisa el preprocesamiento"
 
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

param_dist = {
    "learning_rate": uniform(0.01, 0.29),        # entre 0.01 y 0.30
    "max_iter": randint(100, 600),                # entre 100 y 600 árboles
    "max_depth": randint(3, 8),                   # entre 3 y 7 niveles
    "max_leaf_nodes": randint(15, 63),            # entre 15 y 62 hojas por árbol
    "l2_regularization": uniform(0.0, 1.0),       # entre 0 y 1
    "min_samples_leaf": randint(10, 100),         # mínimo de muestras por hoja
}
 
base_model = HistGradientBoostingClassifier(
    class_weight="balanced",
    random_state=42,
)

search = RandomizedSearchCV(
    base_model,
    param_distributions=param_dist,
    n_iter=40,             # nº de combinaciones a probar; sube esto si tienes tiempo/paciencia
    scoring="roc_auc",
    cv=skf,
    random_state=42,
    n_jobs=-1,
    verbose=1,
)
 
search.fit(X, y)
 
print(f"\nMejor AUC en CV durante la búsqueda: {search.best_score_:.4f}")
print("Mejores hiperparámetros encontrados:")
for param, value in search.best_params_.items():
    print(f"  {param}: {value}")

best_model = search.best_estimator_
proba_tuned = cross_val_predict(best_model, X, y, cv=skf, method="predict_proba")[:, 1]
auc_tuned = roc_auc_score(y, proba_tuned)
print(f"\nAUC del modelo afinado (out-of-fold, misma CV que los anteriores): {auc_tuned:.4f}")
auc_previo = 0.9410
print(f"Mejora respecto al Gradient Boosting sin afinar: {auc_tuned - auc_previo:+.4f}")

fpr_tuned, tpr_tuned, _ = roc_curve(y, proba_tuned)
 
plt.figure(figsize=(5, 5))
plt.plot(fpr_tuned, tpr_tuned, label=f"GB afinado (AUC = {auc_tuned:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Azar (AUC = 0.5)")
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("Curva ROC - Gradient Boosting afinado")
plt.legend()
plt.tight_layout()
plt.savefig("roc_gb_tuned.png")
plt.close()
 
print("\nListo. Revisa 'roc_gb_tuned.png'.")
# %%
