#%%
#Empezamos con lo habitual de carga y eso
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, roc_curve
pd.set_option("display.max_columns", None)
 
# %%
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
#%% 
print("Shape train:", train.shape)
print("Shape test:", test.shape)
print("\nPrimeras filas:")
print(train.head())
# %%
print("\nTipos de datos:")
print(train.dtypes)
 
print("\nNulos por columna (train):")
nulos = train.isnull().sum()
print(nulos[nulos > 0].sort_values(ascending=False))
 
# %%
TARGET = [c for c in train.columns if c.lower() in ("target", "purchase", "class")]
target_col = TARGET[0] if TARGET else train.columns[-1]
print(f"\nUsando '{target_col}' como variable objetivo")
 
y = train[target_col]
if not pd.api.types.is_numeric_dtype(y):
    y = y.map({"Yes": 1, "No": 0})
X = train.drop(columns=[target_col, "id"], errors="ignore")
print(train[target_col].unique())
 
# %%
print("\nBalance de la variable objetivo:")
print(y.value_counts(normalize=True))
#%%
plt.figure(figsize=(4, 3))
y.value_counts().plot(kind="bar")
plt.title("Distribución de la variable objetivo")
plt.xlabel(target_col)
plt.ylabel("Frecuencia")
plt.tight_layout()
plt.show("target_balance.png")
plt.close()
# %%
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
 
print(f"\nColumnas categóricas ({len(cat_cols)}): {cat_cols}")
print(f"Columnas numéricas ({len(num_cols)}): {num_cols}")
# %%
if cat_cols:
    print("\nCardinalidad de columnas categóricas:")
    for col in cat_cols:
        print(f"  {col}: {X[col].nunique()} niveles -> {X[col].unique()[:10]}")
 
# %%
print("\nDescripción de columnas numéricas:")
print(X[num_cols].describe().T)
#%%
print(y.dtype)
print(y.unique())
# %%
if num_cols:
    corr_with_target = X[num_cols].assign(**{target_col: y}).corr()[target_col].drop(target_col)
    print("\nCorrelación (Pearson) de numéricas con el target:")
    print(corr_with_target.sort_values(ascending=False))
 
print("\nEDA inicial completado. Revisa 'target_balance.png' para el gráfico de balance de clases.")
# %%
#Vamos a volver las colunmas correspondientes en numerico
binary_cols = ["Home_Charging_Possible", "Subsidy_Available"]
for col in binary_cols:
    X[col] = X[col].map({"Yes": 1, "No": 0})
 
# Ordinal -> mapeo respetando el orden
ordinal_map = {"Low": 0, "Medium": 1, "High": 2}
X["Range_Anxiety_Level"] = X["Range_Anxiety_Level"].map(ordinal_map)
 
# Nominales -> one-hot
nominal_cols = ["Gender", "City_Type", "Current_Car_Type"]
X = pd.get_dummies(X, columns=nominal_cols, drop_first=False)
 
print("Shape final de X:", X.shape)
print("Columnas finales:", X.columns.tolist())
 
# Comprobación: no deben quedar columnas de tipo texto
assert X.select_dtypes(include=["object", "string"]).empty, \
    "Quedan columnas sin codificar - revisa el preprocesamiento"
# %%
#Arbol de decision
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
 
tree = DecisionTreeClassifier(
    max_depth=5,           
    class_weight="balanced",  
    random_state=42,
)
proba_oof = cross_val_predict(
    tree, X, y, cv=skf, method="predict_proba"
)[:, 1]
 
auc_oof = roc_auc_score(y, proba_oof)
print(f"\nAUC (out-of-fold, 5-fold CV estratificada): {auc_oof:.4f}")
#%%
#Curva ROC con este modelo
fpr, tpr, thresholds = roc_curve(y, proba_oof)

plt.figure(figsize=(5, 5))
plt.plot(fpr, tpr, label=f"Árbol de decisión (AUC = {auc_oof:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Azar (AUC = 0.5)")
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("Curva ROC - Árbol de decisión")
plt.legend()
plt.tight_layout()
plt.show("roc_decision_tree.png")
plt.close()
 
# %%
#para ver importancias
tree.fit(X, y)  
importances = pd.Series(tree.feature_importances_, index=X.columns).sort_values(ascending=False)
 
print("\nImportancia de variables (árbol de decisión):")
print(importances)
 
print("\nListo. Revisa 'roc_decision_tree.png' para ver la curva ROC.") 
#%%
