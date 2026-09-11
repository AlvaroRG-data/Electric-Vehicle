# Predicting Electric Vehicle Purchases (Kaggle Playground Series S6E9)

Proyecto de portfolio centrado en un problema de **clasificación binaria** evaluado con **ROC-AUC**: predecir si un cliente comprará o no un vehículo eléctrico a partir de datos demográficos, económicos y de contexto (infraestructura de carga, subsidios, tipo de vehículo actual, etc.).

Competición: [Playground Series - Season 6, Episode 9](https://www.kaggle.com/competitions/playground-series-s6e9/overview)

## Objetivo del proyecto

Más allá de maximizar la puntuación en el leaderboard, el objetivo de este proyecto era **entender en profundidad la progresión de modelos de clasificación basados en árboles** (de un árbol individual a gradient boosting) y **diagnosticar honestamente por qué un modelo mejora o deja de mejorar**, en vez de aplicar técnicas de forma mecánica.

## Dataset

- Variable objetivo: `Will_Buy_EV` (binaria, ~90/10 de desbalance hacia "no compra").
- Variables numéricas: `Environmental_Concern_Level`, `Annual_Income_USD`, `Number_of_Cars_Owned`, `Age`, `Charging_Stations_Near_Work`, `Charging_Stations_Near_Home`, `Daily_Commute_km`.
- Variables categóricas: `Gender`, `City_Type`, `Current_Car_Type` (nominales), `Home_Charging_Possible`, `Subsidy_Available` (binarias), `Range_Anxiety_Level` (ordinal).

## Metodología

Se descartó un enfoque inicial basado en clustering de clientes: al tratarse de un problema con etiqueta disponible y evaluado por ranking (ROC-AUC), un modelo supervisado aprovecha mejor la información de compra que una segmentación no supervisada, que optimiza un objetivo distinto (similitud interna de grupos, no separación por probabilidad de compra).

Se siguió una progresión deliberada de modelos, evaluando siempre con **validación cruzada estratificada de 5 folds** y prediciones **out-of-fold** para obtener estimaciones de AUC no sesgadas:

1. **Árbol de decisión** (`max_depth=5`, `class_weight="balanced"`) — modelo base interpretable.
2. **Random Forest** (bagging) — para reducir varianza sobre el árbol individual.
3. **Gradient Boosting** (`HistGradientBoostingClassifier`) — para reducir sesgo mediante corrección secuencial de errores.
4. **Feature engineering**: interacciones explícitas entre las variables más importantes (`Environmental_Concern_Level`, `Subsidy_Available`, `Annual_Income_USD`).
5. **Tuning de hiperparámetros** (`RandomizedSearchCV`, 40 combinaciones) sobre el mejor modelo.
6. **Comparación de librerías**: XGBoost y CatBoost (este último con manejo nativo de categóricas, sin one-hot).
7. **Ensembling**: promedio simple y stacking (meta-modelo logístico) de los tres modelos de boosting.

## Resultados

| Paso | AUC (out-of-fold) | Mejora |
|---|---|---|
| Árbol de decisión | 0.9323 | — |
| Random Forest | 0.9348 | +0.0025 |
| Gradient Boosting | 0.9410 | +0.0062 |
| + Feature engineering (interacciones) | 0.9410 | +0.0000 |
| + Tuning de hiperparámetros | 0.9417 | +0.0007 |
| XGBoost | 0.9412 | −0.0005 |
| CatBoost (categóricas nativas) | 0.9405 | −0.0012 |
| Ensembling (promedio simple) | 0.9414 | −0.0003 |
| Ensembling (stacking) | 0.9417 | +0.0000 |

## Conclusiones

- **El techo de rendimiento del problema está en torno a AUC = 0.94**, y se alcanza ya con un Gradient Boosting razonable sin apenas ajuste. Ninguna técnica adicional (feature engineering manual, tuning exhaustivo, cambio de librería, ensembling) logró superar ese techo de forma significativa.
- **La señal predictiva está muy concentrada en 2-3 variables** (`Environmental_Concern_Level`, `Subsidy_Available`, y en menor medida `Annual_Income_USD`), confirmado de forma consistente por tres métodos de importancia distintos (basada en splits en árbol y Random Forest, permutación en Gradient Boosting).
- **Las interacciones manuales entre esas variables no aportaron señal nueva**: el AUC se mantuvo exactamente igual (0.9410 antes y después), porque los modelos basados en árboles ya aproximaban esas interacciones mediante splits sucesivos.
- **Los tres modelos de boosting probados dieron predicciones altamente correlacionadas (>0.996)**, lo que explica por qué el ensembling no aportó mejora: al no haber diversidad de errores entre modelos, promediar o apilar sus predicciones no añade información nueva.
- Esta consistencia entre técnicas tan distintas (bagging, boosting, feature engineering, tuning, cambio de implementación, ensembling) es en sí misma una conclusión sólida: indica que el techo observado es una propiedad real del dataset, no una limitación de alguna técnica concreta.

## Stack técnico

Python · pandas · scikit-learn (`DecisionTreeClassifier`, `RandomForestClassifier`, `HistGradientBoostingClassifier`, `RandomizedSearchCV`) · XGBoost · CatBoost · matplotlib

## Estructura de scripts

- `eda_ev_purchases.py` — carga y exploración inicial del dataset.
- `tree_model_auc.py` — árbol de decisión y curva ROC.
- `random_forest_auc.py` — Random Forest, comparado contra el árbol.
- `gradient_boosting_auc.py` — Gradient Boosting, comparado contra los dos anteriores.
- `feature_engineering_auc.py` — interacciones entre variables dominantes.
- `gb_hyperparameter_tuning.py` — búsqueda de hiperparámetros con `RandomizedSearchCV`.
- `xgboost_catboost_comparison.py` — comparación con XGBoost y CatBoost.
- `ensembling_stacking.py` — promedio simple y stacking de los tres modelos de boosting.
