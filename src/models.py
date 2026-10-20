"""Definición de modelos candidatos (§8) y métricas de evaluación.

Todos los estimadores se construyen como `Pipeline` que incluyen el
preprocesador de `src.preprocessing`, de modo que el ajuste ocurra siempre
dentro de la partición de entrenamiento (sin fuga de información).

La métrica principal y el umbral se fijan en el notebook de validación (§9);
aquí solo se declaran los candidatos y los espacios de búsqueda.
"""

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.preprocessing import RANDOM_STATE, build_preprocessor

_SCORERS = {
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
}


def build_pipeline(model) -> Pipeline:
    """Pipeline preprocesador (no ajustado) + modelo."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", model),
        ]
    )


def build_candidate_models() -> dict:
    """Modelos candidatos con hiperparámetros razonables (sin tuning aún)."""
    return {
        "Regresion logistica": build_pipeline(
            LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        ),
        "Regresion logistica balanceada": build_pipeline(
            LogisticRegression(
                max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
            )
        ),
        "Arbol de decision": build_pipeline(
            DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE)
        ),
        "Random Forest": build_pipeline(
            RandomForestClassifier(
                n_estimators=400,
                min_samples_leaf=5,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=RANDOM_STATE,
            )
        ),
        "Gradient Boosting": build_pipeline(
            GradientBoostingClassifier(random_state=RANDOM_STATE)
        ),
        "KNN": build_pipeline(
            KNeighborsClassifier(n_neighbors=25, n_jobs=-1)
        ),
    }
