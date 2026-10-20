"""Utilidades de carga y preprocesamiento del proyecto Bank Marketing.

Las decisiones aplicadas aquí están documentadas en el README (§6) y en
`notebooks/02_preprocessing.ipynb`. El objetivo es que todos los notebooks
usen exactamente la misma transformación y que el ajuste ocurra siempre
dentro de un `Pipeline`, únicamente con datos de entrenamiento.

Decisiones principales:
- Se eliminan los duplicados exactos (12 filas, §5).
- Se excluyen las 5 variables no disponibles antes de la llamada (§3):
  `duration`, `campaign`, `month`, `day_of_week` y `contact`.
- Se excluye `pdays`: el 96,3% de sus valores es el centinela 999 y 4.110
  filas lo contradicen (§5); la historia previa queda en `previous` y
  `poutcome`.
- Se excluye `default`: solo 3 `yes` en 41.188 filas y 20,9% `unknown` (§5).
- `unknown` se conserva como categoría propia (One-Hot), sin imputar (§5).
- Numéricas: `StandardScaler`. Categóricas: `OneHotEncoder`.
- El tratamiento del desbalance (`class_weight`) se evalúa en el baseline
  (§7), no se asume aquí.
"""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET = "y"
RAW_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "bank-additional-full.csv"

EXCLUDED_FEATURES = ("duration", "campaign", "month", "day_of_week", "contact")
DROPPED_FEATURES = ("pdays", "default")

NUMERIC_FEATURES = [
    "age",
    "previous",
    "emp.var.rate",
    "cons.price.idx",
    "cons.conf.idx",
    "euribor3m",
    "nr.employed",
]

CATEGORICAL_FEATURES = [
    "job",
    "marital",
    "education",
    "housing",
    "loan",
    "poutcome",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_raw_data(path: Path = RAW_PATH) -> pd.DataFrame:
    """Carga el CSV original sin modificarlo (`;` como separador)."""
    return pd.read_csv(path, sep=";")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina duplicados exactos y valida la columna objetivo."""
    df = df.drop_duplicates().copy()
    valores_y = set(df[TARGET].unique())
    if not valores_y <= {"yes", "no"}:
        raise ValueError(f"Valores inesperados en {TARGET}: {valores_y - {'yes', 'no'}}")
    return df


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Devuelve X (13 features del escenario) e y binaria (yes=1, no=0)."""
    faltantes = [c for c in FEATURES if c not in df.columns]
    if faltantes:
        raise KeyError(f"Faltan columnas en el dataset: {faltantes}")
    X = df.loc[:, FEATURES].copy()
    y = (df[TARGET] == "yes").astype(int)
    y.name = "y_bin"
    return X, y


def train_test_split_data(
    X: pd.DataFrame, y: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split 80/20 estratificado por `y`, con semilla fija (§6)."""
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )


def build_preprocessor() -> ColumnTransformer:
    """Construye el preprocesador (escala numéricas y One-Hot categóricas)."""
    numeric_pipeline = Pipeline(
        steps=[("scaler", StandardScaler())]
    )
    categorical_pipeline = Pipeline(
        steps=[
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            )
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
