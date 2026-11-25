from pathlib import Path
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix
)


# ==========================================================
# 1 PATHS SETTINGS
# ==========================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

DATASET_PATH = (
    ROOT_DIR
    / "data"
    / "scenarios"
    / "synthetic_training_windows.csv"
)

ARTIFACT_DIR = (
    ROOT_DIR
    / "models"
    / "unsupervised"
    / "artifacts"
)

MODEL_PATH = ARTIFACT_DIR / "isolation_forest.joblib"


# ==========================================================
# 2 FEATURES Y TARGET
# ==========================================================

FEATURES = [
    "failed_login_count",
    "connection_count",
    "unique_destinations",
    "unique_ports",
    "denied_count",
    "db01_attempt_count"
]

TARGET = "label"


# ==========================================================
# 3 LOADING DATASET
# ==========================================================

if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"No se encontró el dataset: {DATASET_PATH}"
    )

df = pd.read_csv(DATASET_PATH)

print("\n=== INFORMACIÓN DEL DATASET ===")

print("Total de registros:", len(df))
print("Dimensiones:", df.shape)


# ==========================================================
# 4 COLUMNS VALIDATION
# ==========================================================

required_columns = FEATURES + [TARGET]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Faltan columnas: {missing_columns}"
    )


# ==========================================================
# 5 DATA VALIDATION
# ==========================================================

for column in FEATURES:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

df[TARGET] = pd.to_numeric(
    df[TARGET],
    errors="coerce"
)

if df[required_columns].isna().any().any():
    raise ValueError(
        "Hay valores faltantes o no numéricos."
    )

if not df[TARGET].isin([0, 1]).all():
    raise ValueError(
        "Las etiquetas deben ser 0 o 1."
    )

df[TARGET] = df[TARGET].astype(int)

print("\nDistribución de etiquetas:")
print(df[TARGET].value_counts().sort_index()) 


# ==========================================================
# 6 SEPARATING FEATURES AND LABELS
# ==========================================================

X = df[FEATURES]
y = df[TARGET]


# ==========================================================
# 7 DEVIDING TRAINING AND TESTING
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n=== DIVISIÓN DE DATOS ===")

print("Entrenamiento:", len(X_train))
print("Prueba:", len(X_test))


# ==========================================================
# 8 REFERENCE BUILDING OF A NORMAL BEHAVIOR 
# ==========================================================

X_train_normal = X_train.loc[y_train == 0]

if len(X_train_normal) < 10:
    raise ValueError(
        "Hay muy pocos ejemplos normales para entrenar."
    )

print(
    "\nVentanas normales utilizadas para entrenar:",
    len(X_train_normal)
)


# ==========================================================
# 9 CREATING ISOLATION FOREST MODEL
# ==========================================================

model = IsolationForest(
    n_estimators=200,
    contamination=0.05,
    max_samples="auto",
    random_state=42,
    n_jobs=-1
)


# ==========================================================
# 10 MODEL TRAINING
# ==========================================================

model.fit(X_train_normal)

print("\nIsolation Forest entrenado correctamente.")


# ==========================================================
# 11 DOING PREDICTIONS
# ==========================================================

raw_predictions = model.predict(X_test)


# Isolation Forest:
#  1 = normal
# -1 = anomalía
#
# dataset:
#  0 = normal
#  1 = sospechoso

y_pred = np.where(
    raw_predictions == -1,
    1,
    0
)


# ==========================================================
# 12 GETTING ANOMALY SCORE
# ==========================================================

anomaly_scores = model.decision_function(X_test)


# ==========================================================
# 13 RESULTS EVALUATION
# ==========================================================

print("\n===== MATRIZ DE CONFUSIÓN =====")

matrix = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1]
)

print(matrix)

print("\nFilas: valores reales (0, 1)")
print("Columnas: predicciones (0, 1)")


print("\n===== REPORTE DE CLASIFICACIÓN =====")

print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Normal", "Sospechoso"],
        zero_division=0
    )
)


# ==========================================================
# 14 CHECKING ANOMALY OBSERVATIONS
# ==========================================================

results = pd.DataFrame({
    "real_label": y_test.to_numpy(),
    "predicted_label": y_pred,
    "anomaly_score": anomaly_scores
})

results = results.sort_values(
    by="anomaly_score",
    ascending=True
)

print("\n=== 10 OBSERVACIONES MÁS ANÓMALAS ===")

print(results.head(10).to_string(index=False))


# ==========================================================
# 15 SAVING THE MODEL
# ==========================================================

ARTIFACT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

artifact = {
    "model": model,
    "features": FEATURES,
    "target": TARGET,
    "normal_label": 0,
    "anomaly_label": 1,
    "contamination": 0.05
}

dump(
    artifact,
    MODEL_PATH
)

print("\n===== FINALIZADO =====")

print(
    f"Modelo guardado en: {MODEL_PATH}"
)
