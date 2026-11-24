from pathlib import Path

import pandas as pd
from joblib import dump

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


# ==========================================================
# 1 DEFINITION OF PATHS DEFINITIONS
# ==========================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

DATASET_PATH = (
    ROOT_DIR
    / "data"
    / "scenarios"
    / "synthetic_training_windows.csv"
)

MODEL_DIR = (
    ROOT_DIR
    / "models"
    / "supervised"
    / "artifacts"
)

MODEL_PATH = MODEL_DIR / "random_forest.joblib"


# ==========================================================
# 2 FEATURES Y TARGET DEFINITION
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
# 3 DATASET LOADING
# ==========================================================

if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"No se encontró el dataset: {DATASET_PATH}"
    )

df = pd.read_csv(DATASET_PATH)

print("\n=== INFORMACIÓN DEL DATASET ===")
print("Cantidad de registros:", len(df))
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
        f"Faltan columnas necesarias: {missing_columns}"
    )


# ==========================================================
# 5 VALIDATE DATA TYPES AND MISSING VALUES
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
    missing_report = (
        df[required_columns]
        .isna()
        .sum()
    )

    raise ValueError(
        "Existen valores faltantes o no numéricos:\n"
        f"{missing_report}"
    )

if not df[TARGET].isin([0, 1]).all():
    raise ValueError(
        "La columna label debe contener únicamente 0 y 1."
    )

df[TARGET] = df[TARGET].astype(int)


# ==========================================================
# 6 CHECK CLASS DISTRIBUTION
# ==========================================================

print("\n========== DISTRIBUCIÓN DE ETIQUETAS ==========")

class_counts = df[TARGET].value_counts().sort_index()

print(class_counts)

if set(class_counts.index) != {0, 1}:
    raise ValueError(
        "Se necesitan ejemplos de ambas clases: "
        "0 (normal) y 1 (sospechoso)."
    )

if len(df) < 10 or class_counts.min() < 2:
    raise ValueError(
        "Hay muy pocos registros para realizar "
        "una división estratificada."
    )

if class_counts.min() < 20:
    print(
        "\nADVERTENCIA: Hay pocos ejemplos de al menos "
        "una clase. El entrenamiento será únicamente "
        "experimental y sus métricas no serán confiables."
    )


# ==========================================================
# 7 FEATURES (X) Y TARGET (y) SEPARATION
# ==========================================================

X = df[FEATURES]
y = df[TARGET]

print("\nFeatures utilizadas:")
print(FEATURES)


# ==========================================================
# 8 DIVIDE DATASET INTO TRAINING AND TEST SETS
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n=== DIVISIÓN DE DATOS ===")
print("Registros de entrenamiento:", len(X_train))
print("Registros de prueba:", len(X_test))


# ==========================================================
# 9 CREATE AND CONFIGURE RANDOM FOREST MODEL
# ==========================================================

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)


# ==========================================================
# 10 TRAIN THE MODEL
# ==========================================================

model.fit(X_train, y_train)

print("\nModelo Random Forest entrenado correctamente.")


# ==========================================================
# 11 MAKE PREDICTIONS ON TEST SET
# ==========================================================

y_pred = model.predict(X_test)


# ==========================================================
# 12 RESULTS EVALUATION
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


print("\n==== REPORTE DE CLASIFICACIÓN ===")

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
# 13 FEATURES RELEVATION
# ==========================================================

print("\n===IMPORTANCIA DE CARACTERÍSTICAS ===")

importance = pd.Series(
    model.feature_importances_,
    index=FEATURES
).sort_values(ascending=False)

print(importance)


# ==========================================================
# 14 SAVE THE MODEL
# ==========================================================

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

artifact = {
    "model": model,
    "features": FEATURES,
    "target": TARGET
}

dump(artifact, MODEL_PATH)

print("\n===== FINALIZADO =====")
print(f"Modelo guardado en: {MODEL_PATH}")
