
from pathlib import Path
import pandas as pd

# ==========================================================
# 1 CONFIGURATION
# ==========================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

DATASET_PATH = (
    ROOT_DIR
    / "data"
    / "scenarios"
    / "synthetic_training_windows.csv"
)

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
# 2 LOADING DATASET
# ==========================================================

if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"No se encontro el dataset: {DATASET_PATH}"
    )

df = pd.read_csv(DATASET_PATH)

print("\n=== AUDITORIA DEL DATASET ===")

print("\nDimensiones:", df.shape)

print("\nColumnas:")
print(df.columns.tolist())


# ==========================================================
# 3 DATA VALIDATE
# ==========================================================

required_columns = FEATURES + [TARGET, "scenario_type"]

missing = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing:
    raise ValueError(
        f"Faltan columnas: {missing}"
    )

print("\nValores faltantes:")
print(df[required_columns].isna().sum())

print("\nRegistros duplicados en todas las columnas:")
print(df.duplicated().sum())

print("\nRegistros duplicados en las features:")
print(df.duplicated(subset=FEATURES).sum())


# ==========================================================
# 4 CLASS DISTRIBUTION
# ==========================================================

print("\n=== DISTRIBUCION DE ETIQUETAS ===")

print(df[TARGET].value_counts().sort_index())

print("\n=== TIPOS DE ESCENARIOS ===")

print(
    df.groupby(
        ["scenario_type", TARGET]
    ).size()
)


# ==========================================================
# 5 STADISTICS PER CLASS
# ==========================================================

print("\n=== ESTADISTICAS POR CLASE ===")

for label in sorted(df[TARGET].unique()): 

    subset = df[df[TARGET] == label]

    class_name = (
        "NORMAL" if label == 0 else "SOSPECHOSO"
    )

    print(f"\n--- {class_name} ---")

    print(
        subset[FEATURES].agg(
            ["min", "median", "max", "mean"]
        ).round(2)
    )


# ==========================================================
# 6 SCENARIO COMPARATION
# ==========================================================

print("\n=== PROMEDIOS POR ESCENARIO ===")

print(
    df.groupby("scenario_type")[FEATURES]
    .mean()
    .round(2)
)


# ==========================================================
# 7 VALIDATE GENERATOR RULES
# ==========================================================

print("\n=== VALIDACION DE REGLAS ===")

normal_db_attempts = df.loc[
    df[TARGET] == 0,
    "db01_attempt_count"
]

print(
    "Ventanas normales con intentos a DB-01:",
    (normal_db_attempts > 0).sum()
)

suspicious_db_attempts = df.loc[
    df[TARGET] == 1,
    "db01_attempt_count"
]

print(
    "Ventanas sospechosas con intentos a DB-01:",
    (suspicious_db_attempts > 0).sum()
)

print("\nAuditoria finalizada.")
