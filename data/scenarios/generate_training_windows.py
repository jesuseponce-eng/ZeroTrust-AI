
from pathlib import Path

import numpy as np
import pandas as pd


# ==========================================================
# 1 SETTINGS
# ==========================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

OUTPUT_PATH = (
    ROOT_DIR
    / "data"
    / "scenarios"
    / "synthetic_training_windows.csv"
)

SEED = 42
SAMPLES_PER_CLASS = 300

rng = np.random.default_rng(SEED)


# ==========================================================
# 2 AUXILIARY FUNCTIONS
# ==========================================================

def randint(low, high):
    return int(rng.integers(low, high + 1))


def unique_count(connection_count, maximum):
    if connection_count == 0:
        return 0

    return randint(
        1,
        min(connection_count, maximum)
    )


# ==========================================================
# 3 NORMAL SCENARIOS
# ==========================================================

def generate_normal_behavior():

    scenario = rng.choice(
        ["student", "professor", "maintenance"],
        p=[0.60, 0.25, 0.15]
    )

    if scenario == "student":

        failed_login_count = randint(0, 2)
        connection_count = randint(2, 12)

        denied_count = randint(0, 2)
        db01_attempt_count = 0

        max_destinations = 3
        max_ports = 4

    elif scenario == "professor":

        failed_login_count = randint(0, 2)
        connection_count = randint(5, 25)

        denied_count = randint(0, 2)
        db01_attempt_count = 0

        max_destinations = 5
        max_ports = 6

    else: 

        failed_login_count = randint(0, 2)
        connection_count = randint(25, 100)

        denied_count = randint(0, 4)
        db01_attempt_count = 0

        max_destinations = 8
        max_ports = 12

    return {
        "failed_login_count": failed_login_count,
        "connection_count": connection_count,
        "unique_destinations": unique_count(
            connection_count,
            max_destinations
        ),
        "unique_ports": unique_count(
            connection_count,
            max_ports
        ),
        "denied_count": denied_count,
        "db01_attempt_count": db01_attempt_count,
        "scenario_type": f"normal_{scenario}",
        "label": 0
    }


# ==========================================================
# 4 SUSPICIOUS SCENARIOS
# ==========================================================

def generate_suspicious_behavior():

    scenario = rng.choice(
        [
            "brute_force",
            "port_scan",
            "db_probe",
            "lateral_movement"
        ],
        p=[0.35, 0.25, 0.20, 0.20]
    )

    if scenario == "brute_force":

        failed_login_count = randint(6, 30)
        connection_count = randint(1, 10)

        denied_count = randint(5, 25)
        db01_attempt_count = 0

        max_destinations = 3
        max_ports = 3

    elif scenario == "port_scan":

        failed_login_count = randint(0, 5)
        connection_count = randint(15, 80)

        denied_count = randint(3, 25)
        db01_attempt_count = 0

        max_destinations = 16
        max_ports = 20

    elif scenario == "db_probe":

        failed_login_count = randint(0, 5)
        connection_count = randint(3, 25)

        denied_count = randint(1, 12)
        db01_attempt_count = randint(
            1,
            min(8, connection_count)
        )

        max_destinations = 5
        max_ports = 8

    else:

        failed_login_count = randint(0, 8)
        connection_count = randint(8, 40)

        denied_count = randint(0, 15)
        db01_attempt_count = randint(
            0,
            min(3, connection_count)
        )

        max_destinations = 12
        max_ports = 10

    return {
        "failed_login_count": failed_login_count,
        "connection_count": connection_count,
        "unique_destinations": unique_count(
            connection_count,
            max_destinations
        ),
        "unique_ports": unique_count(
            connection_count,
            max_ports
        ),
        "denied_count": denied_count,
        "db01_attempt_count": db01_attempt_count,
        "scenario_type": f"suspicious_{scenario}",
        "label": 1
    }


# ==========================================================
# 5 WINDOWS GENERATOR
# ==========================================================

records = []

base_time = pd.Timestamp("2026-11-01 08:00:00")

for i in range(SAMPLES_PER_CLASS):

    normal = generate_normal_behavior()
    suspicious = generate_suspicious_behavior()

    records.append({
        **normal,
        "scenario_ids": f"SCN-{2 * i:04d}",
        "src_device": f"PC-SIM-{2 * i:04d}",
        "user_id": f"USR-SIM-{2 * i:04d}",
        "window_start": (
            base_time + pd.Timedelta(minutes=5 * (2 * i))
        )
    })

    records.append({
        **suspicious,
        "scenario_ids": f"SCN-{2 * i + 1:04d}",
        "src_device": f"PC-SIM-{2 * i + 1:04d}",
        "user_id": f"USR-SIM-{2 * i + 1:04d}",
        "window_start": (
            base_time + pd.Timedelta(minutes=5 * (2 * i + 1))
        )
    })


# ==========================================================
# 6 BUILDING AND MIXING THE DATASET
# ==========================================================

df = pd.DataFrame(records)

df = df.sample(
    frac=1,
    random_state=SEED
).reset_index(drop=True)


# ==========================================================
# 7 VALIDATE RESULTS
# ==========================================================

feature_columns = [
    "failed_login_count",
    "connection_count",
    "unique_destinations",
    "unique_ports",
    "denied_count",
    "db01_attempt_count"
]

if df[feature_columns].isna().any().any():
    raise ValueError("Existen valores faltantes en las features.")

if not df["label"].isin([0, 1]).all():
    raise ValueError("Las etiquetas deben ser 0 o 1.")

if df["scenario_ids"].duplicated().any():
    raise ValueError("Existen escenarios duplicados.")


# ==========================================================
# 8 SAVE THE DATASET
# ==========================================================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUTPUT_PATH,
    index=False
)


# ==========================================================
# 9 SHOW RESUME 
# ==========================================================

print("\nDATASET GENERADO")

print("Total de ventanas:", len(df))

print("\nDistribución de etiquetas:")
print(df["label"].value_counts().sort_index())

print("\nDistribución de escenarios:")
print(df["scenario_type"].value_counts())

print(f"\nArchivo guardado en: {OUTPUT_PATH}")
