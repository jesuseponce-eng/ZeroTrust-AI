
from pathlib import Path
import ipaddress
import pandas as pd


# ==========================================================
# 1 Project Settings
# ==========================================================

ROOT = Path(__file__).resolve().parents[2] 

RAW_DIR = ROOT / "data" / "raw" 
PROCESSED_DIR = ROOT / "data" / "processed" 

RAW_DIR.mkdir(parents=True, exist_ok=True) 
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================================
# 2 Simulated Events
# ==========================================================

events = []

def add_event( 
    event_id,
    timestamp,
    event_type,
    user_id,
    src_device,
    src_ip,
    dst_asset=None,
    dst_ip=None,
    dst_port=None,
    protocol=None,
    status="INFO",
    scenario_id="N-01",
    label=0
):
    events.append({ 
        "event_id": event_id,
        "timestamp": timestamp,
        "event_type": event_type,
        "user_id": user_id,
        "src_device": src_device,
        "src_ip": src_ip,
        "dst_asset": dst_asset,
        "dst_ip": dst_ip,
        "dst_port": dst_port,
        "protocol": protocol,
        "status": status,
        "scenario_id": scenario_id,
        "label": label
    })

# NORMAL SCENARIOS
# ----------------------------------------------------------

#Student uses the web portal successfully, no suspicious activity.
add_event(
    "EVT-N001",
    "2026-11-20 09:00:00",
    "auth_success",
    "USR-EST-01",
    "PC-EST-01",
    "10.10.10.11",
    status="SUCCESS",
    scenario_id="N-01",
    label=0
)

for i in range(3): 
    add_event(
        f"EVT-N{i+2:03d}",
        f"2026-11-20 09:0{i+1}:00",
        "network_connection",
        "USR-EST-01",
        "PC-EST-01",
        "10.10.10.11",
        "WEB-01",
        "10.10.40.10",
        443,
        "TCP",
        "ALLOWED",
        "N-01",
        0
    )

# Professor uses the research server successfully, no suspicious activity.
add_event(
    "EVT-N005",
    "2026-11-20 09:01:00",
    "network_connection",
    "USR-PROF-01",
    "PC-PROF-01",
    "10.10.10.12",
    "SRV-INV-01",
    "10.10.30.10",
    443,
    "TCP",
    "ALLOWED",
    "N-03",
    0
)

add_event(
    "EVT-N006",
    "2026-11-20 09:02:00",
    "network_connection",
    "USR-PROF-01",
    "PC-PROF-01",
    "10.10.10.12",
    "SRV-INV-01",
    "10.10.30.10",
    443,
    "TCP",
    "ALLOWED",
    "N-03",
    0
)

# SUSPICIOUS SCENARIOS
# ----------------------------------------------------------

# Multiple failed login attempts from the same student device, indicating a potential brute-force attack.
start_time = pd.Timestamp("2026-11-20 09:15:00")

for i in range(25):

    timestamp = start_time + pd.Timedelta(seconds=i * 8)

    add_event(
        f"EVT-A{i+1:03d}",
        str(timestamp),
        "auth_failure",
        "USR-EST-01",
        "PC-EST-01",
        "10.10.10.11",
        status="FAIL",
        scenario_id="A-01",
        label=1
    )

# Intent to connect to the database from the student device, which is unusual and potentially malicious.
add_event(
    "EVT-A026",
    "2026-11-20 09:18:30",
    "connection_attempt",
    "USR-EST-01",
    "PC-EST-01",
    "10.10.10.11",
    "DB-01",
    "10.10.50.10",
    5432,
    "TCP",
    "DENIED",
    "A-04",
    1
)


# ==========================================================
# 3 EVENTS TO DATAFRAME
# ==========================================================

df = pd.DataFrame(events)

print("\nEVENTOS GENERADOS:")
print(df.head())


# ==========================================================
# 4 NORMALIZATION AND VALIDATION
# ==========================================================

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    errors="raise"
)

df["event_type"] = (
    df["event_type"]
    .astype("string")
    .str.strip()
    .str.lower()
)

df["status"] = (
    df["status"]
    .astype("string")
    .str.strip()
    .str.upper()
)

df["protocol"] = (
    df["protocol"]
    .astype("string")
    .str.strip()
    .str.upper()
)


def normalize_ip(value):

    if pd.isna(value) or str(value).strip() == "":
        return pd.NA

    return str(ipaddress.ip_address(str(value).strip()))


df["src_ip"] = df["src_ip"].apply(normalize_ip)
df["dst_ip"] = df["dst_ip"].apply(normalize_ip)


df["dst_port"] = pd.to_numeric(
    df["dst_port"],
    errors="raise"
)

invalid_ports = (
    df["dst_port"].notna()
    & ~df["dst_port"].between(1, 65535)
)

if invalid_ports.any():
    raise ValueError("Se encontraron puertos inválidos.")


df["label"] = pd.to_numeric(
    df["label"],
    errors="raise"
).astype("int8")

if not df["label"].isin([0, 1]).all():
    raise ValueError("Las etiquetas deben ser 0 o 1.")


if df["event_id"].duplicated().any():
    raise ValueError("Existen identificadores de eventos duplicados.")


# ==========================================================
# 5 Save Normalized Events
# ==========================================================

raw_path = RAW_DIR / "events_normalized.csv"

df.to_csv( 
    raw_path,
    index=False
)

print("\nEventos normalizados guardados en:")
print(raw_path)


# ==========================================================
# 6 GENERATE DERIVED FEATURES
# ==========================================================

df["is_auth_failure"] = (
    df["event_type"] == "auth_failure"
).astype(int)

df["is_connection"] = (
    df["event_type"].isin([
        "network_connection",
        "connection_attempt"
    ])
).astype(int)

df["is_denied"] = (
    df["status"].isin([
        "DENIED",
        "BLOCKED"
    ])
).astype(int)

df["is_db_attempt"] = (
    df["dst_asset"] == "DB-01"
).astype(int)


df["window_start"] = (
    df["timestamp"].dt.floor("5min")
)


# ==========================================================
# 7 GROUP BY USER, DEVICE AND TIME WINDOW
# ==========================================================

features = ( 
    df.groupby([
        "src_device",
        "user_id",
        "window_start"
    ])
    .agg(
        failed_login_count=(
            "is_auth_failure",
            "sum"
        ),

        connection_count=(
            "is_connection",
            "sum"
        ),

        unique_destinations=(
            "dst_asset",
            "nunique"
        ),

        unique_ports=(
            "dst_port",
            "nunique"
        ),

        denied_count=(
            "is_denied",
            "sum"
        ),

        db01_attempt_count=(
            "is_db_attempt",
            "sum"
        ),

        label=(
            "label",
            "max"
        ),

        scenario_ids=(
            "scenario_id",
            lambda values: "|".join(
                sorted(set(values.dropna()))
            )
        )
    )
    .reset_index()
)


features["hour"] = (
    features["window_start"].dt.hour
)


# ==========================================================
# 8 SAVE FEATURES DATASET
# ==========================================================

features_path = (
    PROCESSED_DIR / "features_by_window.csv"
)

features.to_csv(
    features_path,
    index=False
)


# ==========================================================
# 9 SHOW RESULTS
# ==========================================================

print("\nDATASET DE CARACTERÍSTICAS:")
print(features.to_string(index=False))

print("\nRESUMEN DE ETIQUETAS:")
print(features["label"].value_counts())

print("\nARCHIVOS GENERADOS:")
print(raw_path)
print(features_path)
