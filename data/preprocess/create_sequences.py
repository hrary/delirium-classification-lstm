# python
from pathlib import Path
import pandas as pd
import numpy as np
import torch

# ---------- CONFIG ----------
DATA_PROCESSED = Path("../../data_processed")
DATA_SEQUENCES = Path("../../data_sequences")
SEQUENCE_LENGTH = 30      # number of timesteps per sequence (30 x 2s = 60s)
SEQUENCE_STEP = 30         # sliding step (1 -> maximum sequences) - adjust to reduce overlap
HR_BASELINE_WINDOW_SEC = 300
SAMPLE_PERIOD_S = 2
# ----------------------------

HR_PRIORITY = [
    "HR [bpm]",
    "HR",
    "hr",
    "HeartRate",
    "Heart Rate"
]

BTB_HR_COLS = ["btbHR [bpm]", "btbHR", "HR_bb", "ECG_RR_HR"]

OTHER_REQUIRED_FEATURE_SETS = {
    "SpO2": ["SpO2 [%]", "SpO2", "spo2"],
    "Temp": ["Temperature", "Temp", "temp"],
}


def compute_hrv_rmssd(hr_series_bpm):
    hr = np.asarray(hr_series_bpm, dtype=float)
    hr = hr[np.isfinite(hr) & (hr > 0)]
    if hr.size < 2:
        return np.nan
    rr_ms = 60000.0 / hr
    diff_rr = np.diff(rr_ms)
    rmssd = np.sqrt(np.mean(diff_rr ** 2))
    return float(rmssd)


def resolve_hr_columns(df):
    hr_col = None
    for c in HR_PRIORITY:
        if c in df.columns:
            hr_col = c
            break

    btb_col = None
    for c in BTB_HR_COLS:
        if c in df.columns:
            btb_col = c
            break

    return hr_col, btb_col


def resolve_feature_columns(df, required_sets):

    hr_col, btb_col = resolve_hr_columns(df)
    if hr_col is None:
        return None

    resolved = {"HR": hr_col}
    if btb_col is not None:
        resolved["BTB"] = btb_col

    for logical_name, possible in required_sets.items():
        found = None
        for col in possible:
            if col in df.columns:
                found = col
                break
        if logical_name == "SpO2" and found is None:
            return None
        if logical_name == "Temp" and found is None:
            resolved["Temp"] = None
        else:
            resolved[logical_name] = found

    return resolved


def create_sequences(df, resolved_cols, seq_length, step):
    keys = list(resolved_cols.keys())  # logical order
    df_sub = pd.DataFrame(index=df.index)

    for key in keys:
        actual_col = resolved_cols.get(key)
        if actual_col is None:
            # local filler, do not mutate original df
            if key == "Temp":
                df_sub[key] = 36.8
            else:
                df_sub[key] = np.nan
        else:
            df_sub[key] = df[actual_col]

    # drop rows missing any selected features
    df_sub = df_sub.dropna()
    if len(df_sub) < seq_length:
        return None

    sequences = []
    for start in range(0, len(df_sub) - seq_length + 1, step):
        window = df_sub.iloc[start:start + seq_length]
        hrv = compute_hrv_rmssd(window["HR"].values)
        arr = window.values.astype(np.float32)
        hrv_column = np.full((seq_length, 1), np.float32(hrv), dtype=np.float32)
        arr = np.hstack([arr, hrv_column])
        sequences.append(arr)

    if not sequences:
        return None

    return np.stack(sequences)


def save_patient_sequences(patient_id, sequences):
    DATA_SEQUENCES.mkdir(parents=True, exist_ok=True)
    out_path = DATA_SEQUENCES / f"{patient_id}.pt"
    torch.save({"sequences": sequences, "label": 0}, out_path)


def to_numeric_cols(df: pd.DataFrame, cols):
    for c in cols:
        if c is None or c not in df.columns:
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def process_all():
    DATA_SEQUENCES.mkdir(parents=True, exist_ok=True)
    all_csv_files = list(DATA_PROCESSED.glob("*_resampled.csv"))
    total = 0

    # build candidate numeric columns from known lists
    candidate_cols = set(HR_PRIORITY + BTB_HR_COLS)
    for vals in OTHER_REQUIRED_FEATURE_SETS.values():
        candidate_cols.update(vals)

    for csv_file in all_csv_files:
        try:
            df = pd.read_csv(csv_file)
            # coerce only likely numeric columns that exist in the dataframe
            to_numeric = [c for c in candidate_cols if c in df.columns]
            df = to_numeric_cols(df, to_numeric)

            resolved_cols = resolve_feature_columns(df, OTHER_REQUIRED_FEATURE_SETS)
            if resolved_cols is None:
                print(f"[SKIP] {csv_file.name} — missing required HR/SpO2")
                continue

            sequences = create_sequences(df, resolved_cols, SEQUENCE_LENGTH, SEQUENCE_STEP)
            if sequences is None:
                print(f"[SKIP] {csv_file.name} — no sequences produced")
                continue

            patient_id = csv_file.stem.replace("_resampled", "")
            save_patient_sequences(patient_id, sequences)
            total += sequences.shape[0]
            print(f"[OK] {csv_file.name} -> {sequences.shape[0]} sequences")
        except Exception as e:
            print(f"[ERROR] processing {csv_file.name}: {e}")

    print(f"Finished. Total sequences created: {total}")


if __name__ == "__main__":
    process_all()
