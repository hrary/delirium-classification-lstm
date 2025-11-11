from pathlib import Path
import pandas as pd
import numpy as np
import torch
import json

# ---------- CONFIG ----------
DATA_PROCESSED = Path("../../data_processed")
DATA_SEQUENCES = Path("../../data_sequences")
DATA_PROCESSED_DELIRIUM = Path("../../data_processed_delirium")
SEQUENCE_LENGTH = 30
SEQUENCE_STEP = 30
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
    "accel_mean_dyn_2s": ["accel_mean_dyn_2s"],
    "accel_std_2s": ["accel_std_2s"],
    "accel_max_2s": ["accel_max_2s"],
    "accel_peak_count_2s": ["accel_peak_count_2s"],
    "gyro_mean_2s": ["gyro_mean_2s"],
    "gyro_std_2s": ["gyro_std_2s"],
    "gyro_large_change_count_2s": ["gyro_large_change_count_2s"],
}

FEATURE_ORDER = [
    "HR",
    "HR_dev",
    "BTB",
    "HRV",
    "SpO2",
    "Temp",
    "accel_mean_dyn_2s",
    "accel_std_2s",
    "accel_max_2s",
    "accel_peak_count_2s",
    "gyro_mean_2s",
    "gyro_std_2s",
    "gyro_large_change_count_2s",
]


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

    if btb_col is None:
        return None
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

def to_numeric_cols(df: pd.DataFrame, cols):
    for c in cols:
        if c is None or c not in df.columns:
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def create_sequences(df, resolved_cols, seq_length, step):
    keys = list(resolved_cols.keys())
    df_sub = pd.DataFrame(index=df.index)

    for key in keys:
        actual_col = resolved_cols.get(key)
        if actual_col is None:
            if key == "Temp":
                df_sub[key] = 36.8
            else:
                df_sub[key] = np.nan
        else:
            df_sub[key] = df[actual_col]

    df_sub = df_sub.dropna()
    if len(df_sub) < seq_length:
        return None, None

    mean_hr = float(df_sub["HR"].mean())
    hr_dev = df_sub["HR"] - mean_hr
    hr_idx = df_sub.columns.get_loc("HR")
    df_sub.insert(hr_idx + 1, "HR_dev", hr_dev.astype(np.float32))

    if "HRV" not in df_sub.columns:
        df_sub["HRV"] = np.nan
    ordered_cols = [c for c in FEATURE_ORDER if c in df_sub.columns]
    df_sub = df_sub.reindex(columns=ordered_cols)

    hrv_col_idx = df_sub.columns.get_loc("HRV")

    sequences = []
    intervals = []  # store (start_idx, end_idx) in original df index space
    for start in range(0, len(df_sub) - seq_length + 1, step):
        window = df_sub.iloc[start:start + seq_length]
        hrv = compute_hrv_rmssd(window["BTB"].values)
        arr = window.values.astype(np.float32)
        arr[:, hrv_col_idx] = np.float32(hrv)
        sequences.append(arr)
        # window.index are original df indices; convert to native ints
        intervals.append((int(window.index[0]), int(window.index[-1])))

    if not sequences:
        return None, None

    return np.stack(sequences), intervals


def save_patient_sequences(patient_id, sequences, labels):
    DATA_SEQUENCES.mkdir(parents=True, exist_ok=True)
    out_path = DATA_SEQUENCES / f"{patient_id}.pt"
    # labels as a numpy array of ints
    torch.save({"sequences": sequences, "labels": np.asarray(labels, dtype=np.int64)}, out_path)


def process_all(dir):
    import json
    DATA_SEQUENCES.mkdir(parents=True, exist_ok=True)
    all_csv_files = list(dir.glob("*_resampled.csv"))
    total = 0
    delirium_total = 0

    candidate_cols = set(HR_PRIORITY + BTB_HR_COLS)
    for vals in OTHER_REQUIRED_FEATURE_SETS.values():
        candidate_cols.update(vals)

    for csv_file in all_csv_files:
        try:
            df = pd.read_csv(csv_file)
            to_numeric = [c for c in candidate_cols if c in df.columns]
            df = to_numeric_cols(df, to_numeric)

            resolved_cols = resolve_feature_columns(df, OTHER_REQUIRED_FEATURE_SETS)
            if resolved_cols is None:
                print(f"[SKIP] {csv_file.name} — missing required HR/SpO2/BTB")
                continue

            sequences, intervals = create_sequences(df, resolved_cols, SEQUENCE_LENGTH, SEQUENCE_STEP)
            if sequences is None:
                print(f"[SKIP] {csv_file.name} — no sequences produced")
                continue

            # load delirium windows json if present
            delirium_json = DATA_PROCESSED / f"{csv_file.stem}.json"
            delirium_windows = []
            if delirium_json.exists():
                try:
                    with open(delirium_json, "r") as f:
                        data = json.load(f)
                        delirium_windows = data.get("delirium_windows", [])
                        # normalize types to native ints
                        delirium_windows = [(int(s), int(e)) for (s, e) in delirium_windows]
                except Exception:
                    delirium_windows = []

            # build labels per sequence by checking interval overlap
            labels = []
            for (s_idx, e_idx) in intervals:
                overlap = False
                for (w_s, w_e) in delirium_windows:
                    if not (e_idx < w_s or s_idx > w_e):
                        overlap = True
                        break
                labels.append(1 if overlap else 0)

            patient_id = csv_file.stem.replace("_resampled", "")
            save_patient_sequences(patient_id, sequences, labels)
            total += sequences.shape[0]
            delirium_total += int(sum(labels))
            print(f"[OK] {csv_file.name} -> {sequences.shape[0]} sequences (labels: {sum(labels)} delirium)")
        except Exception as e:
            print(f"[ERROR] processing {csv_file.name}: {e}")

    print(f"Finished. Total sequences created: {total} \nTotal delirium sequences: {delirium_total}")


if __name__ == "__main__":
    process_all(DATA_PROCESSED)
