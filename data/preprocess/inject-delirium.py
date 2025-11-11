from pathlib import Path
import pandas as pd
import numpy as np
import json

DATA_PROCESSED = Path("../../data_processed")
DATA_PROCESSED_DELIRIUM = Path("../../data_processed")
DATA_PROCESSED_DELIRIUM.mkdir(exist_ok=True)

# delirium window length in rows (based on 2s resampling)
MIN_WINDOW = 225
MAX_WINDOW = 450

# fraction of patients to convert
DELIRIUM_RATIO = 0.40  # for now

HR_PRIORITY = [
    "HR [bpm]",
    "HR",
    "hr",
    "HeartRate",
    "Heart Rate"
]

BTB_HR_COLS = ["btbHR [bpm]", "btbHR", "HR_bb", "ECG_RR_HR"]


def choose_windows(n_rows, rng, max_attempts=2000):
    """
    Produce a list of (start, end) windows that fit within n_rows.
    - If the series is too short (< MIN_WINDOW) returns [].
    - Caps the target number of windows based on available data.
    - Tries up to max_attempts to produce valid windows to avoid infinite loops.
    """
    windows = []
    if n_rows <= MIN_WINDOW:
        return windows

    # heuristic cap: roughly how many MIN_WINDOW-sized bursts might fit (allow some overlap)
    max_nonoverlap = max(1, n_rows // MIN_WINDOW)
    max_windows = max(1, max_nonoverlap * 3)
    target = int(rng.integers(1, min(250, max_windows) + 1))

    attempts = 0
    while len(windows) < target and attempts < max_attempts:
        # ensure window length is less than n_rows
        max_len_allowed = min(MAX_WINDOW, n_rows - 1)
        if max_len_allowed < MIN_WINDOW:
            break
        win_len = int(rng.integers(MIN_WINDOW, max_len_allowed + 1))
        # allow start up to n_rows
        start = int(rng.integers(0, n_rows - win_len + 1))
        end = start + win_len
        windows.append((start, end))
        attempts += 1

    return windows


def modify_window(df, start, end, rng):
    """
    Apply delirium-like modifications to rows [start, end) (end-exclusive).
    Uses label selection via df.index[start:end] so the selected row count equals (end - start).
    """
    if end <= start:
        return

    labels = df.index[start:end]
    length = end - start

    hr_col = None
    for col in HR_PRIORITY:
        if col in df.columns:
            hr_col = col
            break
    if hr_col is None:
        return  # no HR data, can't proceed

    # don't require BTB column to proceed; proceed with HR-based modifications
    # HR increase pattern
    baseline_shift = float(rng.normal(8, 3))
    drift = np.linspace(0, float(rng.normal(4, 1.5)), length)
    occasional_spikes = rng.choice([0, float(rng.normal(12, 5))], size=length, p=[0.9, 0.1])

    df.loc[labels, hr_col] += baseline_shift + drift + occasional_spikes

    # Strong variance + peak event increases
    if "accel_mean_dyn_2s" in df.columns:
        df.loc[labels, "accel_mean_dyn_2s"] *= rng.uniform(1.4, 1.8)

    if "accel_std_2s" in df.columns:
        df.loc[labels, "accel_std_2s"] *= rng.uniform(1.8, 3.2)  # spread out movement

    if "accel_peak_count_2s" in df.columns:
        df.loc[labels, "accel_peak_count_2s"] += int(rng.integers(8, 18)) #makes jerking motion (but not jorking it?)

    if "accel_max_2s" in df.columns:  # large burst
        spike_burst = rng.normal(0.15, 0.07, size=length)
        df.loc[labels, "accel_max_2s"] += np.abs(spike_burst)

    # emulating sitting up (or at least an attempt)
    if "gyro_mean_2s" in df.columns:
        df.loc[labels, "gyro_mean_2s"] *= rng.uniform(1.6, 2.2)

    if "gyro_std_2s" in df.columns:
        df.loc[labels, "gyro_std_2s"] *= rng.uniform(2.0, 3.0)

    if "gyro_large_change_count_2s" in df.columns:
        df.loc[labels, "gyro_large_change_count_2s"] += int(rng.integers(5, 14))


def inject_delirium():
    files = list(DATA_PROCESSED.glob("*_resampled.csv"))
    if not files:
        print("No files found in", DATA_PROCESSED)
        return

    num_delirium = int(len(files) * DELIRIUM_RATIO)
    if num_delirium == 0:
        print("DELIRIUM_RATIO results in 0 files to modify; exiting.")
        return

    delirium_files = np.random.choice(files, num_delirium, replace=False)

    for csv_path in delirium_files:
        df = pd.read_csv(csv_path).copy()

        rng = np.random.default_rng(abs(hash(csv_path.stem)) % (2**32))
        windows = choose_windows(len(df), rng)

        if not windows:
            print(f"Skipping {csv_path.name}: not enough rows ({len(df)}) to inject windows")
            continue

        for (start, end) in windows:
            modify_window(df, start, end, rng)

        out_csv = DATA_PROCESSED_DELIRIUM / csv_path.name
        df.to_csv(out_csv, index=False)

        json_path = out_csv.with_suffix(".json")
        windows_reformat = [[int(s), int(e)] for (s, e) in windows]
        with open(json_path, "w") as f:
            json.dump({"delirium_windows": windows_reformat}, f, indent=2)

        print(f"Injected delirium to {csv_path.name} with {len(windows)} windows")


if __name__ == "__main__":
    inject_delirium()