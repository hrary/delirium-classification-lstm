from pathlib import Path
import pandas as pd
import numpy as np

DATA_PROCESSED = Path("../../data_processed")

def simulate_movement(n):
    # Mostly still baseline movement
    accel_base = np.random.normal(0.015, 0.005, n)

    # Occasional motion bursts (turning/adjusting)
    spikes = np.random.binomial(1, 0.01, n) * np.random.uniform(0.1, 0.8, n)

    accel_mean = accel_base + spikes
    accel_std = accel_mean * np.random.uniform(0.05, 0.15, n)
    accel_max = accel_mean + accel_std * np.random.uniform(2, 4, n)

    # Peak counts: mostly 0-2, occasional 3-5 during movement bursts
    accel_peak_count = np.where(spikes > 0,
                                np.random.randint(1, 5, n),
                                np.random.randint(0, 2, n))

    # Gyro small baseline drift (very tiny orientation change)
    gyro_mean = np.random.normal(0.5, 0.15, n)

    # Gyro variability scales naturally
    gyro_std = np.abs(np.random.normal(0.2, 0.05, n)) * (1 + accel_mean * 4)

    # Large rotation events (turning over / sitting up) are rare
    gyro_large_change_count = np.random.binomial(1, 0.002, n)

    return {
        "accel_mean_dyn_2s": accel_mean,
        "accel_std_2s": accel_std,
        "accel_max_2s": accel_max,
        "accel_peak_count_2s": accel_peak_count,
        "gyro_mean_2s": gyro_mean,
        "gyro_std_2s": gyro_std,
        "gyro_large_change_count_2s": gyro_large_change_count,
    }

def simulate_temperature(n):
    base = 36.8
    drift = np.cumsum(np.random.normal(0, 0.0005, n))
    noise = np.random.normal(0, 0.05, n)
    return base + drift + noise

def fill_in(df):
    n = len(df)

    movement = simulate_movement(n)
    for key, val in movement.items():
        df[key] = val

    df["Temp"] = simulate_temperature(n)

    return df

def __main__():
    files = list(DATA_PROCESSED.glob("*_resampled.csv"))
    for f in files:
        try:
            df = pd.read_csv(f)
            df = fill_in(df)
            df.to_csv(f, index=False)
            print(f"Added movement + temperature → {f.name}")
        except Exception as e:
            print(f"Error on {f.name}: {e}")

if __name__ == "__main__":
    __main__()
