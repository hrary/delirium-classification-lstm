from pathlib import Path
from extract_numerics import extract_numerics, resample_to_2s
import os

print("cwd: " + os.getcwd())

'''
record_path = Path("../../data_raw/physionet.org/files/mimic4wdb/0.1.0/waves/p100/p10014354/81739927")
df = extract_numerics(record_path)
df_resampled = resample_to_2s(df)
df_resampled.to_csv("dataframe_sample.csv")
'''

DATA_RAW = Path("../../data_raw/physionet.org/files/mimic4wdb/0.1.0/waves")
DATA_PROCESSED = Path("../../data_processed")
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

all_possible_patient_dirs = list(DATA_RAW.rglob("*[0-9]"))

num_csv = 0
skip_if_exists = True

for patient_dir in all_possible_patient_dirs:
    try:
        if (DATA_PROCESSED / f"{str(patient_dir.name)}_resampled.csv").exists() and skip_if_exists:
            print("Skipping existing ", patient_dir.name)
            continue
        print("Processing ", patient_dir)
        df = extract_numerics(patient_dir)
        df_resampled = resample_to_2s(df)
        patient_id = str(patient_dir.name)
        df_resampled.to_csv(DATA_PROCESSED / f"{patient_id}_resampled.csv")
        num_csv += 1
    except:
        print("Skipping ", patient_dir)

print("Finished processing, made", num_csv, " files")