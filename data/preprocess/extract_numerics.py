import gzip
import csv
import pandas as pd
from pathlib import Path
import wfdb

def extract_numerics(data_dir):
    data_dir = Path(data_dir)
    numeric_files = list(data_dir.glob('*n.csv.gz'))
    if not numeric_files:
        raise FileNotFoundError("No numeric CSV files found in the specified directory.")
    numeric_file = numeric_files[0]

    header_dir = str(data_dir / data_dir.name)
    header = wfdb.rdheader(str(header_dir))

    rows = []
    with gzip.open(str(numeric_file), 'rt') as gzf:
        reader = csv.DictReader(gzf)
        next(reader)
        for row in reader:
            t = float(row['time']) / header.counter_freq
            row["time_sec"] = t
            rows.append(row)

    df = pd.DataFrame(rows)
    df.set_index("time_sec", inplace=True)

    df = df.apply(pd.to_numeric, errors='coerce')

    return df

def resample_to_2s(df, duplicate_method='mean'):
    """
    Resample dataframe to 2 second intervals.

    duplicate_method: 'mean' (aggregate duplicates by mean), 'first' (keep first), or 'drop' (remove duplicates).
    """
    df = df.copy()

    # ensure index is numeric seconds
    df.index = pd.to_numeric(df.index, errors='coerce')
    df = df[~df.index.isna()]

    # sort by time
    df.sort_index(inplace=True)

    # handle duplicate timestamps
    if df.index.duplicated().any():
        if duplicate_method == 'first' or duplicate_method == 'drop':
            df = df[~df.index.duplicated(keep='first')]
        else:  # default 'mean'
            df = df.groupby(level=0).mean()

    # convert to TimedeltaIndex required by resample
    df.index = pd.to_timedelta(df.index, unit='s')

    # resample and interpolate
    df_resampled = df.resample('2s').interpolate()

    return df_resampled