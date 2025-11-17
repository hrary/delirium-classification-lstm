import time

from model_clone import ClassificationAlgorithm
import torch
import numpy as np
from pymongo import MongoClient

# CONFIG STUFF --------------------
client = MongoClient('mongodb://localhost:27017/')
db = client['delirium_detector']
assignments = db['assignments']
data = db['data']

limit = 32
long_term_limit = 128

model = ClassificationAlgorithm(input_dim=13, hidden_dim=30, num_layers=2, output_dim=1)
model.load_state_dict(torch.load('delirium_lstm_2.pth'))
model.eval()

sleep_time_seconds = 4
# ---------------------------------


def average(items: list[dict], key: str) -> float:
    total = 0
    for item in items:
        total += item[key]
    return total / len(items)

def compute_hrv_rmssd(hr_series_bpm):
    hr = np.asarray(hr_series_bpm, dtype=float)
    hr = hr[np.isfinite(hr) & (hr > 0)]
    if hr.size < 2:
        return np.nan
    rr_ms = 60000.0 / hr
    diff_rr = np.diff(rr_ms)
    rmssd = np.sqrt(np.mean(diff_rr ** 2))
    return float(rmssd)

"""
Feature order:
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
"gyro_large_change_count_2s"
"""

while True:
    try:
        for assignment in assignments.find():
            deviceId = assignment['deviceId']
            timestamp = assignment['timestamp']
            query = {
                'deviceId': deviceId,
                'timestamp': {'$gt': timestamp}
            }

            long_term_data = list(data.find(query).sort({'timestamp': -1}).limit(long_term_limit))

            patient_data_sequence = long_term_data[:limit]

            if len(patient_data_sequence) < limit:
                assignments.update_one({"deviceId": deviceId}, {"$set": {'status': "Not Enough Data"}})
                continue

            long_term_avg_hr = average(long_term_data, 'HR')

            for data_packet in patient_data_sequence:
                HR_dev = data_packet['HR'] - long_term_avg_hr
                HRV = compute_hrv_rmssd([d['HR'] for d in long_term_data])
                data_packet['HR_dev'] = HR_dev
                data_packet['HRV'] = HRV

            sequence = []
            patient_data_sequence = list(reversed(patient_data_sequence)) # LSTM expects oldest entries first

            for data_packet in patient_data_sequence:
                entry = [
                    data_packet['HR'],
                    data_packet["HR_dev"],
                    data_packet['BTB'],
                    data_packet["HRV"],
                    data_packet['SpO2'],
                    data_packet['Temp'],
                    data_packet['accel_mean_dyn_2s'],
                    data_packet['accel_std_2s'],
                    data_packet['accel_max_2s'],
                    data_packet['accel_peak_count_2s'],
                    data_packet['gyro_mean_2s'],
                    data_packet['gyro_std_2s'],
                    data_packet['gyro_large_change_count_2s']
                ]
                sequence.append(entry)

            input_tensor = torch.tensor([sequence], dtype=torch.float32)

            with torch.no_grad():
                output = model(input_tensor)

            status = "ok"

            if output.item() >= 0.7:
                status = "critical"
            elif output.item() >= 0.3:
                status = "warning"

            assignments.update_one({"deviceId": deviceId}, {"$set": {'status': status}})

        print("Finished analysis loop")
        time.sleep(sleep_time_seconds)
    except KeyboardInterrupt:
        print("Deployment loop interrupted by user.")
        break
    except Exception as e:
        print(f"Error in deployment loop: {e}")
        time.sleep(10)

