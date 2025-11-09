import torch
import matplotlib.pyplot as plt

sample = torch.load("../../data_sequences/80057524.pt", weights_only=False)
print(sample.keys())

sequences = sample['sequences']
print(sequences.shape)

seq = sequences[19000]
plt.figure()
plt.plot(seq[:,0], label='HR')
plt.plot(seq[:,1], label='HR_dev')
plt.plot(seq[:,2], label='BTB_HR')
plt.plot(seq[:,3], label='HRV')
plt.plot(seq[:,4], label='SpO2')
plt.plot(seq[:,5], label='Temp')
plt.legend()
plt.grid(True)

plt.figure()
plt.plot(seq[:,6], label='accel_mean_dyn_2s')
plt.plot(seq[:,7], label='accel_std_2s')
plt.plot(seq[:,8], label='accel_max_2s')
plt.plot(seq[:,9], label='accel_peak_count_2s')
plt.plot(seq[:,10], label='gyro_mean_2s')
plt.plot(seq[:,11], label='gyro_std_2s')
plt.plot(seq[:,12], label='gyro_large_change_count_2s')
plt.legend()

plt.show()
