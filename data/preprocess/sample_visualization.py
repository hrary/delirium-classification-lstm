import torch
import matplotlib.pyplot as plt

sample = torch.load("../../data_sequences/80057524.pt", weights_only=False)

sequences = sample['sequences']
print(sequences.shape)

print(sequences[0])

seq = sequences[0]
plt.plot(seq[:,0], label='HR')
plt.plot(seq[:,1], label='SpO2')
plt.legend()
plt.show()
