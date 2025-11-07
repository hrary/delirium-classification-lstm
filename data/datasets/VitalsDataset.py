from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from data.preprocess.extract_numerics import extract_numerics, resample_to_2s

class VitalsDataset(Dataset):
    def __init__(self):
        self.sequences = np.load('sequences.npy')
        self.labels = np.load('labels.npy')

    def __len__(self):
        return len(self.sequences)

    def __getitem(self, idx):
        return self.sequences[idx], self.labels[idx]