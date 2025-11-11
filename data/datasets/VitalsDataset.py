from torch.utils.data import Dataset
import torch
import numpy as np
from pathlib import Path

class VitalsDataset(Dataset):
    def __init__(self, data_dir=None, transform=None):
        base = Path(__file__).resolve().parent
        data_dir = Path(data_dir) if data_dir is not None else base
        self.sequences = np.load(data_dir / "sequences.npy")
        self.labels = np.load(data_dir / "labels.npy")
        self.transform = transform

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.sequences[idx]).float()
        y = torch.tensor(self.labels[idx]).long()
        if self.transform:
            x = self.transform(x)
        return x, y