from torch.utils.data import Dataset
import torch
import numpy as np
from pathlib import Path
from typing import Any, Optional, Tuple

DATA_SEQUENCES = Path("../../data_sequences")

class VitalsDataset(Dataset):
    def __init__(
        self,
        data_dir: Optional[str] = None,
        transform=None,
        shuffle: bool = True,
        label_missing_value: int = -1,
    ):
        self.data_dir = Path(data_dir) if data_dir else DATA_SEQUENCES
        self.files = sorted(self.data_dir.glob("*.pt"))
        if shuffle:
            np.random.shuffle(self.files)

        self.samples = []  # list of (file_index, seq_index)
        self.data_cache = {}  # cache extracted (sequences, labels) per file
        self.transform = transform
        self.label_missing_value = label_missing_value

        for file_index, pt_file in enumerate(self.files):
            obj = torch.load(pt_file, map_location="cpu", weights_only=False)
            seqs, labels = self._extract(obj)
            if seqs is None:
                raise ValueError(f"Could not find sequences in {pt_file}")
            num_sequences = int(seqs.shape[0])
            for sequence_index in range(num_sequences):
                self.samples.append((file_index, sequence_index))
            # optionally cache metadata only (not heavy)
            self.data_cache[file_index] = (seqs, labels)

    def _extract(self, obj: Any) -> Tuple[Optional[torch.Tensor], Optional[Any]]:
        # dict with keys
        if isinstance(obj, dict):
            seq = None
            for k in ("sequences", "sequence", "x", "data", "inputs", "vals"):
                if k in obj:
                    seq = obj[k]
                    break
            lbl = None
            for k in ("labels", "label", "y", "target"):
                if k in obj:
                    lbl = obj[k]
                    break
            # convert numpy -> torch if needed
            if not torch.is_tensor(seq) and seq is not None:
                try:
                    seq = torch.as_tensor(seq)
                except Exception:
                    pass
            if lbl is not None and not torch.is_tensor(lbl):
                try:
                    lbl = torch.as_tensor(lbl)
                except Exception:
                    pass
            return seq, lbl

        # tuple/list (seqs, labels) or single object
        if isinstance(obj, (tuple, list)):
            if len(obj) == 2:
                seq, lbl = obj
            else:
                seq, lbl = obj[0], None
            if not torch.is_tensor(seq):
                try:
                    seq = torch.as_tensor(seq)
                except Exception:
                    pass
            if lbl is not None and not torch.is_tensor(lbl):
                try:
                    lbl = torch.as_tensor(lbl)
                except Exception:
                    pass
            return seq, lbl

        # raw tensor/ndarray -> sequences only
        if torch.is_tensor(obj):
            return obj, None
        try:
            import numpy as _np
            if isinstance(obj, _np.ndarray):
                return torch.from_numpy(obj), None
        except Exception:
            pass

        return None, None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_idx, seq_idx = self.samples[idx]

        seqs, labels = self.data_cache.get(file_idx, (None, None))
        if seqs is None:
            # fallback load if cache was cleared
            pt_file = self.files[file_idx]
            obj = torch.load(pt_file, map_location="cpu")
            seqs, labels = self._extract(obj)
            self.data_cache[file_idx] = (seqs, labels)

        x = seqs[seq_idx]
        y = labels[seq_idx] if labels is not None else self.label_missing_value

        # ensure tensor types
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32)
        else:
            x = x.float()
        if not torch.is_tensor(y):
            try:
                y = torch.tensor(int(y)).long()
            except Exception:
                y = torch.tensor(self.label_missing_value).long()
        else:
            y = y.long()

        if self.transform:
            x = self.transform(x)

        return x, y
