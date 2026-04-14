import pandas as pd
import random
import torch
from torch.utils.data import Dataset
import config as config
from audio_normalization import normalize_single

class DrumDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        augment: bool = False,
        augment_labels: set[str] | None = None,
        augment_prob: float = 0.5,
    ):
        """
        Load dataset metadata from a split CSV file
        """

        self.df = pd.read_csv(csv_path)

        # lock label -> index mapping using config class order
        self.class_to_idx = {c: i for i, c in enumerate(config.classes)}
        self.augment = augment
        self.augment_labels = set(augment_labels or [])
        self.augment_prob = float(augment_prob)

    def __len__(self):
        """
        Return number of samples in the split
        """

        # dataset size drives epoch length
        return len(self.df)

    def __getitem__(self, idx):
        """
        Load and return a normalized waveform and label index
        """
        
        row = self.df.iloc[idx]
        label = row["label"]
        audio_path = str((config.project_root / str(row["path"])).resolve())

        # load and normalize audio so model always sees fixed shape
        x = normalize_single(audio_path)

        if self.augment and label in self.augment_labels and random.random() < self.augment_prob:
            x = self._augment_waveform(x)

        # convert string label into stable integer index
        y = self.class_to_idx[label]

        return x, torch.tensor(y, dtype = torch.long)

    def _augment_waveform(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply light, transient-safe waveform augmentation for confused classes
        """

        out = x.clone()

        # random gain jitter
        gain = random.uniform(0.85, 1.15)
        out = out * gain

        # temporal shift to diversify transients
        shift = random.randint(-480, 480)
        if shift != 0:
            out = torch.roll(out, shifts = shift, dims = -1)

        # add gaussian noise as regularization
        noise_std = random.uniform(0.0, 0.008)
        if noise_std > 0:
            out = out + torch.randn_like(out) * noise_std

        # keep waveform in a valid range after augmentation
        out = out.clamp(-1.0, 1.0)
        return out