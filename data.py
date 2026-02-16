import pandas as pd
import torch
from torch.utils.data import Dataset
import config as config
from audio_normalization import normalize_single

class DrumDataset(Dataset):
    def __init__(self, csv_path: str):
        """
        Load dataset metadata from a split CSV file
        """

        self.df = pd.read_csv(csv_path)

        # lock label -> index mapping using config class order
        self.class_to_idx = {c: i for i, c in enumerate(config.classes)}

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

        # load and normalize audio so model always sees fixed shape
        x = normalize_single(row["path"])

        # convert string label into stable integer index
        y = self.class_to_idx[row["label"]]

        return x, torch.tensor(y, dtype = torch.long)