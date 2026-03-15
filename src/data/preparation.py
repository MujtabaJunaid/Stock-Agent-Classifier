import torch
from torch.utils.data import Dataset
from typing import Tuple
from sklearn.preprocessing import StandardScaler
from src.config import Config
from src.exception import PipelineError
import pandas as pd

class StockDataset(Dataset):
    """Dataset for stock price sequences (Data Preparation Stage)."""
    def __init__(self, df: pd.DataFrame, scaler: StandardScaler, context_len: int = Config().context_len, pred_len: int = Config().pred_len):
        self.context_len = context_len
        self.pred_len = pred_len
        try:
            vals = scaler.transform(df[Config().features]).astype("float32")
            self.samples = []
            for t in range(context_len, len(df) - pred_len):
                past = vals[t - context_len:t]
                fut = vals[t:t + pred_len]
                if past.shape == (context_len, len(Config().features)) and fut.shape == (pred_len, len(Config().features)):
                    self.samples.append((past, fut))
                else:
                    print(f"Skipping invalid sample at index {t}: past shape {past.shape}, fut shape {fut.shape}")
            if not self.samples:
                raise PipelineError("No valid samples created for dataset")
        except Exception as e:
            raise PipelineError(f"Failed to create dataset: {e}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        past, fut = self.samples[idx]
        return torch.tensor(past), torch.tensor(fut)