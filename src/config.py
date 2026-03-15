from dataclasses import dataclass, field
from typing import List
import torch

@dataclass
class Config:
    """Configuration for the stock prediction pipeline."""
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    context_len: int = 60
    pred_len: int = 5
    features: List[str] = field(default_factory=lambda: ["Open", "High", "Low", "Close", "Volume", "RSI14", "MACD"])
    batch_size: int = 32
    parent_ticker: str = "^GSPC"
    child_tickers: List[str] = field(default_factory=lambda: ["GOOG", "AMZN", "META", "TSLA"])
    start_date: str = "2004-08-19"  # Google's IPO date
    parent_epochs: int = 20
    child_epochs: int = 10
    transfer_strategy: str = "freeze"  # Options: "freeze", "fine_tune"
    fine_tune_lr: float = 1e-4
    parent_dir: str = "outputs/parent"
    fine_tune_lr: float = 1e-4
    parent_dir: str = "outputs/parent"
    workdir: str = "outputs"
    workdir: str = "outputs"

    @property
    def input_size(self) -> int:
        return len(self.features)