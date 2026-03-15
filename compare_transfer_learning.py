import os
import time
import torch
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, random_split
from src.config import Config
from src.data.ingestion import fetch_ohlcv
from src.data.preparation import StockDataset
from src.model.definition import LSTMModel
from src.model.training import fit_model
from src.model.evaluation import evaluate_model_temp
from src.pipelines.training_pipeline import train_parent
from src.utils import setup_dagshub_mlflow

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_model_from_scratch(ticker: str, epochs: int):
    """
    Trains a model from random initialization (no transfer learning).
    """
    cfg = Config()
    logger.info(f"🚀 Training {ticker} from SCRATCH (Random Init)...")
    
    # 1. Prepare Data
    df = fetch_ohlcv(ticker, cfg.start_date)
    scaler = StandardScaler().fit(df[cfg.features])
    dataset = StockDataset(df, scaler)
    
    train_size = int(0.8 * len(dataset))
    train_ds, val_ds = random_split(
        dataset, [train_size, len(dataset) - train_size]
    )
    
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size)
    
    # 2. Init Model
    model = LSTMModel().to(cfg.device)
    
    # 3. Train
    start_time = time.time()
    model = fit_model(model, train_loader, val_loader, epochs=epochs, lr=1e-3)
    end_time = time.time()
    train_duration = end_time - start_time
    
    # 4. Evaluate
    scratch_dir = os.path.join(cfg.workdir, f"{ticker}_scratch")
    os.makedirs(scratch_dir, exist_ok=True)
    metrics = evaluate_model_temp(model, df, scaler, scratch_dir, ticker)
    
    metrics["Training Time (s)"] = train_duration
    return metrics

def train_model_with_transfer(ticker: str, epochs: int, parent_model_path: str):
    """
    Trains a model using weights from a parent model.
    """
    cfg = Config()
    logger.info(f"🚀 Training {ticker} with TRANSFER LEARNING (Parent Weights)...")
    
    # 1. Prepare Data
    df = fetch_ohlcv(ticker, cfg.start_date)
    scaler = StandardScaler().fit(df[cfg.features])
    dataset = StockDataset(df, scaler)
    
    train_size = int(0.8 * len(dataset))
    train_ds, val_ds = random_split(
        dataset, [train_size, len(dataset) - train_size]
    )
    
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size)
    
    # 2. Load Parent Model
    if not os.path.exists(parent_model_path):
        raise FileNotFoundError(f"Parent model not found at {parent_model_path}")
        
    model = LSTMModel().to(cfg.device)
    # Load weights
    try:
        model.load_state_dict(torch.load(parent_model_path, map_location=cfg.device))
        logger.info("✅ Loaded parent weights successfully")
    except Exception as e:
        logger.error(f"Failed to load parent weights: {e}")
        raise e

    # 3. Fine-tune
    # For transfer learning efficiency demo, we usually train for FEWER epochs or expect faster convergence
    # But to compare Apple-to-Apple time per epoch or total time for convergence, we track time here.
    start_time = time.time()
    model = fit_model(model, train_loader, val_loader, epochs=epochs, lr=1e-3)
    end_time = time.time()
    train_duration = end_time - start_time
    
    # 4. Evaluate
    transfer_dir = os.path.join(cfg.workdir, f"{ticker}_transfer")
    os.makedirs(transfer_dir, exist_ok=True)
    metrics = evaluate_model_temp(model, df, scaler, transfer_dir, ticker)
    
    metrics["Training Time (s)"] = train_duration
    return metrics

def main():
    cfg = Config()
    setup_dagshub_mlflow()
    
    # Use GOOG as the child ticker for this experiment
    child_ticker = "GOOG"
    
    print(f"\n============================================================")
    print(f"🧪 TRANSFER LEARNING COMPARISON: {child_ticker}")
    print(f"============================================================\n")
    
    # 1. Ensure Parent Model Exists
    parent_path = os.path.join(cfg.parent_dir, f"{cfg.parent_ticker}_parent_model.pt")
    if not os.path.exists(parent_path):
        print(f"⚠️ Parent model NOT found. Training {cfg.parent_ticker} first...")
        train_parent()
    else:
        print(f"✅ Parent Model Found: {parent_path}")
        
    # 2. Train with Transfer Learning
    print(f"\nrunning Scenario A: Transfer Learning...")
    try:
        tl_metrics = train_model_with_transfer(child_ticker, cfg.child_epochs, parent_path)
    except Exception as e:
        print(f"❌ Transfer Learning Failed: {e}")
        return

    # 3. Train from Scratch
    print(f"\nrunning Scenario B: Scratch Training...")
    try:
        scratch_metrics = train_model_from_scratch(child_ticker, cfg.child_epochs)
    except Exception as e:
        print(f"❌ Scratch Training Failed: {e}")
        return
        
    # 4. Show Results
    print(f"\n{'='*80}")
    print(f"📢 FINAL RESULTS: {child_ticker}")
    print(f"{'='*80}")
    print(f"{'Metric':<20} | {'scratch':<15} | {'transfer':<15} | {'Improvement':<25}")
    print(f"{'-'*80}")
    
    for key in tl_metrics.keys():
        s_val = scratch_metrics.get(key, 0.0)
        t_val = tl_metrics.get(key, 0.0)
        
        diff = s_val - t_val
        percent = 0.0
        if s_val != 0:
            percent = (diff / s_val) * 100
            
        # Interpretation
        is_error = key in ["MSE", "RMSE", "Training Time (s)"]
        
        if is_error:
            # Lower is better
            if t_val < s_val:
                # e.g. 100s -> 50s = 50% reduction
                if s_val != 0: percent = ((s_val - t_val) / abs(s_val)) * 100
                note = f"✅ Reduced by {percent:.1f}%"
            else:
                if s_val != 0: percent = ((t_val - s_val) / abs(s_val)) * 100
                note = f"❌ Increased by {percent:.1f}%"
        else:
            # Higher is better (R2)
            if t_val > s_val:
                 if s_val != 0: percent = ((t_val - s_val) / abs(s_val)) * 100
                 note = f"✅ Improved by {percent:.1f}%"
            else:
                 if s_val != 0: percent = ((s_val - t_val) / abs(s_val)) * 100
                 note = f"❌ Worse by {percent:.1f}%"
                 
        print(f"{key:<20} | {s_val:.5f}         | {t_val:.5f}         | {note}")
    
    print(f"{'='*80}\n")
    print("Interpretation:")
    print(" - Training Time: Lower implies less hardware usage and faster iteration.")
    print(" - MSE/RMSE: Lower implies better prediction accuracy.")
    print(" - R2: Higher implies better fit.")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
