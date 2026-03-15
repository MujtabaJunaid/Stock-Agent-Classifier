import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import mlflow
from src.config import Config
from logger.logger import get_logger

logger = get_logger()

def fit_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, epochs: int = 8, lr: float = 1e-3) -> nn.Module:
    """Train the LSTM model with early stopping (Model Training Stage)."""
    model.to(Config().device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=3)
    best_val_loss = float('inf')
    patience, counter = 5, 0

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for X, Y in train_loader:
            X, Y = X.to(Config().device), Y.to(Config().device)
            opt.zero_grad()
            pred = model(X)
            loss = criterion(pred, Y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {ep}/{epochs} - Train Loss: {avg_train_loss:.5f}")

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, Y in val_loader:
                X, Y = X.to(Config().device), Y.to(Config().device)
                pred = model(X)
                val_loss += criterion(pred, Y).item()
        avg_val_loss = val_loss / len(val_loader)
        logger.info(f"Epoch {ep}/{epochs} - Val Loss: {avg_val_loss:.5f}")

        # Log metrics to MLflow
        try:
            current_lr = opt.param_groups[0]['lr']
            mlflow.log_metric("train_loss", avg_train_loss, step=ep)
            mlflow.log_metric("val_loss", avg_val_loss, step=ep)
            mlflow.log_metric("learning_rate", current_lr, step=ep)
        except Exception as e:
            logger.warning(f"Failed to log metrics to MLflow: {e}")

        scheduler.step(avg_val_loss)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                logger.info("Early stopping triggered")
                break

    return model