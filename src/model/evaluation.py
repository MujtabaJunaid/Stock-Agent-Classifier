import os
import json
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from typing import Dict
import mlflow
from src.config import Config
from src.data.ingestion import fetch_ohlcv
# from src.utils import save_json
# from src.saving import load_model
from logger.logger import get_logger
# import onnxruntime as ort
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

logger = get_logger()

def plot_predictions(Y: np.ndarray, preds: np.ndarray, ticker: str, save_path: str):
    """Plot Actual vs Predicted for the first 5 dimensions (OHLCV)."""
    plt.figure(figsize=(12, 8))
    features = ["Open", "High", "Low", "Close", "Volume"]
    for i, feature in enumerate(features):
        plt.subplot(3, 2, i + 1)
        plt.plot(Y[:, i], label="Actual", alpha=0.7)
        plt.plot(preds[:, i], label="Predicted", alpha=0.7)
        plt.title(f"{ticker} - {feature}")
        plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_residuals(Y: np.ndarray, preds: np.ndarray, ticker: str, save_path: str):
    """Plot Residuals (Actual - Predicted) for the first 5 dimensions (OHLCV)."""
    residuals = Y - preds
    plt.figure(figsize=(12, 8))
    features = ["Open", "High", "Low", "Close", "Volume"]
    for i, feature in enumerate(features):
        plt.subplot(3, 2, i + 1)
        plt.plot(residuals[:, i], label="Residuals", alpha=0.7)
        plt.axhline(0, color='r', linestyle='--')
        plt.title(f"{ticker} - {feature} Residuals")
        plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def evaluate_model_temp(model, df: pd.DataFrame, scaler: StandardScaler, temp_dir: str, ticker: str) -> Dict:
    """Evaluate model performance and save metrics directly to MLflow without local persistence."""
    try:
        config = Config()
        vals = scaler.transform(df[config.features]).astype("float32")
        X, Y = [], []
        for t in range(config.context_len, len(vals) - config.pred_len):
            past = vals[t - config.context_len:t]
            fut = vals[t:t + config.pred_len]
            if past.shape == (config.context_len, config.input_size) and fut.shape == (config.pred_len, config.input_size):
                X.append(past)
                Y.append(fut)
            else:
                logger.error(f"Skipping invalid evaluation sample at index {t}: past shape {past.shape}, fut shape {fut.shape}")

        if not X:
            logger.error(f"No valid samples for evaluation for {ticker}")
            return {}

        X, Y = np.array(X), np.array(Y)
        
        import torch
        with torch.no_grad():
            preds = []
            for x in X:
                x_tensor = torch.tensor(x.reshape(1, config.context_len, config.input_size), dtype=torch.float32).to(config.device)
                pred = model(x_tensor).cpu().numpy()[0]
                preds.append(pred)
        
        preds = np.array(preds)
        Y_ohlcv = Y.reshape(-1, config.input_size)[:, :5]
        preds_ohlcv = preds.reshape(-1, config.input_size)[:, :5]

        mse = mean_squared_error(Y_ohlcv, preds_ohlcv)
        rmse = np.sqrt(mse)
        r2 = r2_score(Y_ohlcv, preds_ohlcv)

        metrics = {"MSE": mse, "RMSE": rmse, "R2": r2}
        
        # Save metrics to temporary file and log to MLflow
        metrics_filename = f"{ticker}_metrics.json"
        metrics_path = os.path.join(temp_dir, metrics_filename)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"{ticker} → MSE: {mse:.5f}, RMSE: {rmse:.5f}, R²: {r2:.5f}")

        # Log metrics to MLflow
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(metrics_path, f"metrics/{ticker}")

        # Generate and log plots
        plot_filename = f"{ticker}_predictions.png"
        plot_path = os.path.join(temp_dir, plot_filename)
        plot_predictions(Y_ohlcv, preds_ohlcv, ticker, plot_path)
        mlflow.log_artifact(plot_path, f"plots/{ticker}")

        resid_filename = f"{ticker}_residuals.png"
        resid_path = os.path.join(temp_dir, resid_filename)
        plot_residuals(Y_ohlcv, preds_ohlcv, ticker, resid_path)
        mlflow.log_artifact(resid_path, f"plots/{ticker}")

        return metrics
    except Exception as e:
        logger.error(f"Evaluation failed for {ticker}: {e}")
        return {}

def evaluate_model(model, df: pd.DataFrame, scaler: StandardScaler, out_dir: str, ticker: str) -> Dict:
    """Evaluate model performance (Model Evaluation Stage) - Legacy function for backward compatibility."""
    try:
        os.makedirs(out_dir, exist_ok=True)
        config = Config()
        vals = scaler.transform(df[config.features]).astype("float32")
        X, Y = [], []
        for t in range(config.context_len, len(vals) - config.pred_len):
            past = vals[t - config.context_len:t]
            fut = vals[t:t + config.pred_len]
            if past.shape == (config.context_len, config.input_size) and fut.shape == (config.pred_len, config.input_size):
                X.append(past)
                Y.append(fut)
            else:
                logger.error(f"Skipping invalid evaluation sample at index {t}: past shape {past.shape}, fut shape {fut.shape}")

        if not X:
            logger.error(f"No valid samples for evaluation for {ticker}")
            return {}

        X, Y = np.array(X), np.array(Y)
        
        import torch
        with torch.no_grad():
            preds = []
            for x in X:
                x_tensor = torch.tensor(x.reshape(1, config.context_len, config.input_size), dtype=torch.float32).to(config.device)
                pred = model(x_tensor).cpu().numpy()[0]
                preds.append(pred)

        preds = np.array(preds)
        Y_ohlcv = Y.reshape(-1, config.input_size)[:, :5]
        preds_ohlcv = preds.reshape(-1, config.input_size)[:, :5]

        mse = mean_squared_error(Y_ohlcv, preds_ohlcv)
        rmse = np.sqrt(mse)
        r2 = r2_score(Y_ohlcv, preds_ohlcv)

        metrics = {"MSE": mse, "RMSE": rmse, "R2": r2}
        metrics_filename = f"{ticker}_parent_metrics.json" if "parent" in out_dir else f"{ticker}_child_metrics.json"
        metrics_path = os.path.join(out_dir, metrics_filename)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"{ticker} → MSE: {mse:.5f}, RMSE: {rmse:.5f}, R²: {r2:.5f}")

        # Log metrics to MLflow
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(metrics_path)

        # Generate and log plots
        plot_filename = f"{ticker}_predictions.png"
        plot_path = os.path.join(out_dir, plot_filename)
        plot_predictions(Y_ohlcv, preds_ohlcv, ticker, plot_path)
        mlflow.log_artifact(plot_path)

        resid_filename = f"{ticker}_residuals.png"
        resid_path = os.path.join(out_dir, resid_filename)
        plot_residuals(Y_ohlcv, preds_ohlcv, ticker, resid_path)
        mlflow.log_artifact(resid_path)

        return metrics
    except Exception as e:
        logger.error(f"Evaluation failed for {ticker}: {e}")
        return {}