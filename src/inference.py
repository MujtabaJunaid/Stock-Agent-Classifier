import numpy as np
import pandas as pd
from typing import Dict
# import onnxruntime as ort
from sklearn.preprocessing import StandardScaler
from src.config import Config
from logger.logger import get_logger
from src.exception import PipelineError

logger = get_logger()



def predict_one_step_and_week(model, df: pd.DataFrame, scaler: StandardScaler, ticker: str) -> Dict:
    """
    Generate predictions for the next trading days using PyTorch model.
    Includes next-day, next-week, and full 5-day forecast.
    """
    try:
        cfg = Config()
        vals = scaler.transform(df[cfg.features]).astype("float32")
        # Shape: (1, context_len, input_size)
        X = vals[-cfg.context_len:].reshape(1, cfg.context_len, cfg.input_size)
        
        # Run inference
        import torch
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(cfg.device)
            preds = model(X_tensor).cpu().numpy()[0]  # shape: (pred_len, input_size)

        preds_inv = scaler.inverse_transform(preds.reshape(-1, cfg.input_size))[:, :5]

        # Prepare dates
        last_date = df["date"].iloc[-1]
        next_days = pd.bdate_range(last_date + pd.Timedelta(days=1), periods=cfg.pred_len)

        # Construct forecasts
        forecast = []
        for i, date in enumerate(next_days):
            forecast.append({
                "date": str(date.date()),
                "open": float(preds_inv[i][0]),
                "high": float(preds_inv[i][1]),
                "low": float(preds_inv[i][2]),
                "close": float(preds_inv[i][3]),
                "volume": float(preds_inv[i][4])
            })

        # Response structure
        return {
            "ticker": ticker,
            "last_date": str(last_date.date()),
            "future_window_days": cfg.pred_len,
            "next_business_days": [str(d.date()) for d in next_days],
            "predictions": {
                "next_day": forecast[0],
                "next_week": {
                    "high": float(np.max([d["high"] for d in forecast])),
                    "low": float(np.min([d["low"] for d in forecast]))
                },
                "full_forecast": forecast
            }
        }

    except Exception as e:
        logger.error(f"Prediction failed for {ticker}: {e}")
        raise PipelineError(f"Prediction failed for {ticker}: {e}")
