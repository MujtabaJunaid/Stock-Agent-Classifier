import os
import json
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from logger.logger import get_logger

logger = get_logger()

# =========================================================
# HELPER: Fetch Data
# =========================================================
def fetch_ohlcv(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch numeric data from YFinance."""
    try:
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if df.empty:
            return pd.DataFrame()
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        cols = ["Open", "High", "Low", "Close", "Volume"]
        cols = [c for c in cols if c in df.columns]
        return df[cols].dropna()
    except Exception as e:
        logger.error(f"Drift Fetch Error {ticker}: {e}")
        return pd.DataFrame()

# =========================================================
# CORE: Custom Drift Logic (No Dependencies)
# =========================================================
def calculate_custom_drift(ref_df: pd.DataFrame, curr_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Industry standard custom drift detection.
    - Data Drift: Z-score shift of feature means.
    - Model/Concept Drift: Volatility (StdDev) shift.
    """
    metrics = {}
    drift_scores = []
    
    for col in ref_df.columns:
        ref_mean = ref_df[col].mean()
        ref_std = ref_df[col].std()
        curr_mean = curr_df[col].mean()
        
        # 1. Z-Score Mean Shift (Data Drift)
        # How many historical standard deviations has the mean moved?
        shift = abs(curr_mean - ref_mean) / (ref_std + 1e-9)
        drift_scores.append(shift)
        metrics[col] = {
            "ref_mean": round(ref_mean, 2),
            "curr_mean": round(curr_mean, 2),
            "shift_score": round(shift, 4)
        }

    # Average drift score (normalized)
    # 0.0 - 1.0 is healthy, > 2.0 is drifting
    avg_drift = np.mean(drift_scores)
    
    # 2. Volatility Shift (Concept/Model Drift)
    # If market volatility changes significantly, the model's technical patterns may break.
    ref_vol = ref_df["Close"].pct_change().std()
    curr_vol = curr_df["Close"].pct_change().std()
    vol_ratio = (curr_vol / (ref_vol + 1e-9)) if ref_vol > 0 else 1.0
    
    # Health Mapping
    # Lower is better. 
    # Healthy: < 1.0
    # Degraded: 1.0 - 2.0
    # Critical: > 2.0
    
    status = "Healthy"
    if avg_drift > 2.0 or vol_ratio > 2.5 or vol_ratio < 0.4:
        status = "Critical (Drift Detected)"
    elif avg_drift > 1.0 or vol_ratio > 1.5 or vol_ratio < 0.6:
        status = "Degraded (Warning)"

    return {
        "health": status,
        "drift_score": round(avg_drift, 4),
        "volatility_index": round(vol_ratio, 4),
        "feature_metrics": metrics,
        "timestamp": datetime.now().isoformat()
    }

def check_drift(ticker: str, output_base: str) -> Dict[str, Any]:
    """
    Standard check drift function.
    No heavy dependencies (Evidently removed).
    """
    logger.info(f"📊 [Custom Drift] Checking {ticker}...")
    
    # 1. Setup Dates
    now = datetime.now()
    cutoff_current = now - timedelta(days=30)
    cutoff_reference = now - timedelta(days=180)
    
    # 2. Get Data
    ref_df = fetch_ohlcv(ticker, cutoff_reference.strftime("%Y-%m-%d"), cutoff_current.strftime("%Y-%m-%d"))
    curr_df = fetch_ohlcv(ticker, cutoff_current.strftime("%Y-%m-%d"), now.strftime("%Y-%m-%d"))
    
    if len(ref_df) < 20 or len(curr_df) < 3:
        return {"status": "skipped", "detail": "Insufficient data points"}
    
    # 3. Calculate
    try:
        drift_res = calculate_custom_drift(ref_df, curr_df)
        drift_res["status"] = "success"
        drift_res["ticker"] = ticker
        
        # 4. Save JSON
        drift_dir = os.path.join(output_base, ticker.lower(), "drift")
        os.makedirs(drift_dir, exist_ok=True)
        
        json_path = os.path.join(drift_dir, "latest_drift.json")
        with open(json_path, "w") as f:
            json.dump(drift_res, f, indent=2)
            
        # Also save a simple text report for logs
        logger.info(f"✅ [Drift] {ticker}: Status={drift_res['health']}, Score={drift_res['drift_score']}")
        
        return drift_res
    except Exception as e:
        logger.error(f"❌ Custom Drift Failed: {e}")
        return {"status": "failed", "error": str(e)}
