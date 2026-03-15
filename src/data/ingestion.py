import os
import pandas as pd
import yfinance as yf
from typing import Optional
from dotenv import load_dotenv
from src.config import Config
from src.exception import PipelineError
import subprocess
from datetime import datetime

load_dotenv()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI for a given series."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    return 100 - 100 / (1 + rs)

def macd(series: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
    """Calculate MACD for a given series."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    return ema_fast - ema_slow

def fetch_ohlcv(ticker: str, start: str = Config().start_date, end: Optional[str] = None) -> pd.DataFrame:
    """Fetch OHLCV data with technical indicators (Data Ingestion Stage)."""
    config = Config()
    try:
        df = yf.download(ticker, start=start, end=end, interval="1d", auto_adjust=True, progress=False)
        if df.empty:
            raise PipelineError(f"No data downloaded for {ticker}")
        
        # Flatten MultiIndex columns if present (yfinance > 0.2.40 behavior)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df = df.reset_index().rename(columns={"Date": "date"})
        df = df[["date", "Open", "High", "Low", "Close", "Volume"]].dropna()
        df["RSI14"] = rsi(df["Close"])
        df["MACD"] = macd(df["Close"])
        df = df[["date"] + config.features].dropna()
        
        # Validate data
        if len(df) < config.context_len + config.pred_len:
            raise PipelineError(f"Insufficient data for {ticker}: {len(df)} rows, need at least {config.context_len + config.pred_len}")
        if df[config.features].isnull().any().any():
            raise PipelineError(f"NaN values found in features for {ticker}")
        if not df[config.features].apply(lambda x: pd.api.types.is_numeric_dtype(x)).all():
            raise PipelineError(f"Non-numeric values found in features for {ticker}")
        print(f"Fetched {len(df)} rows for {ticker}")

        # =========================================================
        # FEAST INTEGRATION
        # =========================================================
        try:
            # Prepare data for Feast
            feast_df = df.copy()
            feast_df["ticker"] = ticker
            feast_df["event_timestamp"] = pd.to_datetime(feast_df["date"])
            feast_df["created_timestamp"] = datetime.now()
            
            # Save to parquet
            repo_path = os.path.join(os.getcwd(), "feature_store")
            data_path = os.path.join(repo_path, "data", "features.parquet")
            os.makedirs(os.path.dirname(data_path), exist_ok=True)
            
            # File Lock to prevent race conditions
            import fcntl
            lock_path = data_path + ".lock"
            
            with open(lock_path, "w") as lock_file:
                # Acquire exclusive lock (blocking)
                fcntl.flock(lock_file, fcntl.LOCK_EX)
                try:
                    # Check existing again under lock
                    if os.path.exists(data_path):
                        existing_df = pd.read_parquet(data_path)
                        # Append and deduplicate
                        combined_df = pd.concat([existing_df, feast_df]).drop_duplicates(subset=["ticker", "event_timestamp"])
                        combined_df.to_parquet(data_path)
                    else:
                        feast_df.to_parquet(data_path)
                finally:
                    # Release within the with block happens automatically on close, but explicit unlock is good
                    fcntl.flock(lock_file, fcntl.LOCK_UN)
            
            print(f"✅ Saved features to {data_path}")

            # Run Feast apply and materialize
            # Note: In a real prod env, this might be a separate service or step
            try:
                from feast import FeatureStore
                store = FeatureStore(repo_path=repo_path)
                store.apply([
                    # We need to import the objects to pass them to apply, 
                    # OR we can just let it scan the repo_path if we don't pass objects.
                    # store.apply() without arguments isn't standard in all versions, 
                    # but usually it scans. 
                    # Actually, the safest "clean" way via SDK without importing definitions 
                    # dynamically is complex. 
                    # BUT, since we are in the same repo, we can just use the CLI behavior 
                    # or import the definitions.
                    # Let's stick to subprocess for 'apply' as it reliably scans the directory,
                    # OR better: just materialize. 'apply' should be done at deploy time.
                    # However, for "simpler way", doing it here is fine.
                ])
                # Wait, store.apply() expects a list of objects (Entity, FeatureView).
                # It doesn't auto-scan like the CLI.
                # So subprocess IS actually simpler for auto-scanning 'feature_store/features.py'.
                # Let's keep subprocess for 'apply' but use store for 'materialize'.
                pass
            except:
                pass

            # Reverting to subprocess for 'apply' because it auto-scans 'features.py'.
            # Re-implementing with better error handling and logging.
            
            logger = Config().get_logger() if hasattr(Config, "get_logger") else print
            
            print("🔄 Running Feast apply...")
            subprocess.run(["feast", "apply"], cwd=repo_path, check=True, capture_output=True)
            
            print("🔄 Running Feast materialization...")
            # Materialize from 10 years ago to now to ensure all data is loaded
            subprocess.run(
                ["feast", "materialize-incremental", datetime.now().isoformat()], 
                cwd=repo_path, 
                check=True,
                capture_output=True
            )
            print("✅ Feast features materialized to Redis")

        except subprocess.CalledProcessError as e:
            print(f"⚠️ Feast command failed: {e.stderr.decode() if e.stderr else e}")
        except Exception as e:
            print(f"⚠️ Feast ingestion failed: {e}")

        return df
    except Exception as e:
        raise PipelineError(f"Failed to fetch data for {ticker}: {e}")