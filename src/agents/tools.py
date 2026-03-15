"""
Tools for predictions & news fetching.
Finnhub is ALWAYS used as the primary source.
Yahoo Finance is only a fallback if Finnhub fails.
"""
import os
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
FINNHUB_API_KEY = os.getenv("FMI_API_KEY")
FINNHUB_URL = "https://finnhub.io/api/v1/company-news"

# Optional fallback
try:
    from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool
except Exception:
    YahooFinanceNewsTool = None


# ---------------------------------------------------------
# PREDICTIONS
# ---------------------------------------------------------
# ---------------------------------------------------------
# PREDICTIONS
# ---------------------------------------------------------
def fetch_prediction_data(ticker: str):
    """
    Fetch raw prediction data (dict) or return error/training string.
    Returns:
       - dict: if successful (contains 'result')
       - str: if error or training ("__MODEL_TRAINING__")
    """
    try:
        r = requests.post(
            f"{API_BASE_URL}/predict-child",
            json={"ticker": ticker},
            timeout=30
        )

        # ---- FIX: detect training state ----
        if r.status_code == 202:
            return "__MODEL_TRAINING__"

        if r.status_code != 200:
            return f"Prediction error {r.status_code}: {r.text}"

        return r.json()
    except Exception as e:
        return f"Prediction fetch failed: {e}"


def get_stock_predictions(ticker: str) -> str:
    """Tool for Agents: Fetch predictions and return as formatted string."""
    data = fetch_prediction_data(ticker)
    
    if isinstance(data, str):
        return data
        
    try:
        forecast = (
            data.get("result", {})
            .get("predictions", {})
            .get("full_forecast", [])
        )

        if not forecast:
            return f"No predictions available for {ticker}"

        lines = [f"7-Day Price Forecast for {ticker}:"]
        for row in forecast[:7]:
            price = float(row.get("close", 0))
            lines.append(f"  {row['date']}: ${price:.2f}")

        return "\n".join(lines)

    except Exception as e:
        return f"Prediction parsing failed: {e}"


# ---------------------------------------------------------
# NEWS (FINNHUB ALWAYS PRIMARY)
# ---------------------------------------------------------
def get_stock_news(ticker: str) -> str:
    """
    Finnhub is ALWAYS used as the primary news source.
    Yahoo Finance is ONLY used if Finnhub fails.
    """
    # Try Finnhub
    try:
        end = datetime.utcnow().date()
        start = end - timedelta(days=7)

        r = requests.get(
            FINNHUB_URL,
            params={
                "symbol": ticker,
                "from": start,
                "to": end,
                "token": FINNHUB_API_KEY
            },
            timeout=30
        )

        if r.status_code == 200:
            articles = r.json()[:5]
            if articles:
                out = [f"Latest News for {ticker} (Finnhub):"]
                for a in articles:
                    dt = datetime.utcfromtimestamp(a.get("datetime", 0)).strftime("%Y-%m-%d")
                    out.append(
                        f"- {a.get('headline')} ({dt})\n"
                        f"  {a.get('summary','')}\n"
                        f"  {a.get('url','')}"
                    )
                return "\n".join(out)

        # If error or empty → fallback to Yahoo
        raise Exception("Finnhub returned empty / error")

    except Exception:
        # Fallback to Yahoo
        if not YahooFinanceNewsTool:
            return "News unavailable (Finnhub + Yahoo both unavailable)"

        try:
            news = YahooFinanceNewsTool().invoke(ticker)
            return f"Latest News for {ticker} (Yahoo):\n{news}"
        except Exception as e:
            return f"News fetch failed: {e}"


# LangChain tools
TOOLS_LIST = [get_stock_predictions, get_stock_news]
