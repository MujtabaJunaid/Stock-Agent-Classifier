import os
import uuid
import io
import time
import requests
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime

# ----------------- CONFIG -----------------
API_URL = os.getenv("API_URL", "http://localhost:8000")
st.set_page_config(page_title="AI Financial Analyst", layout="wide", page_icon="📈")

# Custom CSS for Professional Look
st.markdown("""
    <style>
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    .metric-card {
        background-color: #1e2130;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #2e3140;
        text-align: center;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #4ade80;
    }
    .metric-label {
        font-size: 14px;
        color: #9ca3af;
    }
    </style>
    """, unsafe_allow_html=True)

# ----------------- SESSION ----------------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# ----------------- SIDEBAR ----------------
with st.sidebar:
    st.title("📈 AI Analyst")
    st.caption("Institutional-grade market intelligence")
    
    ticker = st.text_input("Ticker Symbol", value="NVDA", help="e.g., AAPL, TSLA, BTC-USD").strip().upper()
    run_btn = st.button("Generate Analysis", type="primary", use_container_width=True)
    
    st.markdown("---")
    st.info("Powered by LangGraph & Feast")
    st.markdown(f"Session: `{st.session_state.session_id[:8]}`")

# ----------------- HELPERS ----------------
def parse_predictions(preds_raw):
    """Robustly parse prediction data into forecast/history lists."""
    forecast, history = [], []
    
    if isinstance(preds_raw, list):
        # Direct list of dicts treatment
        for item in preds_raw:
            if isinstance(item, dict):
                # Heuristic: if it has 'close' and 'date'
                d = item.get("date") or item.get("dt")
                c = item.get("close") or item.get("price")
                if d and c:
                    forecast.append({"date": str(d), "close": float(c)})
        return forecast, history

    if isinstance(preds_raw, dict):
        # Dictionary format
        candidates = []
        # Try known keys
        for k in ["full_forecast", "forecast", "predictions", "data"]:
            if k in preds_raw and isinstance(preds_raw[k], list):
                candidates = preds_raw[k]
                break
        
        # If no key found, check values for any list
        if not candidates:
            for v in preds_raw.values():
                if isinstance(v, list):
                    candidates = v
                    break
        
        for it in candidates:
            if isinstance(it, dict):
                d = it.get("date") or it.get("dt")
                c = it.get("close") or it.get("price")
                if d and c:
                    forecast.append({"date": str(d), "close": float(c)})
                    
        # History extraction if available
        if "history" in preds_raw and isinstance(preds_raw["history"], list):
            for h in preds_raw["history"]:
                d = h.get("date")
                c = h.get("close")
                if d and c:
                    history.append({"date": str(d), "close": float(c)})
                    
        return forecast, history

    # String parsing (Fallback)
    if isinstance(preds_raw, str):
        lines = [l.strip() for l in preds_raw.splitlines() if l.strip()]
        for line in lines:
            if ":" in line and "$" in line:
                try:
                    left, right = line.rsplit(":", 1)
                    d_str = left.strip().split()[-1]
                    p_str = right.strip().replace("$", "").replace(",", "")
                    if d_str and p_str:
                         forecast.append({"date": d_str, "close": float(p_str)})
                except:
                    pass
        return forecast, history

    return forecast, history

def plot_chart(forecast, history, ticker):
    """Create a professional Plotly chart."""
    fig = go.Figure()

    # Historical
    if history:
        hist_df = pd.DataFrame(history)
        hist_df["date"] = pd.to_datetime(hist_df["date"])
        hist_df = hist_df.sort_values("date")
        
        fig.add_trace(go.Scatter(
            x=hist_df["date"], 
            y=hist_df["close"],
            mode="lines",
            name="History",
            line=dict(color="#3b82f6", width=2)
        ))
        
        # Connector
        if forecast:
            last_hist = hist_df.iloc[-1]
            last_hist_df = pd.DataFrame([last_hist])
            # Forecast
            pred_df = pd.DataFrame(forecast)
            pred_df["date"] = pd.to_datetime(pred_df["date"])
            pred_df = pred_df.sort_values("date")
            
            # Combine for smooth line
            combo_df = pd.concat([last_hist_df, pred_df], ignore_index=True)
            
            fig.add_trace(go.Scatter(
                x=combo_df["date"],
                y=combo_df["close"],
                mode="lines",
                name="Forecast",
                line=dict(color="#10b981", width=2, dash="dash")
            ))
    elif forecast:
        pred_df = pd.DataFrame(forecast)
        pred_df["date"] = pd.to_datetime(pred_df["date"])
        fig.add_trace(go.Scatter(
            x=pred_df["date"],
            y=pred_df["close"],
            name="Forecast",
            line=dict(color="#10b981", width=2, dash="dash")
        ))

    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis_title=None,
        yaxis_title="Price (USD)",
        hovermode="x unified",
        margin=dict(l=0, r=0, t=10, b=0),
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

# ----------------- MAIN LOGIC ----------------
if run_btn:
    st.session_state.active_ticker = ticker
    st.session_state.analysis_data = None
    st.session_state.error = None

if "active_ticker" in st.session_state and st.session_state.active_ticker:
    ticker = st.session_state.active_ticker
    
    if st.session_state.analysis_data is None:
        status_container = st.container()
        
        with status_container:
            with st.spinner(f"🔍 Analyzing {ticker}..."):
                try:
                    payload = {"ticker": ticker, "thread_id": st.session_state.session_id}
                    resp = requests.post(f"{API_URL}/analyze", json=payload, timeout=120)
                    
                    if resp.status_code == 200:
                        data = resp.json()
                        
                        # Case A: Need Training
                        if data.get("status") == "training":
                            st.warning(f"🏗️ Constructing Neural Model (First-time run)...")
                            prog_bar = st.progress(0)
                            status_text = st.empty()
                            
                            # Use task_id from response if available, else fallback to ticker
                            poll_id = data.get("task_id", ticker.lower())
                            
                            # Polling Loop
                            start_poll = time.time()
                            while True:
                                time.sleep(3)
                                try:
                                    st_resp = requests.get(f"{API_URL}/status/{poll_id}")
                                    if st_resp.status_code == 200:
                                        sdata = st_resp.json()
                                        status = sdata.get("status")
                                        elapsed = sdata.get("elapsed_seconds", 0)
                                        
                                        if status == "completed":
                                            prog_bar.progress(100)
                                            status_text.success("✅ Model Training Complete!")
                                            time.sleep(1) # Small delay for visual feedback
                                            st.rerun() # Refresh to start clean analysis
                                            break
                                        elif status == "failed":
                                            st.error(f"❌ Model Training Failed for {ticker}")
                                            st.session_state.active_ticker = None
                                            st.stop()
                                        else:
                                            # Progress heuristic (most stocks take 30-60s)
                                            prog = min(elapsed * 2, 95) 
                                            prog_bar.progress(int(prog))
                                            status_text.text(f"🚀 Training LSTM... {elapsed}s elapsed")
                                except Exception as e:
                                    status_text.warning(f"Waiting for updates... ({e})")
                        
                        # Case B: Error from Agent
                        if data.get("status") == "error":
                            st.error(f"❌ Analysis failed: {data.get('detail')}")
                            st.session_state.active_ticker = None
                            st.stop()

                        # Case C: Success
                        st.session_state.analysis_data = data
                    else:
                        st.error(f"Analysis Error ({resp.status_code}): {resp.text}")
                        st.session_state.active_ticker = None
                        st.stop()
                        
                except Exception as e:
                    st.error(f"Connection Failed: {e}")
                    st.session_state.active_ticker = None
                    st.stop()

    # If we have data, we proceed to display
    data = st.session_state.analysis_data
    if data:
        # ---- DASHBOARD DISPLAY ----
        
        # 1. Parse Data
        report = data.get("report") or data.get("final_report", "")
        rec = data.get("recommendation", "N/A")
        conf = data.get("confidence", "N/A")
        preds_raw = data.get("predictions", {})
        forecast, history = parse_predictions(preds_raw)
        
        # KPIs
        st.markdown(f"## 🏛️ Analysis: {ticker}")
        
        kpi1, kpi2, kpi3 = st.columns(3)
        
        latest_price = "N/A"
        if history:
            latest_price = f"${history[-1]['close']:.2f}"
        elif forecast:
             # Fallback to first forecast if no history
             latest_price = f"${forecast[0]['close']:.2f}"

        with kpi1:
            st.markdown(f"<div class='metric-card'><div class='metric-label'>Latest Price</div><div class='metric-value'>{latest_price}</div></div>", unsafe_allow_html=True)
        with kpi2:
            color = "#4ade80" if "BUY" in rec.upper() or "BULL" in rec.upper() else "#f87171"
            st.markdown(f"<div class='metric-card' style='border-color: {color};'><div class='metric-label'>Recommendation</div><div class='metric-value' style='color: {color}'>{rec}</div></div>", unsafe_allow_html=True)
        with kpi3:
            st.markdown(f"<div class='metric-card'><div class='metric-label'>Confidence</div><div class='metric-value'>{conf}</div></div>", unsafe_allow_html=True)
        
        st.markdown("---")
        
        # TABS
        tab_report, tab_chart = st.tabs(["📝 Strategic Report", "📊 Technical Forecast"])
        
        with tab_report:
            if report:
                st.markdown(report)
            else:
                st.warning("No report content generated.")
                
        with tab_chart:
            fig = plot_chart(forecast, history, ticker)
            st.plotly_chart(fig, use_container_width=True)

else:
    # Landing Page State
    st.markdown("### Ready to Analyze")
    st.markdown("""
    Enter a stock ticker in the sidebar to begin.
    
    **Features:**
    - 🧠 autonomous multi-agent research
    - 📈 LSTM technical price forecasting
    - 📰 Real-time news sentiment analysis
    """)