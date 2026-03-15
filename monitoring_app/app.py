import streamlit as st
import os
import json
import pandas as pd
import requests
from datetime import datetime
import streamlit.components.v1 as components

# Configuration
OUTPUTS_DIR = "/app/outputs"
API_URL = os.getenv("API_URL", "http://fastapi:8000")

st.set_page_config(
    page_title="MLOps Monitoring",
    page_icon="📊",
    layout="wide"
)

# Custom Styling for "Minimal" feel
st.markdown("""
<style>
    .metric-container { background-color: #1a1c24; padding: 15px; border-radius: 10px; border: 1px solid #30363d; margin-bottom: 10px;}
    .status-healthy { color: #238636; font-weight: bold; }
    .status-warning { color: #d29922; font-weight: bold; }
    .status-error { color: #f85149; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.title("📊 Monitoring Dashboard")

# Sidebar
with st.sidebar:
    st.header("Diagnostic Tools")
    
    # 1. Parent Training & Health
    st.subheader("🛠️ Parent Management")
    if st.button("🏗️ Master Train Parent (^GSPC)", use_container_width=True, help="Priming the root model (requires ~1-2 mins)"):
        try:
            t_resp = requests.post(f"{API_URL}/train-parent")
            if t_resp.status_code == 200:
                st.success("Training Started! Task: parent_training")
            else:
                st.error(f"Error: {t_resp.text}")
        except:
            st.error("API Offline")

    if st.button("🏥 Run Parent Health Audit", use_container_width=True):
        st.session_state.current_ticker = "^GSPC"
        st.session_state.do_monitor = True

    st.markdown("---")
    
    # 2. Child Ticker Input
    user_ticker = st.text_input("Analyze Child Ticker", placeholder="AAPL, NVDA...").strip().upper()
    if st.button("🔍 Run Quality Check", use_container_width=True):
        if user_ticker:
            st.session_state.current_ticker = user_ticker
            st.session_state.do_monitor = True
        else:
            st.error("Input ticker first")

    st.markdown("---")
    st.markdown("---")
    # 3. Enhanced Logs
    st.subheader("📝 System Logs")
    num_lines = st.slider("Lines", 10, 200, 50)
    if st.button("🔄 Refresh Logs", use_container_width=True):
        try:
            l_resp = requests.get(f"{API_URL}/system/logs?lines={num_lines}")
            if l_resp.status_code == 200:
                log_data = l_resp.json()
                st.session_state.logs = log_data.get("logs", "N/A")
                st.session_state.log_file = log_data.get("filename", "N/A")
        except:
            st.error("API Offline")
    
    if "logs" in st.session_state:
        st.caption(f"File: {st.session_state.get('log_file')}")
        st.code(st.session_state.logs, language="text")

# Main Content
if "current_ticker" in st.session_state:
    ticker = st.session_state.current_ticker
    
    if st.session_state.get("do_monitor"):
        with st.spinner(f"Running Diagnostics for {ticker}..."):
            try:
                requests.post(f"{API_URL}/monitor/{ticker}")
                st.session_state.do_monitor = False # Success
            except Exception as e:
                st.error(f"API Error: {e}")

    st.divider()
    st.header(f"Results for {ticker}")
    
    # Grid for Output
    if ticker == "^GSPC":
        col_a, col_b = st.columns([1, 1])
    else:
        col_a = st.container() # Just a container for agent eval
        col_b = None
    
    path_base = os.path.join(OUTPUTS_DIR, ticker.lower())
    
    with col_a:
        st.subheader("🤖 Agent Integrity")
        
        # Check if we just ran a monitor and we have it in memory, or if it's on disk
        eval_json = os.path.join(path_base, "agent_eval", "latest_eval.json")
        if os.path.exists(eval_json):
            with open(eval_json, "r") as f:
                d = json.load(f)
                m = d.get("metrics", {})
                
                # Big Score
                score = m.get('overall_score', 0)
                status_label = m.get("status", "N/A")
                st.metric("Trust Score", f"{score*100:.0f}%", delta=status_label)
                
                # Check Details
                st.write("**Assessment Details:**")
                checks = m.get("checks", {})
                if not checks:
                    st.warning("No heuristic checks performed. Agent might be in training or error state.")
                else:
                    cols = st.columns(len(checks))
                    for i, (k, v) in enumerate(checks.items()):
                        with cols[i]:
                            st.markdown(f"{'✅' if v else '❌'}\n\n**{k.replace('_',' ').title()}**")
                
                st.markdown("---")
                with st.expander("🔍 View Final Report Analysis"):
                    report_text = d.get("output_preview_text", "No preview available")
                    if "status" in report_text and "training" in report_text:
                         st.warning("⚠️ Agent reported model training in progress. Evaluation is a placeholder.")
                    st.markdown(report_text)
        else:
            st.info("No audit logs found for this ticker. Run 'Quality Check' to initiate.")

    if col_b:
        with col_b:
            st.subheader("📉 Data Drift Status")
            drift_json = os.path.join(path_base, "drift", "latest_drift.json")
            if os.path.exists(drift_json):
                with open(drift_json, "r") as f:
                    dj = json.load(f)
                
                # Display Health and Score
                health = dj.get("health", "Unknown")
                score = dj.get("drift_score", 0)
                
                color_class = "status-healthy" if "Healthy" in health else "status-warning" if "Degraded" in health else "status-error"
                st.markdown(f"**Model Health:** <span class='{color_class}'>{health}</span>", unsafe_allow_html=True)
                st.metric("Drift Score", f"{score:.2f}")
                st.metric("Volatility Index", f"{dj.get('volatility_index', 0):.2f}")
                
                # Feature Detail
                with st.expander("Feature Drift Breakdown"):
                    feat_data = dj.get("feature_metrics", {})
                    df_feats = pd.DataFrame.from_dict(feat_data, orient='index')
                    st.table(df_feats)

                # Interactive Button
                drift_dir = os.path.join(path_base, "drift")
                htmls = [f for f in os.listdir(drift_dir) if f.endswith(".html")]
                if htmls:
                    if st.button("Open Full Drift Report", use_container_width=True):
                        with open(os.path.join(drift_dir, sorted(htmls)[-1]), 'r') as f:
                            components.html(f.read(), height=800, scrolling=True)
            else:
                st.warning("No drift analysis found. Click Parent Health to generate.")

else:
    # Minimal Landing
    st.markdown("### 🔍 Select a diagnostic tool to start.")
    st.write("Use the sidebar check Parent Model Health or evaluate a specific stock agent.")
    
    if os.path.exists(OUTPUTS_DIR):
        recent = [d.upper() for d in os.listdir(OUTPUTS_DIR) if os.path.isdir(os.path.join(OUTPUTS_DIR, d)) and not d.startswith(".")]
        if recent:
            st.markdown("---")
            st.write("**Recent Activities:**")
            st.caption(", ".join(sorted(recent)))

st.sidebar.markdown("---")
st.sidebar.info("Syncing with FastAPI @ " + API_URL)
