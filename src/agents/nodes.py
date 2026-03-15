"""
Agent nodes for: 
- performance analysis
- market sentiment
- final Bloomberg-style report
"""
from datetime import datetime
import os
from langchain_core.messages import SystemMessage, AIMessage

from src.agents.tools import get_stock_predictions, get_stock_news, TOOLS_LIST
from logger.logger import get_logger

logger = get_logger()


# LLM setup (Ollama)
try:
    from langchain_ollama import ChatOllama

    llm = ChatOllama(
        model="gpt-oss:20b-cloud",
        temperature=0.3,
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ).bind_tools(TOOLS_LIST)

except Exception as e:
    print(f"⚠️ LLM Init Error: {e}")
    class Mock:
        def invoke(self, *args, **kwargs):
            return AIMessage(content=f"Mock response: LLM unavailable. Error: {e}")
    llm = Mock()


# --------------------------------------------------------------------------
# PERFORMANCE ANALYST
# --------------------------------------------------------------------------
def performance_analyst_node(state: dict) -> dict:
    ticker = state["ticker"]
    predictions = state.get("predictions")
    logger.info(f"📈 [Agent: Performance] Analyzing trends for {ticker}")

    if predictions == "__MODEL_TRAINING__":
        logger.warning(f"⚠️ [Agent: Performance] Model for {ticker} is still training.")
        return {
            "messages": [AIMessage(content=f"Model for {ticker} is currently training.")],
            "predictions": predictions
        }
    
    prompt = f"""
    You are a Performance Analyst. Analyze the 7-day price forecast for {ticker}.
    DATA:
    {predictions}
    
    Give a concise 2-3 line summary of the projected trend (Bullish/Bearish/Side-ways) and the price range.
    """
    resp = llm.invoke([SystemMessage(content=prompt)])
    content = resp.content if hasattr(resp, "content") else str(resp)
    logger.info(f"DEBUG: Perf Output: {content[:100]}...")
    
    return {
        "messages": [resp],
        "predictions": predictions
    }


# --------------------------------------------------------------------------
# MARKET EXPERT
# --------------------------------------------------------------------------
def market_expert_node(state: dict) -> dict:
    ticker = state["ticker"]
    logger.info(f"🗞️ [Agent: Market Expert] Fetching and summarizing news for {ticker}")
    news = get_stock_news(ticker)

    prompt = f"""
You are a market strategist summarizing sentiment.
News:

{news}

Return a 3–5 line sentiment summary.
"""
    resp = llm.invoke([SystemMessage(content=prompt)])
    content = resp.content if hasattr(resp, "content") else str(resp)
    logger.info(f"DEBUG: News Output: {content[:100]}...")

    return {
        "messages": [resp],
        "news_sentiment": news
    }


# --------------------------------------------------------------------------
# REPORT GENERATOR
# --------------------------------------------------------------------------
def report_generator_node(state: dict) -> dict:
    ticker = state["ticker"]
    logger.info(f"📝 [Agent: Report Gen] Assembling Bloomberg-style report for {ticker}")
    predictions = state.get("predictions", "")
    news = state.get("news_sentiment", "")

    prompt = f"""
Write a clean Bloomberg-style markdown report.

PREDICTIONS:
{predictions}

NEWS:
{news}

End with: **Market Stance:** BULLISH/BEARISH/NEUTRAL | **Confidence:** High/Medium/Low
"""
    resp = llm.invoke([SystemMessage(content=prompt)])
    text = resp.content if hasattr(resp, "content") else str(resp)
    logger.info(f"DEBUG: Report Gen Output: {text[:100]}...")

    # Extract stance
    upper = text.upper()
    stance = (
        "BULLISH" if "BULLISH" in upper else
        "BEARISH" if "BEARISH" in upper else
        "NEUTRAL"
    )

    # Confidence
    confidence = (
        "High" if "HIGH" in upper else
        "Low" if "LOW" in upper else
        "Medium"
    )

    return {
        "messages": [resp],
        "final_report": text,
        "recommendation": stance,
        "confidence": confidence
    }


# --------------------------------------------------------------------------
# CRITIC NODE
# --------------------------------------------------------------------------
def critic_node(state: dict) -> dict:
    """
    Criticizes and refines the report. 
    It checks for consistency between predictions and recommendation.
    """
    ticker = state.get("ticker", "N/A")
    logger.info(f"⚖️ [Agent: Critic] Reviewing and refining report for {ticker}")
    current_report = state.get("final_report", "")
    predictions = state.get("predictions", "")
    
    prompt = f"""
    You are a Senior Editor. critique and refine this financial report.
    
    DATA:
    {predictions}
    
    DRAFT REPORT:
    {current_report}
    
    Your Job:
    1. Verify if the 'Market Stance' aligns with the data.
    2. Ensure the tone is professional (Bloomberg style).
    3. If everything is good, just output the Original Report.
    4. If there are issues, rewrite it to be better.
    
    Output ONLY the Final Report (whether original or improved).
    """
    
    resp = llm.invoke([SystemMessage(content=prompt)])
    final_text = resp.content if hasattr(resp, "content") else str(resp)
    logger.info(f"DEBUG: Critic Output: {final_text[:100]}...")

    # We treat the critic's output as the definitive 'final_report'
    return {
        "messages": [resp],
        "final_report": final_text
    }
