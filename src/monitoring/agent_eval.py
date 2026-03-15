import os
import json
import time
from datetime import datetime
from typing import Dict, Any

from src.agents.graph import analyze_stock
from logger.logger import get_logger

logger = get_logger()

class AgentEvaluator:
    def __init__(self, output_base: str):
        self.output_base = output_base

    def evaluate_live(self, ticker: str) -> Dict[str, Any]:
        """
        Runs the agent and evaluates the output using custom heuristics.
        No external dependencies (LangSmith/LangChain) required for evaluation.
        """
        logger.info(f"🤖 [Agent Eval] Evaluating {ticker}...")
        
        eval_dir = os.path.join(self.output_base, ticker.lower(), "agent_eval")
        os.makedirs(eval_dir, exist_ok=True)
        
        start_ts = time.time()
        
        # 1. Run Agent
        try:
            result = analyze_stock(ticker) 
            # Handle different return types (Dict or String)
            if isinstance(result, dict):
                text = result.get("final_report", str(result))
            else:
                text = str(result)
        except Exception as e:
            logger.error(f"❌ Agent Run Failed: {e}")
            return {"status": "failed", "error": str(e)}

        duration = time.time() - start_ts
        
        # 2. Custom Heuristic Evaluation
        metrics = self._heuristic_checks(ticker, text)
        metrics["duration_seconds"] = round(duration, 2)
        metrics["timestamp"] = datetime.now().isoformat()
        
        # 3. Save Result
        # We save as 'latest_eval.json' for easy retrieval and a timestamped one for history
        result_data = {
            "ticker": ticker,
            "metrics": metrics,
            "output_preview_text": text[:1000] # Save first 1000 chars
        }
        
        # History
        history_path = os.path.join(eval_dir, f"eval_{int(start_ts)}.json")
        with open(history_path, "w") as f:
            json.dump(result_data, f, indent=2)
            
        # Latest
        latest_path = os.path.join(eval_dir, "latest_eval.json")
        with open(latest_path, "w") as f:
            json.dump(result_data, f, indent=2)
            
        logger.info(f"✅ [Agent Eval] Score={metrics['overall_score']:.2f}")
        return result_data

    def _heuristic_checks(self, ticker: str, text: str) -> Dict[str, Any]:
        """Industry standard agent evaluation metrics."""
        text_lower = text.lower()
        
        # 1. Relevance: Does it talk about the requested ticker?
        has_ticker = ticker.lower() in text_lower
        
        # 2. Trustworthiness: Does it cite sources or mention specific data points?
        # Check for numeric data patterns like "$123.45" or "10%" or keywords like "source", "news", "report"
        has_citations = any(kw in text_lower for kw in ["source", "according to", "news", "report", "data"])
        has_metrics = any(char.isdigit() for char in text) # Contains numbers
        trustworthiness = has_citations and has_metrics
        
        # 3. Completeness: Check for key sections or length
        keywords = ["bullish", "bearish", "neutral", "buy", "sell", "hold"]
        has_rec = any(w in text_lower for w in keywords)
        
        # Calculation
        checks = {
            "relevance": has_ticker,
            "trustworthiness": trustworthiness,
            "has_recommendation": has_rec
        }
        
        score = sum(checks.values()) / len(checks) if checks else 0.0
        
        return {
            "checks": checks,
            "overall_score": round(score, 2),
            "status": "Trustworthy" if score > 0.6 else "Questionable"
        }
