# ⚡ Quick Commands Reference

## 🚀 Setup & Run
- **Start Project**: `./run_docker.sh` (Builds & starts all services)
- **Restart Services**: `docker-compose restart`
- **View Logs**: `docker-compose logs -f`
- **Stop Project**: `docker-compose down`

## 🧠 Training & Prediction (API)
- **Train Parent (S&P 500)**: `curl -X POST http://localhost:8000/train-parent`
- **Train Child (AAPL)**: `curl -X POST http://localhost:8000/train-child -d '{"ticker": "AAPL"}'`
- **Predict Child**: `curl -X POST http://localhost:8000/predict-child -d '{"ticker": "AAPL"}'`
- **Check Task Status**: `curl http://localhost:8000/status/aapl`

## 🕵️ AI Agent
- **Full Analysis**: `curl -X POST http://localhost:8000/analyze -d '{"ticker": "AAPL"}'`

## 🛠️ System Maintenance
- **Health Check**: `curl http://localhost:8000/health`
- **List Outputs**: `curl http://localhost:8000/outputs`
- **Reset System (Wipe Data)**: `curl -X DELETE http://localhost:8000/system/reset`
- **View Metrics**: `curl http://localhost:8000/metrics`

## 🌐 URLs
- **FastAPI Docs**: http://localhost:8000/docs
- **Frontend UI**: http://localhost:8501
- **Monitoring UI**: http://localhost:8502
- **Redis Insight**: http://localhost:8001
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000
- **Qdrant**: http://localhost:6333

