import asyncio
import redis
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

# Import modules from backend
from backend.state import registry, REDIS_STATUS
from backend.tasks import refresh_system_metrics
import backend.state as app_state # to modify globals
from backend.api import router

from src.utils import setup_dagshub_mlflow, initialize_dirs
from logger.logger import get_logger

logger = get_logger()

# Setup
setup_dagshub_mlflow()
app = FastAPI(title="MLOps Stock Pipeline", version="3.1")

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(router)

@app.get("/metrics")
async def prometheus_metrics():
    refresh_system_metrics()
    return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

# Instrument
Instrumentator(registry=registry).instrument(app)

# Startup
@app.on_event("startup")
async def startup():
    initialize_dirs()
    
    # Retry logic for Redis connection
    for i in range(10):
        try:
            # We set the global in backend.state
            client = redis.Redis(host="redis", port=6379, db=0)
            client.ping()
            app_state.redis_client = client # Update the global
            REDIS_STATUS.set(1)
            logger.info("✅ Systems online (Redis, MLflow, Agents)")
            return
        except Exception as e:
            logger.warning(f"⏳ Waiting for Redis... attempt {i+1}/10")
            await asyncio.sleep(5)
            
    REDIS_STATUS.set(0)
    logger.error("❌ Failed to connect to Redis after multiple attempts.")

if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
