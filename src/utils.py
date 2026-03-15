import os
import json
import base64
from io import BytesIO
# import matplotlib.pyplot as plt
import pandas as pd
import mlflow
from typing import Dict
from src.config import Config
from src.exception import PipelineError
from logger.logger import get_logger
from dotenv import load_dotenv


logger = get_logger()

def setup_dagshub_mlflow():
    """Initialize DagsHub for remote MLflow tracking with authentication."""
    load_dotenv()
    
    dagshub_user = os.getenv("DAGSHUB_USER_NAME")
    dagshub_repo = os.getenv("DAGSHUB_REPO_NAME")
    dagshub_token = os.getenv("DAGSHUB_TOKEN")
    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    
    # Check if DagsHub credentials are provided
    if dagshub_user and dagshub_repo:
        try:
            import dagshub
            import dagshub.auth
            
            # Authenticate if token is present
            if dagshub_token:
                try:
                    dagshub.auth.add_app_token(dagshub_token)
                    logger.info("✓ Added DagsHub app token")
                except Exception as e:
                    if "File exists" in str(e):
                        logger.info("✓ DagsHub app token already exists")
                    else:
                        logger.warning(f"Failed to add DagsHub token: {e}")

            # Initialize DagsHub
            dagshub.init(repo_owner=dagshub_user, repo_name=dagshub_repo, mlflow=True)
            
            # Set MLflow tracking URI from .env
            if mlflow_tracking_uri:
                mlflow.set_tracking_uri(mlflow_tracking_uri)
                logger.info(f"✓ DagsHub MLflow tracking initialized: {mlflow_tracking_uri}")
            else:
                # Fallback to constructed URI
                dagshub_mlflow_uri = f"https://dagshub.com/{dagshub_user}/{dagshub_repo}.mlflow"
                mlflow.set_tracking_uri(dagshub_mlflow_uri)
                logger.info(f"✓ DagsHub MLflow tracking initialized: {dagshub_mlflow_uri}")
            
            # Ensure Model Registry URI points to the same backend
            try:
                registry_uri = mlflow.get_tracking_uri()
                mlflow.set_registry_uri(registry_uri)
                logger.info(f"✓ MLflow Model Registry initialized: {registry_uri}")
            except Exception as e:
                logger.warning(f"Failed setting MLflow registry URI: {e}")
            
            # Set authentication credentials for MLflow
            if dagshub_token:
                os.environ['MLFLOW_TRACKING_USERNAME'] = dagshub_user
                os.environ['MLFLOW_TRACKING_PASSWORD'] = dagshub_token
                logger.info("✓ DagsHub authentication configured")
            else:
                logger.warning("DAGSHUB_TOKEN not set - you may have read-only access")
            
            return True
        except ImportError:
            logger.warning("dagshub package not installed. Install with: pip install dagshub")
            logger.info("Please install: pip install dagshub")
        except Exception as e:
            logger.warning(f"Failed to initialize DagsHub: {e}")
    else:
        logger.warning("DAGSHUB_USER_NAME or DAGSHUB_REPO_NAME not set in .env file")
    
    return False

def initialize_dirs():
    """Initialize output directories."""
    config = Config()
    os.makedirs(config.parent_dir, exist_ok=True)
    # for ticker in config.child_tickers:
    #     os.makedirs(os.path.join(config.workdir, ticker), exist_ok=True)

def save_json(data: Dict, path: str) -> str:
    """Save dictionary to JSON file."""
    import json
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    return path