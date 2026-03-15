# Use an official Python runtime as a parent image
FROM python:3.13-slim

LABEL maintainer="Mujtaba Junaid <mujtabajunaid@github.com>"
LABEL project="MLOps Stock Analysis Pipeline"
LABEL description="FastAPI backend for ML inference and agent orchestration"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y build-essential git && rm -rf /var/lib/apt/lists/*

# Install uv for dependency management
RUN pip install uv

# Silence ONNXRuntime CPU vendor warnings
ENV ORT_LOGGING_LEVEL=ERROR
ENV SERVICE_NAME=mujtaba-mlops-backend
ENV OWNER=mujtaba-junaid-ops

# Copy project files
# We expect the build context to be the project root
COPY . /app

# Install dependencies via uv (system-wide)
RUN uv pip install --system -r backend/requirements.txt

# Expose the FastAPI port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Run the FastAPI app with Uvicorn
# Updated entry point
CMD ["python", "main.py"]
