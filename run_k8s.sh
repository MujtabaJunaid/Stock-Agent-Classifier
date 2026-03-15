#!/bin/bash

# Configuration
CPUS=4
MEMORY=6144
DISK="20g"

echo "Starting Minikube..."
minikube start --driver=docker --cpus $CPUS --memory $MEMORY --disk-size $DISK

echo "Building Docker Images (on HOST for speed)..."
# Build Backend
docker build -t stock-agent-ops-fastapi:latest -f backend/Dockerfile .
# Build Frontend
docker build -t stock-agent-ops-frontend:latest -f frontend/Dockerfile frontend/
# Build Monitoring
docker build -t stock-agent-ops-monitoring:latest -f monitoring_app/Dockerfile monitoring_app/

echo "📥 Loading images into Minikube..."
minikube image load stock-agent-ops-fastapi:latest
minikube image load stock-agent-ops-frontend:latest
minikube image load stock-agent-ops-monitoring:latest

echo "🚢 Deploying to Kubernetes..."
kubectl apply -f k8s/

echo "⏳ Waiting for pods to be ready..."
# Increase timeout for heavy ML app
kubectl wait --for=condition=ready pod -l app=fastapi --timeout=300s

echo "🌐 DONE!"
echo "--------------------------------------------------"
echo "IMPORTANT: Run 'sudo minikube tunnel' in a NEW terminal"
echo "to access the apps at localhost:"
echo "- FastAPI: http://localhost:8000/docs"
echo "- Main UI: http://localhost:8501"
echo "- Monitoring: http://localhost:8502"
echo "--------------------------------------------------"
