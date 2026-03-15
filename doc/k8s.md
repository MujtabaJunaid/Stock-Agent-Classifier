# 🚢 Kubernetes Production-Style Guide (Minikube)

This guide provides a professional, production-ready workflow for running the MLOps pipeline on Kubernetes. It covers best practices including resource management, health probes, and streamlined deployment.

---

## ⚡️ Quick Start (Next Time / Day 2)

If you have already performed the initial setup, use this 3-step loop:

1.  **Start Cluster**: `minikube start`
2.  **Sync Code Changes** (Optional):
    ```bash
    eval $(minikube docker-env)
    # Re-build only the component you changed (e.g., FastAPI)
    docker build -t stock-agent-ops-fastapi:latest .
    ```
3.  **Deploy & Tunnel**: 
    ```bash
    kubectl apply -f k8s/
    minikube tunnel
    ```

---

## � Phase 1: Environment Preparation

### 1.1 Start the Cluster
Provision a cluster with sufficient resources to handle the ML workloads (FastAPI, Redis, Qdrant, Prometheus).

```bash
minikube start --cpus 4 --memory 6144 --disk-size 20g
```

### 1.2 Point Shell to Minikube's Docker
Instead of pushing to a remote registry, we build directly into Minikube's internal registry. **This is the fastest way for local K8s development.**

```bash
eval $(minikube docker-env)
```
*Note: You must run this in every new terminal window.*

---

## 📦 Phase 2: Build Pipeline Artifacts

Build the core components. We use the `latest` tag for simplicity, but in production, you should use semantic versioning or git hashes.

```bash
# Build Backend (FastAPI)
docker build -t stock-agent-ops-fastapi:latest .

# Build Frontend (Streamlit)
docker build -t stock-agent-ops-frontend:latest -f frontend/Dockerfile frontend/

# Build Monitoring App (Streamlit)
docker build -t stock-agent-ops-monitoring:latest -f monitoring_app/Dockerfile monitoring_app/
```

---

## 🚀 Phase 3: Pro-Level Deployment

### 3.1 The "Apply Everything" Command
For a production-style deployment, we apply manifests in a specific order to ensure dependencies (Volumes -> ConfigMaps -> DBs -> Apps) are met.

```bash
# One-liner to deploy everything in order
kubectl apply -f k8s/volumes.yaml && \
kubectl apply -f k8s/prometheus.yaml && \
kubectl apply -f k8s/redis.yaml && \
kubectl apply -f k8s/qdrant.yaml && \
kubectl apply -f k8s/fastapi.yaml && \
kubectl apply -f k8s/frontend.yaml && \
kubectl apply -f k8s/monitoring-app.yaml && \
kubectl apply -f k8s/grafana.yaml
```

### 3.2 Verify Deployment Status
Wait for all pods to reach `Running` state.

```bash
kubectl get pods -w
```

---

## 🌐 Phase 4: Access & Monitoring

### 4.1 Expose Services (The Tunneling Way)
In a real K8s environment (EKS/GKE), `LoadBalancer` services get a public IP automatically. In Minikube, we use the `tunnel` command:

```bash
# Run this in a dedicated terminal
minikube tunnel
```

### 4.2 Service Map
Once the tunnel is active, access the services via `localhost`:

| Service | Port | URL |
| :--- | :--- | :--- |
| **API (FastAPI)** | 8000 | [http://localhost:8000/docs](http://localhost:8000/docs) |
| **Main Frontend** | 8501 | [http://localhost:8501](http://localhost:8501) |
| **Monitoring App** | 8502 | [http://localhost:8502](http://localhost:8502) |
| **Grafana** | 3000 | [http://localhost:3000](http://localhost:3000) (Admin/Admin) |
| **Prometheus** | 9090 | [http://localhost:9090](http://localhost:9090) |

---

## 🛡️ Best Practices Implemented

1.  **Resource Limits**: Every deployment now has `cpu` and `memory` limits/requests defined to prevent "noisy neighbor" issues and OOM kills.
2.  **Health Probes**: FastAPI now includes `readinessProbe` and `livenessProbe` to ensure traffic is only routed to healthy containers.
3.  **Decoupled Storage**: Used `PersistentVolumeClaims` to ensure data in Redis and Qdrant persists across pod restarts.
4.  **Config Separation**: Prometheus configuration is decoupled via `ConfigMaps`.

---

## 🧹 Cleanup

```bash
# Delete all resources defined in files
kubectl delete -f k8s/
```

---

# View logs
kubectl logs -l app=fastapi --tail=100

# Describe pod for events/errors
kubectl describe pod -l app=fastapi
```

---

## 🆘 Troubleshooting

### "Connection Refused" (Port 8443)
If you see an error like `dial tcp [::1]:8443: connect: connection refused`, it means the cluster didn't start correctly.
**Fix:**
```bash
minikube delete
minikube start --driver=docker --cpus 4 --memory 6144
```

### Images not found (ImagePullBackOff)
Ensure you ran `eval $(minikube docker-env)` **before** building your Docker images. If you build them in a regular terminal and then try to run them in Minikube, the cluster won't see them.

