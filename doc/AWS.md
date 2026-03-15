# AWS EKS Deployment Guide: Minimal & Cost-Effective

This guide covers deploying the MLOps pipeline to AWS EKS (Kubernetes) using a "Minimal Viable Infrastructure" approach. We will use **Terraform** for infrastructure, **GitHub Actions** for CI/CD, and **ECR** for images.

---

## 1. Prerequisites
Ensure you have these CLI tools installed:
*   `aws` (AWS CLI v2)
*   `kubectl` (Kubernetes CLI)
*   `terraform` (Infrastructure as Code)
*   `docker` (For local builds)

**Configure AWS:**
```bash
aws configure
# Enter Access Key, Secret Key, Region (e.g., us-east-1)
```

---

## 2. Infrastructure (Terraform vs. Eksctl)
*Question: Do I need Helm?*
**Answer**: No. For this project, standard Kubernetes manifests (`.yaml` files) are sufficient. Helm is great for packaging but adds complexity you don't need yet.

*Question: How to use Terraform?*
**Answer**: We will use a minimal Terraform configuration to spawn a VPC and EKS Cluster.

### A. Instance Selection (LSTM Project)
Since you are running LSTM (PyTorch), Redis, Qdrant, and FastAPI on the same cluster:
*   **Recommended**: `t3.xlarge` (4 vCPU, 16 GB RAM).
    *   *Why?* Sufficient memory for Qdrant/Redis + Model loading. Cost-effective.
*   **Alternative (Performance)**: `m5.large` or `g4dn.xlarge` (if GPU is mandatory, but expensive).
*   **Recommendation**: Start with **Run on CPU** (`t3.xlarge`) for the 1-2 hour demo to keep costs under $1.

### B. Minimal Terraform Setup
Create a file `infra/main.tf`:

```hcl
provider "aws" {
  region = "us-east-1"
}

module "vpc" {
  source = "terraform-aws-modules/vpc/aws"
  name = "mlops-vpc"
  cidr = "10.0.0.0/16"
  azs = ["us-east-1a", "us-east-1b"]
  public_subnets = ["10.0.1.0/24", "10.0.2.0/24"]
}

module "eks" {
  source          = "terraform-aws-modules/eks/aws"
  cluster_name    = "mlops-cluster"
  cluster_version = "1.27"
  vpc_id          = module.vpc.vpc_id
  subnet_ids      = module.vpc.public_subnets

  eks_managed_node_groups = {
    one = {
      min_size     = 1
      max_size     = 2
      desired_size = 1
      instance_types = ["t3.xlarge"]
      key_name       = "my-ssh-key" # Optional: ensure this key pair exists in AWS
    }
  }
}
```

**Run Infrastructure:**
```bash
cd infra
terraform init
terraform apply -auto-approve
# ⏳ Takes ~15 minutes
```

**Connect kubectl to EKS:**
```bash
aws eks update-kubeconfig --region us-east-1 --name mlops-cluster
```

---

## 3. Container Registry (ECR)
We need a place to store your Docker images.

1.  **Create Repository:**
    ```bash
    aws ecr create-repository --repository-name mlops-backend
    aws ecr create-repository --repository-name mlops-frontend
    ```
2.  **Login:**
    ```bash
    aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <AWS_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com
    ```

---

## 4. CI/CD: GitHub Actions
Automate building and pushing images upon code push.
Create `.github/workflows/deploy.yml`:

```yaml
name: Deploy to EKS

on:
  push:
    branches: [ "main" ]

jobs:
  build-and-push:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Configure AWS credentials
      uses: aws-actions/configure-aws-credentials@v1
      with:
        aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
        aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        aws-region: us-east-1

    - name: Login to Amazon ECR
      id: login-ecr
      uses: aws-actions/amazon-ecr-login@v1

    - name: Build and push Backend
      env:
        ECR_REGISTRY: ${{ steps.login-ecr.outputs.registry }}
        ECR_REPOSITORY: mlops-backend
        IMAGE_TAG: latest
      run: |
        docker build -t $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG .
        docker push $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG

    - name: Update KubeConfig
      run: aws eks update-kubeconfig --name mlops-cluster --region us-east-1

    - name: Deploy to EKS
      run: |
        kubectl apply -f k8s/
        kubectl rollout restart deployment/fastapi-deployment
```

---

## 5. Storage Strategy (S3 & Outputs)
*Question: Logic for outputs/ dir?*

In a container (`Pod`), files in `/app/outputs` are lost when the Pod restarts.
**Solution**: Use separate storage.

1.  **S3 (Recommended for Artifacts)**:
    Since you already use **MLflow (DagsHub)**, your models (`.pt`) and metrics should ideally go there. This is the cleanest MLOps approach.
    
2.  **S3 Sync (Minimal approach for 'outputs/')**:
    If you specifically want files in `outputs/` synced to a private S3 bucket:
    *   Create a bucket: `aws s3 mb s3://my-mlops-outputs`
    *   Give your Node Group IAM role permission to write to S3.
    *   Update `main.py` code to upload on completion:
        ```python
        import boto3
        s3 = boto3.client('s3')
        s3.upload_file("outputs/model.pt", "my-mlops-outputs", "model.pt")
        ```
    
3.  **Quick Fix (PVC)**:
    In your `k8s/deployment.yaml`, mount an AWS EBS volume to `/app/outputs`. This persists data as long as the Volume exists (even if Pod crashes).

---

## 6. Deploying Kubernetes Manifests
Run manually (or via GitHub Actions):

```bash
# 1. Apply Secrets (Env vars)
kubectl apply -f k8s/secrets.yaml

# 2. Deploy Services (Redis, Qdrant first)
kubectl apply -f k8s/redis-deployment.yaml
kubectl apply -f k8s/qdrant-deployment.yaml

# 3. Deploy App
kubectl apply -f k8s/fastapi-deployment.yaml
kubectl apply -f k8s/frontend-deployment.yaml
kubectl apply -f k8s/ingress.yaml
```

*Question: Networking?*
*   **Load Balancer**: When you apply a Service of `type: LoadBalancer` (or Ingress with ALB Controller), AWS automatically creates a Classic or Network Load Balancer. It costs ~$0.025/hour.
*   **Access**: You will get a DNS name (e.g., `a45...us-east-1.elb.amazonaws.com`) to access your Streamlit UI.

---

## 7. Cost Estimate (1-2 Hours)

If you run this setup for just 2 hours:

| Resource | Logic | Est. Cost |
| :--- | :--- | :--- |
| **EKS Control Plane** | $0.10/hr × 2 | $0.20 |
| **EC2 (t3.xlarge)** | $0.16/hr × 2 | $0.32 |
| **Load Balancer** | $0.025/hr × 2 | $0.05 |
| **EBS Volume** | 20GB ($0.08/GB/mo) / 720hrs * 2 | < $0.01 |
| **Data Transfer** | Minimal | < $0.01 |
| **TOTAL** | | **~ $0.60 USD** |

*Note: EKS clusters are billed pro-rata but have no startup fee.*

---

## 8. Pause, Destroy, Rebuild

**To Pause (Stop Billing for Compute):**
```bash
# Scale nodes to 0 (Stops EC2 cost, keeps EKS control plane cost)
kubectl scale deployment --all --replicas=0
# Update terraform node group min_size = 0
```
*Warning: EKS Control Plane ($0.10/hr) charges continue even if nodes are 0.*

**To Destroy (Stop ALL Billing):**
```bash
# Delete Services first (removes Load Balancers)
kubectl delete -f k8s/

# Destroy Infra
cd infra
terraform destroy -auto-approve
```

**To Rebuild:**
1.  `cd infra && terraform apply`
2.  `aws eks update-kubeconfig...`
3.  `kubectl apply -f k8s/`
