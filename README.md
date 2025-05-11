# 🍊 Orange Disease Detection - MLOps Project

This project is a complete MLOps pipeline for detecting orange leaf diseases using a Convolutional Neural Network (CNN) model. It integrates MLflow for experiment tracking and model versioning, a Web UI built with Flask for user interaction, and uses Docker Compose to containerize the entire system. The solution is production-ready and deployable on AWS EC2.

---

## 📌 Features

- 🧠 **CNN-based Image Classification Model**
- 📦 **Dockerized** for scalable deployment
- 🚀 **MLflow** integration for experiment tracking
- 💽 **Flask Web UI** to upload images and get predictions
- 🔀 **Docker Compose** for local orchestration
- ☁️ **EC2-ready** architecture for cloud deployment

---

## 📂 Project Structure

```
.
├── frontend/                # Django Web UI for uploading and predicting
│   └── uploads/            # Folder for storing uploaded images
├── mlflow-data/            # Persistent MLflow data
├── models/                 # CNN model architecture
├── utils/                  # Data preprocessing, model utilities
├── scripts/train.py        # Training script using MLflow tracking
├── Dockerfile              # For the app (Web + Model)
└── README.md
```
![Image](https://github.com/user-attachments/assets/d986b4d1-790b-4da8-9766-fb7c12e2bdcf)
---

## 🐳 Docker Setup (Local)

### ✅ Prerequisites

- Docker & Docker Compose installed
- Basic knowledge of Python and Docker

### ⚠️ One-time setup

```bash
mkdir -p mlflow-data frontend/uploads
```

### 🔥 Build and Run

```bash
docker-compose up --build
```

### ✅ Services

- **Orange Disease Web App**: [http://localhost:5000](http://localhost:5000)
- **MLflow UI**: [http://localhost:8000](http://localhost:8000)

---

## ⚙️ MLflow Configuration

In the app container, MLflow is configured using:

```bash
MLFLOW_TRACKING_URI=http://mlflow:8000
```

MLflow is served from the `ghcr.io/mlflow/mlflow:latest` image and stores tracking data in:

- `./mlflow-data/` → Mounted to `/mlruns` in container
- SQLite DB (`mlflow.db`) used as backend store

---

## 📤 ECS Deployment Guide

1. **Push images to Docker Hub**

   ```bash
   docker tag orange-disease-detection:latest <your-dockerhub>/orange-disease-detection:latest
   docker push <your-dockerhub>/orange-disease-detection:latest
   ```

2. **Create an ECS Cluster**

3. **Define Task Definition with 2 containers**:
   - `orange-disease-app`
   - `mlflow`
   - Use same ports (5000 and 8000), and shared volume (EFS recommended for `/mlruns`)

4. **Create a Service** for the task

5. **Expose public access** to both containers (or use an ALB with path-based routing)

---

## 🧪 Model Training

Training script is located at:

```bash
scripts/train.py
```

It logs:
- Parameters
- Metrics
- Model artifacts
into MLflow automatically.

You can view runs at: [http://localhost:8000](http://localhost:8000)

---

## 🖼️ Prediction Flow

1. User uploads an image via Web UI
2. Image passed to backend model
3. Model returns prediction: Healthy / Disease
4. Result displayed on frontend


## 📜 License

This project is licensed under the MIT License.

![Image](https://github.com/user-attachments/assets/3906613f-c5f2-4703-8f4c-352c11f2673d)


