
---
title: Siamese Match
emoji: 🤗
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 3.50.2
app_file: app/serve.py
pinned: false
---

# 🧬 Siamese MLP – Identity Matching via Deep Learning

This project implements a Siamese neural network for structured identity matching, inspired by real-world use cases like adtech identity resolution (e.g., data ingestion and linkage).

Built with:
- ✅ PyTorch (custom Siamese MLP model)
- ✅ Scikit-learn (OneHotEncoder)
- ✅ FastAPI (inference API)
- ✅ MLflow + Weights & Biases (training trace)
- ✅ Docker & Docker Compose
- ✅ Optional: Hugging Face Spaces deployment

---

## 🚀 Quick Start (Docker)

### 🔧 1. Build and Run

```bash
make build        # build Docker image
make up           # start FastAPI + MLflow
API available at: http://localhost:8000/docs
MLflow UI: http://localhost:5001

make test-api
payload example：
{
  "a": {"browser": "Chrome", "country": "US"},
  "b": {"browser": "Safari", "country": "US"}
}

🧠 Model Architecture
UserA ----> Encoder -----\
                          \
                           ---> Comparator --> [0,1] match score
                          /
UserB ----> Encoder -----/

•	Encoder: 2-layer MLP with ReLU, BatchNorm, Dropout
•	Comparator: Takes concat of [x1, x2, |x1-x2|, x1*x2]

🧪 MLflow & W&B Tracking
make train

Cleanup
make down     # stop containers
make clean    # delete checkpoints, mlruns



📁 Project Structure
├── app/
│   ├── serve.py              # FastAPI app
│   └── models/
│       └── siamese_mlp.py
├── train.py                  # PyTorch training script
├── data/user_pairs.csv       # Training data
├── checkpoints/              # Saved model + encoder
├── mlruns/                   # MLflow logs
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── Makefile
└── README.md