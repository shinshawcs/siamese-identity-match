from fastapi import FastAPI
from pydantic import BaseModel
import torch
import joblib
import numpy as np
from app.models.siamese_mlp import SiameseMLP
import os
import pandas as pd
app = FastAPI()

# 加载模型和 encoder
model_path = "checkpoints/siamese_model.pt"
encoder_path = "checkpoints/encoder.pkl"

encoder = joblib.load(encoder_path)
input_dim = encoder.transform([encoder.feature_names_in_]).shape[1]
model = SiameseMLP(input_dim=input_dim)
model.load_state_dict(torch.load(model_path))
model.eval()

class InputFeatures(BaseModel):
    a: dict
    b: dict
def fill_missing_keys(d, keys, default="unknown"):
    return {k: d.get(k, default) for k in keys}

@app.post("/predict")
def predict(data: InputFeatures):
    required_keys = encoder.feature_names_in_

    a_fixed = fill_missing_keys(data.a, required_keys)
    b_fixed = fill_missing_keys(data.b, required_keys)

    df_a = pd.DataFrame([a_fixed])
    df_b = pd.DataFrame([b_fixed])
    a = encoder.transform(df_a)
    b = encoder.transform(df_b)

    a = torch.tensor(a, dtype=torch.float32)
    b = torch.tensor(b, dtype=torch.float32)

    with torch.no_grad():
        out = model(a, b).item()

    return {"match_score": out}