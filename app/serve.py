from fastapi import FastAPI
from pydantic import BaseModel
import torch
import joblib
import numpy as np
from app.models.siamese_mlp import SiameseMLP
import os
import pandas as pd
import warnings

import os
print("✅ Current working directory:", os.getcwd())
print("📂 Files in working dir:", os.listdir())

print("🔎 encoder exists:", os.path.exists("checkpoints/encoder.pkl"))
print("🔎 model exists:", os.path.exists("checkpoints/siamese_model.pt"))

app = FastAPI()

# 加载模型和 encoder
model_path = "checkpoints/siamese_model.pt"
encoder_path = "checkpoints/encoder.pkl"

try:
    encoder = joblib.load(encoder_path)
    required_columns = encoder.feature_names_in_.tolist()

    dummy_df = pd.DataFrame(columns=encoder.feature_names_in_)
    dummy_df.loc[0] = ["unknown"] * len(encoder.feature_names_in_)
    input_dim = encoder.transform(dummy_df).shape[1]

    model = SiameseMLP(input_dim=input_dim)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    print("✅ Model loaded successfully")
except Exception as e:
    print("❌ Failed to load model:", e)

class InputFeatures(BaseModel):
    a: dict
    b: dict
    
def fill_missing_keys(d, keys, default="unknown"):
        return {k: d.get(k, default) for k in keys}

@app.get("/")
def health_check():
    return {"status": "✅ Running", "version": "1.0.0"}

@app.post("/predict")
async def predict(data: InputFeatures):

    try:
        a_fixed = fill_missing_keys(data.a, required_columns)
        b_fixed = fill_missing_keys(data.b, required_columns)

        df_a = pd.DataFrame([a_fixed], columns=encoder.feature_names_in_)
        df_b = pd.DataFrame([b_fixed], columns=encoder.feature_names_in_)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a = encoder.transform(df_a)
            b = encoder.transform(df_b)

        a = torch.tensor(a, dtype=torch.float32)
        b = torch.tensor(b, dtype=torch.float32)

        with torch.no_grad():
            out = model(a, b).item()

        return {"match_score": out}
    except Exception as e:
        return {"error": str(e)}, 500
