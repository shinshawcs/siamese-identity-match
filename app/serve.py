from fastapi import FastAPI
from pydantic import BaseModel
import torch
import joblib
import numpy as np
from app.models.siamese_mlp import SiameseMLP
from app.models.siamese_mlp_v2 import SiameseMLP_v2
import os
import pandas as pd
import warnings
from prometheus_fastapi_instrumentator import Instrumentator
import psycopg2
import json
from dotenv import load_dotenv
from psycopg2.sql import SQL, Identifier
from datetime import datetime

MODEL_VERSIONS = {
    "v1": {
        "model_path": "checkpoints/v1/siamese_model.pt",
        "encoder_path": "checkpoints/v1/encoder.pkl",
        "model_class": SiameseMLP,
    },
    "v2": {
        "model_path": "checkpoints/v2/siamese_model.pt",
        "encoder_path": "checkpoints/v2/encoder.pkl",
        "model_class": SiameseMLP_v2,
    }
}

MODELS = {}

load_dotenv()
DB_URL = os.environ.get("DATABASE_URL")
def log_to_db(version: str,a: dict, b: dict, score: float):
    try:
        conn = psycopg2.connect(DB_URL)
        cur = conn.cursor()
        table = f"logs_{version}"
        # cur.execute("""
        #     INSERT INTO logs (input_a, input_b, match_score)
        #     VALUES (%s, %s, %s)
        # """, (json.dumps(a), json.dumps(b), score))
        query = SQL("""
            INSERT INTO {} (timestamp, input_a, input_b, match_score)
            VALUES (%s, %s, %s, %s)
        """).format(Identifier(table))
        cur.execute(query, (datetime.utcnow(), json.dumps(a), json.dumps(b), score))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print("❌ Failed to write to DB:", e)

app = FastAPI()

instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app)
# 加载模型和 encoder
model_path = "checkpoints/siamese_model.pt"
encoder_path = "checkpoints/encoder.pkl"

for version, config in MODEL_VERSIONS.items():
    encoder = joblib.load(config["encoder_path"])
    required_columns = encoder.feature_names_in_.tolist()
    dummy_df = pd.DataFrame(columns=encoder.feature_names_in_)
    dummy_df.loc[0] = ["unknown"] * len(encoder.feature_names_in_)
    input_dim = encoder.transform(dummy_df).shape[1]

    model = config["model_class"](input_dim=input_dim)
    model.load_state_dict(torch.load(config["model_path"], map_location="cpu"))
    model.eval()
    MODELS[version] = {
        "model": model,
        "encoder": encoder,
        "columns": encoder.feature_names_in_.tolist()
    }
    print("✅ Model loaded successfully")


class InputFeatures(BaseModel):
    a: dict
    b: dict

def fill_missing_keys(d, keys, default="unknown"):
        return {k: d.get(k, default) for k in keys}

@app.get("/")
def health_check():
    return {"status": "ok", "models": list(MODELS.keys())}

@app.post("/predict")
async def predict(data: InputFeatures, model: str = Query("v1")):
    if model not in MODEL_VERSIONS:
        return {"error": f"Model version {model} not found"}

    encoder = MODELS[model]["encoder"]
    model_obj = MODELS[model]["model"]
    columns = MODELS[model]["columns"]

    try:
        a_fixed = fill_missing_keys(data.a, columns)
        b_fixed = fill_missing_keys(data.b, columns)

        df_a = pd.DataFrame([a_fixed], columns)
        df_b = pd.DataFrame([b_fixed], columns)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a = encoder.transform(df_a)
            b = encoder.transform(df_b)

        a = torch.tensor(a, dtype=torch.float32)
        b = torch.tensor(b, dtype=torch.float32)

        with torch.no_grad():
            output = model_obj(a, b)
            out = output[0].item() if output.numel() > 1 else output.item()

        log_to_db(model, a_fixed, b_fixed, out)
        return {"match_score": out}
    except Exception as e:
        return {"error": str(e)}, 500
