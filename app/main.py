import json
import numpy as np
import pandas as pd
import pickle
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path

APP_DIR = Path(__file__).parent

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load artifacts at startup
with open(APP_DIR / "meta_model.pkl", "rb") as f:
    model = pickle.load(f)

with open(APP_DIR / "feature_cols_final.json") as f:
    feature_cols = json.load(f)

df = pd.read_parquet(APP_DIR / "final_merged_test.parquet")

# Batch predict once at startup
X_all = df[feature_cols].fillna(0).values
probs_all = model.predict_proba(X_all)[:, 1]
df["risk_probability"] = np.round(probs_all, 4)
df["risk_level"] = pd.cut(
    df["risk_probability"],
    bins=[-0.01, 0.25, 0.5, 1.01],
    labels=["Low", "Medium", "High"],
)

# Pre-build patient list response
patients_cache = []
for _, row in df.iterrows():
    patients_cache.append({
        "hadm_id": int(row["hadm_id"]),
        "age": float(row.get("age", 0)),
        "los": float(row.get("los", 0)),
        "icd9_group": str(row.get("icd9_group", "Unknown")),
        "risk_probability": float(row["risk_probability"]),
        "risk_level": str(row["risk_level"]),
        "readmit_30d": int(row.get("readmit_30d", 0)),
    })


class PredictRequest(BaseModel):
    hadm_id: int


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/patients")
def get_patients():
    return patients_cache


@app.post("/predict")
def predict(req: PredictRequest):
    match = df[df["hadm_id"] == req.hadm_id]
    if match.empty:
        raise HTTPException(status_code=404, detail="Patient not found")
    row = match.iloc[0]
    return {
        "hadm_id": int(row["hadm_id"]),
        "risk_probability": float(row["risk_probability"]),
        "risk_level": str(row["risk_level"]),
        "age": float(row.get("age", 0)),
        "los": float(row.get("los", 0)),
        "sofa": float(row.get("sofa", 0) if pd.notna(row.get("sofa")) else 0),
        "apsiii": float(row.get("apsiii", 0) if pd.notna(row.get("apsiii")) else 0),
        "elixhauser": float(row.get("elixhauser", 0) if pd.notna(row.get("elixhauser")) else 0),
        "num_prior_admissions": int(row.get("num_prior_admissions", 0)),
        "days_since_last_admission": float(row.get("days_since_last_admission", -1)),
        "num_diagnoses": int(row.get("num_diagnoses", 0)),
        "is_emergency": int(row.get("is_emergency", 0)),
        "dc_home": int(row.get("dc_home", 0)),
        "dc_snf_rehab": int(row.get("dc_snf_rehab", 0)),
        "dc_other_facility": int(row.get("dc_other_facility", 0)),
        "icd9_group": str(row.get("icd9_group", "Unknown")),
        "readmit_30d": int(row.get("readmit_30d", 0)),
    }
