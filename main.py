from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Credit Scoring API")

# This fixes the CORS error
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model         = joblib.load('models/lightgbm_model.pkl')
median_values = joblib.load('models/median_values.pkl')

class Features(BaseModel):
    pct_on_time: float
    avg_days_late: float
    num_credit_types: int
    approval_rate: float
    AMT_INCOME_TOTAL: float
    DAYS_BIRTH: float
    EXT_SOURCE_2: float

@app.post("/score")
def score(f: Features):
    row = median_values.copy()
    row['pct_on_time']      = f.pct_on_time
    row['avg_days_late']    = f.avg_days_late
    row['num_credit_types'] = f.num_credit_types
    row['approval_rate']    = f.approval_rate
    row['AMT_INCOME_TOTAL'] = f.AMT_INCOME_TOTAL
    row['DAYS_BIRTH']       = f.DAYS_BIRTH
    row['EXT_SOURCE_2']     = f.EXT_SOURCE_2
    X     = pd.DataFrame([row])[model.feature_name_]
    prob  = float(model.predict_proba(X)[0][1])
    score = int((1 - prob) * 800 + 100)
    if prob < 0.3:
        band = "Low risk"
    elif prob < 0.6:
        band = "Medium risk"
    else:
        band = "High risk"
    return {"score": score, "risk_band": band, "probability": round(prob, 4)}

@app.get("/health")
def health():
    return {"status": "ok"}