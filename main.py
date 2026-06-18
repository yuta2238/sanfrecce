from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware # ←追加
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

app = FastAPI(title="サンフレッチェ広島 スコア予測API", version="2.0")

# --- CORS設定を追記 (ここから) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # すべての場所（HTMLファイル）からのアクセスを許可
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# --- CORS設定を追記 (ここまで) ---

model = joblib.load('sanfrecce_score_model.pkl')

class MatchData(BaseModel):
    opponent_past_ppg: float
    recent_xG: float
    recent_xGA: float
    past_1_GF: int
    past_1_GA: int
    past_2_GF: int
    past_2_GA: int
    past_3_GF: int
    past_3_GA: int
    past_4_GF: int
    past_4_GA: int
    past_5_GF: int
    past_5_GA: int

@app.post("/predict")
def predict_match_score(data: MatchData):
    input_df = pd.DataFrame([data.model_dump()])
    prediction = model.predict(input_df)[0]
    
    pred_gf = int(np.round(prediction[0]))
    pred_ga = int(np.round(prediction[1]))
    
    if pred_gf > pred_ga:
        result_text = "勝ち"
    elif pred_gf < pred_ga:
        result_text = "負け"
    else:
        result_text = "引き分け"
    
    return {
        "status": "success",
        "predicted_score": f"{pred_gf} - {pred_ga}",
        "predicted_result": result_text,
        "raw_prediction": {
            "expected_GF": round(prediction[0], 2),
            "expected_GA": round(prediction[1], 2)
        }
    }