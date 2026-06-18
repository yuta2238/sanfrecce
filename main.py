from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from scipy.stats import poisson # ← 確率分布計算用に追加

app = FastAPI(title="サンフレッチェ広島 高度確率予測API", version="4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load('sanfrecce_score_model.pkl')

class MatchData(BaseModel):
    is_home: int
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
    
    # 1. チームごとの決定力・守備力の「重み（係数）」を設定
    # サンフレッチェの現在の戦況に合わせた補正係数（調整可能です）
    HIROSHIMA_ATTACK_WEIGHT = 1.15  # 得点期待値を15%高める重み
    HIROSHIMA_DEFENSE_WEIGHT = 0.90 # 失点期待値を10%抑える重み（堅守）
    
    # 重みづけされた期待値（ポアソン分布のパラメータλ）
    lambda_gf = prediction[0] * HIROSHIMA_ATTACK_WEIGHT
    lambda_ga = prediction[1] * HIROSHIMA_DEFENSE_WEIGHT
    
    # 2. 0点〜5点までのすべてのスコアパターンの確率を計算
    max_prob = -1
    best_gf, best_ga = 0, 0
    score_distribution = [] # 画面に渡す確率分布リスト
    
    for gf in range(6): # 0〜5点
        for ga in range(6): # 0〜5点
            # ポアソン分布からそれぞれの得点・失点になる確率を計算
            prob_gf = poisson.pmf(gf, lambda_gf)
            prob_ga = poisson.pmf(ga, lambda_ga)
            # 同時に起こる確率（掛け算）
            total_prob = prob_gf * prob_ga
            
            score_distribution.append({
                "score": f"{gf} - {ga}",
                "prob": round(total_prob * 100, 1) # パーセント表記 (例: 12.5)
            })
            
            # 最も確率が高いスコアを記憶（最大値の更新）
            if total_prob > max_prob:
                max_prob = total_prob
                best_gf, best_ga = gf, ga
                
    # 確率が高い順にソート（上位5件を画面に表示するため）
    score_distribution = sorted(score_distribution, key=lambda x: x['prob'], reverse=True)
    
    if best_gf > best_ga:
        result_text = "勝ち"
    elif best_gf < best_ga:
        result_text = "負け"
    else:
        result_text = "引き分け"
        
    return {
        "status": "success",
        "predicted_score": f"{best_gf} - {best_ga}",
        "predicted_result": result_text,
        "max_probability": round(max_prob * 100, 1),
        "top_distribution": score_distribution[:5], # 上位5つの確率分布
        "raw_prediction": {
            "weighted_GF": round(lambda_gf, 2),
            "weighted_GA": round(lambda_ga, 2)
        }
    }