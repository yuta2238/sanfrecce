from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# FastAPIのアプリケーションを初期化
app = FastAPI(title="サンフレッチェ広島 試合結果予測API")

# 保存した学習済みモデルを読み込み
model = joblib.load('sanfrecce_model.pkl')

# 入力されるデータの形式（型）を定義
class MatchData(BaseModel):
    opponent_past_ppg: float
    recent_5_points: int
    recent_xG: float
    recent_xGA: float

# POSTメソッドで予測リクエストを受け取るエンドポイントを作成
@app.post("/predict")
def predict_match(data: MatchData):
    # 受け取ったデータをPandasのDataFrameに変換
    input_df = pd.DataFrame([data.model_dump()])

    # モデルを使って予測を実行
    prediction = model.predict(input_df)[0]

    # 予測結果（数値）を文字列に変換
    result_map = {1: "勝ち", 0: "引き分け", -1: "負け"}
    result_text = result_map[int(prediction)]

    # 結果をJSON形式で返す
    return {
        "status": "success",
        "input_data": data.model_dump(),
        "prediction": result_text
    }