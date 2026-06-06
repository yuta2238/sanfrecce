import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

def main():
    # ---------------------------------------------------------
    # 1. データの準備（時系列ラグ特徴量を含むダミーデータ）
    # ---------------------------------------------------------
    print("データの準備を行っています...")
    np.random.seed(42)

    # 基礎データ
    data = {
        'opponent_past_ppg': np.random.uniform(0.0, 3.0, 100), # 過去の相性
        'recent_xG': np.random.uniform(0.5, 3.0, 100),         # 直近平均得点期待値
        'recent_xGA': np.random.uniform(0.5, 3.0, 100),        # 直近平均失点期待値
    }

    # 【追加要素】過去5試合のスコア（時間遅れ・ラグ特徴量）
    # GF = Goals For (得点), GA = Goals Against (失点)
    for i in range(1, 6):
        data[f'past_{i}_GF'] = np.random.randint(0, 4, 100) # i試合前の得点
        data[f'past_{i}_GA'] = np.random.randint(0, 4, 100) # i試合前の失点

    # 【変更要素】予測ターゲット（今回の試合の「得点」と「失点」）
    data['target_GF'] = np.random.randint(0, 4, 100)
    data['target_GA'] = np.random.randint(0, 4, 100)

    df = pd.DataFrame(data)

    # 特徴量（X）と目的変数（y）に分割
    X = df.drop(['target_GF', 'target_GA'], axis=1)
    y = df[['target_GF', 'target_GA']] # 2つの数値を同時に予測対象にする

    # 学習用データ(80%)とテスト用データ(20%)に分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ---------------------------------------------------------
    # 2. モデルの学習（回帰モデルへ変更）
    # ---------------------------------------------------------
    print("AIモデルを学習させています...")
    # Classifier（分類）ではなく Regressor（回帰）を使用
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # ---------------------------------------------------------
    # 3. 予測と評価
    # ---------------------------------------------------------
    print("テストデータでスコア予測精度を評価します...\n")
    y_pred = model.predict(X_test)
    
    # 予測値は小数になるため、四捨五入して整数（スコア）にする
    y_pred_rounded = np.round(y_pred).astype(int)
    y_test_values = y_test.values

    # 評価指標の計算（平均絶対誤差: MAE）
    mae = mean_absolute_error(y_test_values, y_pred)
    print(f"◆ 平均絶対誤差 (MAE): {mae:.2f} 点")
    print("  ※予測スコアが実際のスコアから平均して何点ズレているかを示します\n")

    # 予測スコアのサンプル表示
    print("◆ 予測スコアのサンプル（最初の5件）:")
    for i in range(5):
        actual_gf, actual_ga = y_test_values[i]
        pred_gf, pred_ga = y_pred_rounded[i]
        
        # 勝敗結果の判定テキスト
        actual_res = "勝ち" if actual_gf > actual_ga else "負け" if actual_gf < actual_ga else "引分"
        pred_res = "勝ち" if pred_gf > pred_ga else "負け" if pred_gf < pred_ga else "引分"
        
        print(f"  実際: {actual_gf} - {actual_ga} ({actual_res})  |  予測: {pred_gf} - {pred_ga} ({pred_res})")

    # ---------------------------------------------------------
    # 4. 特徴量の重要度を確認
    # ---------------------------------------------------------
    print("\n◆ 特徴量の重要度 (上位5つ):")
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    for idx in sorted_idx[:5]:
        print(f" - {X.columns[idx]}: {importances[idx]:.3f}")

    # ---------------------------------------------------------
    # 5. 学習済みモデルの保存
    # ---------------------------------------------------------
    joblib.dump(model, 'sanfrecce_score_model.pkl')
    print("\nモデルを 'sanfrecce_score_model.pkl' として保存しました！")

if __name__ == "__main__":
    main()