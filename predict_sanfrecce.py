import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def main():
    # ---------------------------------------------------------
    # 1. データの準備（今回はダミーデータを生成します）
    # ---------------------------------------------------------
    print("データの準備を行っています...")
    
    # サンプルとして100試合分のデータをランダムに生成
    np.random.seed(42)
    data = {
        'opponent_past_ppg': np.random.uniform(0.0, 3.0, 100), # 過去の相性（平均獲得勝ち点）
        'recent_5_points': np.random.randint(0, 16, 100),      # 直近5試合の勝ち点
        'recent_xG': np.random.uniform(0.5, 3.0, 100),         # 直近5試合の平均得点期待値
        'recent_xGA': np.random.uniform(0.5, 3.0, 100),        # 直近5試合の平均失点期待値
        'result': np.random.choice([1, 0, -1], 100)            # 試合結果 (1:勝ち, 0:引き分け, -1:負け)
    }
    df = pd.DataFrame(data)

    # 特徴量（X）と目的変数（y）に分割
    X = df.drop('result', axis=1)
    y = df['result']

    # 学習用データ(80%)とテスト用データ(20%)に分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ---------------------------------------------------------
    # 2. モデルの学習
    # ---------------------------------------------------------
    print("AIモデルを学習させています...")
    # ランダムフォレスト分類器を使用
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # ---------------------------------------------------------
    # 3. 予測と評価
    # ---------------------------------------------------------
    print("テストデータで予測精度を評価します...\n")
    y_pred = model.predict(X_test)
    
    # 正解率の計算
    accuracy = accuracy_score(y_test, y_pred)
    print(f"◆ モデルの正解率 (Accuracy): {accuracy * 100:.2f}%\n")
    
    # 詳細な評価レポート
    print("◆ 分類レポート:")
    print(classification_report(y_test, y_pred, target_names=['負け(-1)', '引き分け(0)', '勝ち(1)'], zero_division=0))

    # ---------------------------------------------------------
    # 4. 特徴量の重要度を確認
    # ---------------------------------------------------------
    # どの入力データが予測に最も貢献したかを確認します
    print("◆ 特徴量の重要度:")
    importances = model.feature_importances_
    for feature, importance in zip(X.columns, importances):
        print(f" - {feature}: {importance:.3f}")
    # ---------------------------------------------------------
    # 5. 学習済みモデルの保存
    # ---------------------------------------------------------
    joblib.dump(model, 'sanfrecce_model.pkl')
    print("モデルを 'sanfrecce_model.pkl' として保存しました！")

if __name__ == "__main__":
    main()