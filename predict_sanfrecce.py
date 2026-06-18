import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestRegressor

def main():
    csv_filename = 'sanfrecce_real_data.csv'
    
    if not os.path.exists(csv_filename):
        print(f"エラー: {csv_filename} が見つかりません。ファイルを作成してください。")
        return

    print("CSVデータを読み込んでいます...")
    df = pd.read_csv(csv_filename)

    # 特徴量（X）と目的変数（y）に分割
    X = df.drop(['target_GF', 'target_GA'], axis=1)
    y = df[['target_GF', 'target_GA']]

    print(f"総学習データ件数: {len(df)} 試合分")
    print("AIモデル（回帰）を学習させています...")
    
    # 実際のデータ数がまだ少ないため、木の本数(n_estimators)を少し調整
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)

    # 特徴量の重要度を出力
    print("\n◆ 各特徴量の重要度:")
    importances = model.feature_importances_
    for feature, importance in zip(X.columns, importances):
        print(f" - {feature}: {importance:.3f}")

    # 学習済みモデルの保存
    joblib.dump(model, 'sanfrecce_score_model.pkl')
    print("\n本番用モデルを 'sanfrecce_score_model.pkl' として保存しました！")

if __name__ == "__main__":
    main()