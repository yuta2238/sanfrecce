import requests
import pandas as pd
import time

def get_football_lab_results():
    # 2024年シーズンのサンフレッチェ広島のデータページ
    url = "https://www.football-lab.jp/hiro/match/?year=2024"
    
    # 正規のブラウザからのアクセスであることを示す
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    
    print(f"URLにアクセス中: {url}")
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        # 日本語の文字化けを防ぐ処理
        response.encoding = response.apparent_encoding
        
        # Pandasの強力な機能で、HTML内の表（テーブル）をすべて自動検出
        tables = pd.read_html(response.text)
        
        target_df = None
        for df in tables:
            # Football LABの表の列名には「相手」と「スコア」が含まれる
            if '相手' in df.columns and 'スコア' in df.columns:
                target_df = df
                break
                
        if target_df is None:
            print("該当するデータテーブルが見つかりませんでした。")
            return None
            
        # 機械学習の特徴量として優秀な列だけを狙って抽出
        # ※サイト側の仕様変更に備え、存在する列のみを動的にフィルタリング
        needed_cols = ['開催日', '相手', 'スコア', 'チャンス構築率', '保持率', 'シュート']
        actual_cols = [c for c in needed_cols if c in target_df.columns]
        
        df_cleaned = target_df[actual_cols].copy()
        
        # 「スコア」列に「-」が含まれていない行（まだ試合が行われていない未来の予定）を除外
        df_cleaned = df_cleaned.dropna(subset=['スコア'])
        df_cleaned = df_cleaned[df_cleaned['スコア'].astype(str).str.contains('-')]
        
        # 直近の5試合分を取得
        latest_5 = df_cleaned.tail(5)
        return latest_5

    except Exception as e:
        print(f"通信または解析中にエラーが発生しました: {e}")
        return None

if __name__ == "__main__":
    results = get_football_lab_results()
    
    if results is not None and not results.empty:
        print("\n--- サンフレッチェ広島 直近5試合のデータ ---")
        print(results.to_string(index=False))
    else:
        print("\nデータを取得できませんでした。")
        
    # サーバーへの配慮（1秒待機）
    time.sleep(1)