# 💜 サンフレッチェ広島 スコア＆勝率予測システム (sanfrecce-predictor)

Jリーグのスタッツ（ゴール期待値など）をもとに、AI（機械学習モデル）がサンフレッチェ広島の試合スコア分布と勝率を確率予測するWebアプリケーションです。

ローカル環境での分析から、FastAPIによるバックエンド構築、そしてHugging Face Spaces（Dockerコンテナ環境）への世界公開まで一連のモダンな開発フローで構築されています。

---

## 🚀 公開URL（Hugging Face Spaces）
[サンフレッチェ広島 スコア予測アプリはこちら](https://huggingface.co/spaces/horihorisoccer/sanfrecce-predictor)

---

## 🛠️ システム概要＆アーキテクチャ

本システムは、軽量かつ高速なWeb APIを構築できる **FastAPI** をバックエンドに採用し、フロントエンド（UI）と学習済みAIモデル（`.pkl`）を統合してDockerコンテナ上で稼働しています。

- **フロントエンド:** HTML5 / CSS3 (サンフレッチェパープル仕様のモダンUI) / JavaScript (Fetch APIによる非同期通信)
- **バックエンド:** FastAPI (Python 3.11) / Uvicorn
- **AI / データ分析:** scikit-learn / joblib / pandas / numpy / scipy
- **インフラ / デプロイ:** Docker / Hugging Face Spaces (無料枠のコンテナ環境: ポート 7860)

---

## 📁 ディレクトリ構造

```text
sanfrecce/
├── .github/workflows/   # CI/CD自動化（GitHub Actions）のデプロイ指示書
├── Dockerfile           # クラウド環境構築用のコンテナ構成台帳
├── requirements.txt     # アプリ起動に必要なPythonライブラリ一覧
├── main.py              # FastAPIバックエンド（予測API・画面配信ロジック）
├── index.html           # フロントエンド（入力フォーム＆結果表示UI）
├── sanfrecce_real_data.csv # 予測のベースとなるチームのスタッツデータ
└── sanfrecce_score_model.pkl # 事前に学習させたスコア予測AIモデル（バイナリファイル）
