# 1. ベースとなるPythonのイメージを指定
FROM python:3.11-slim

# 2. コンテナ内の作業ディレクトリを設定
WORKDIR /app

# 3. 必要なライブラリの一覧（requirements.txt）をコンテナにコピー
COPY requirements.txt .

# 4. コンテナ内でライブラリをインストール
RUN pip install --no-cache-dir -r requirements.txt

# 5. プログラムコードと学習済みモデルをコンテナにコピー
COPY main.py .
COPY sanfrecce_score_model.pkl .

# 6. コンテナが起動したときに実行するコマンドを指定
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]