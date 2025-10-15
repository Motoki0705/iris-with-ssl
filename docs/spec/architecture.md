# アーキテクチャ設計

## 1. システム概要
本プロジェクトは 2 系統の課題（Iris 分類・Auto MPG 回帰）と自己教師あり学習（SSL）を統合した学習/実験基盤を構築する。`src/ml` 以下に共通ユーティリティと課題別モジュールを実装し、ノートブックやスクリプトから再利用できる API を提供する。

```
data (CSV) ──► data loaders ──► feature pipelines ──► models ──► evaluators ──► reports
                                        │                         │
                                        └──────────── logs / artifacts ─► docs/results
```

## 2. コンポーネント構成

### 2.1 共通レイヤー (`src/ml/core`)
- `datasets.py`: データパス解決、CSV ロード、学習/評価分割などを提供。
- `features.py`: 共通の前処理（スケーリング、欠損補完、特徴選択）。
- `metrics.py`: 分類/回帰/再構成に対応した指標算出（accuracy, confusion matrix, MSE, R2, MAE など）。
- `visualization.py`: 散布図、決定境界、残差プロット等の可視化ユーティリティ。
- `logging.py`: 実験設定・結果を `docs/results` と `docs/trace` へ書き出すヘルパ。

### 2.2 Iris 課題モジュール (`src/ml/iris`)
- `plotting.py`: 1-Sample_iris-DataPlot の処理。マーカー色・形のパラメータ化、図の保存。
- `knn_classifier.py`: S2 課題用 KNN モデル生成、グリッド可視化、混同行列算出。
- `kmeans.py`: S3 課題用 K-means クラスタリングと図示。
- `experiments.py`: k 値・特徴ペアを変えた実験ループと結果のエクスポート。

### 2.3 Auto MPG 課題モジュール (`src/ml/auto_mpg`)
- `knn_regressor.py`: S4 課題用 KNN 回帰パイプライン。入力特徴の組み合わせと k を切り替え可能。
- `linear_models.py`: 線形回帰・多項式回帰（X² 含む）の学習と評価。
- `comparison.py`: MSE・R2 ベースのモデル比較とランキング、Discussion のアウトライン生成。
- `visualization.py`: 回帰曲線、予測 vs 実測グラフ、残差ヒストグラム。

### 2.4 SSL モジュール (`src/ml/ssl`)
- `masking.py`: 特徴マスクのサンプリング戦略（割合、一様/ランダム/構造化）を提供。
- `autoencoder.py`: Masked Autoencoder のエンコーダ/デコーダ定義（PyTorch）。
- `trainer.py`: 学習ループ、ノイズ付与、チェックポイント管理。
- `transfer.py`: 事前学習済みバックボーンから `mpg` を推論する小規模ヘッドのファインチューニング。
- `evaluation.py`: 再構成損失、下流回帰指標の測定と比較レポート生成。

### 2.5 実験エントリポイント
- `notebooks/`（新設）: 課題ごとのインタラクティブ実験用。Python モジュールを呼び出す構成に統一。
- `scripts/`（新設）: CLI 実行用スクリプト（例: `run_iris_knn.py`, `run_auto_mpg_ssl.py`）。
- `configs/`（任意）: YAML 等で実験設定を定義し、再現性を確保。

## 3. データフロー
1. `data` から CSV をロード → 必要なクリーニングと特徴加工。
2. Iris/Auto MPG でタスク固有のパイプラインを構築。
3. モデルを訓練後、評価指標と可視化を生成。
4. 実験管理レイヤーが結果をファイル出力し、`docs/results` に保存。
5. SSL 経由の特徴表現は Auto MPG 回帰タスクに転移し、ベースラインと比較される。

## 4. ロギングと成果物
- 主要指標・ハイパーパラメータを JSON/CSV 形式で `docs/results` に保存。
- 画像や図表は `docs/results/figures` などに配置。
- 実験トレース（利用したモジュール、git ハッシュ、実行日時）は `docs/trace` に記録。

## 5. マルチエージェント連携
- planner: 要件と設計（本書・requirements・タスク表）を維持。
- coder: 上記モジュールを実装し、スクリプト/ノートブックから利用できるようにする。
- tester: 単体テスト・スモークテスト・再現実験を `tests/` もしくは `scripts/` で実施。
- judge: `docs/results` と `docs/trace` を確認し成果物をレビュー、必要なフィードバックを planner に返す。

## 6. 技術スタック
- Python 3.11 / PyTorch / scikit-learn / pandas / seaborn / matplotlib
- 追加ツール: pytest（テスト）、hydra または argparse（設定管理）、black/ruff（整形・Lint; 任意）
