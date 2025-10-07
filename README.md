# iris-with-ssl

## 概要
- アヤメ (Iris) データセットを題材に、教師あり・教師なし・自己教師あり学習を横断的に試せる学習リポジトリです。
- `src/` 配下は用途ごとに整理され、`demo/` に教師ありデモ、`materials/` に実習教材、`ssl/` に自己教師あり実装を配置しています。
- 今後 SimCLR など他の自己教師あり手法を追加できるよう、拡張性を意識した構成になっています。

## ディレクトリ構成
```
.
├── README.md
├── requirements.txt        # pip 用の依存パッケージ一覧
├── iris.csv                # 学習・評価用の Iris データセット
├── main.py                 # 最小限のエントリーポイント (placeholder)
├── pyproject.toml          # 依存関係とメタデータ
├── sanity_test.py          # 主要ライブラリとデータの動作確認
├── src
│   ├── demo                # 教師ありベースラインのスクリプト
│   │   ├── knn.py
│   │   └── mlp.py
│   ├── materials           # 実習教材 (ipynb / markdown)
│   │   ├── ipynb
│   │   │   ├── 1-Sample_iris-DataPlot.ipynb
│   │   │   ├── S2-Sample_iris-knnClassifier.ipynb
│   │   │   └── S3-Sample_iris-KmeansCluster.ipynb
│   │   └── markdown
│   │       ├── 1-Sample_iris-DataPlot.md
│   │       ├── S2-Sample_iris-knnClassifier.md
│   │       └── S3-Sample_iris-KmeansCluster.md
│   ├── ssl                 # 自己教師あり学習 (SSL) 実装群
│   │   └── mae.py          # Masked Autoencoder 実装と下流評価
│   └── tools               # 補助スクリプト
│       └── convert_md_to_ipynb.py
└── uv.lock                 # uv でロックした依存関係 (任意)
```

### `src/` 内訳
| パス | 目的 | メモ |
| --- | --- | --- |
| `src/demo/knn.py` | k近傍法 (kNN) による分類ベースライン | 特徴量を標準化してホールドアウト評価します。|
| `src/demo/mlp.py` | scikit-learn の多層パーセプトロンによる分類 | `hidden_layer_sizes` や `solver` を変更して挙動を比較できます。|
| `src/materials/ipynb/*.ipynb` | 実習ノートブック | ノートブック形式で演習を進めるための教材です。|
| `src/materials/markdown/*.md` | 実習教材の Markdown 版 | `convert_md_to_ipynb.py` でノートブックへ再生成可能。|
| `src/tools/convert_md_to_ipynb.py` | Markdown から `.ipynb` を再生成する補助スクリプト | CI や資料更新時に活用できます。|
| `src/ssl/mae.py` | Masked Autoencoder による自己教師ありパイプライン | 事前学習・線形評価・凍結ヘッド微調整を備え、今後の SSL 実装のリファレンスです。|

## セットアップ
### 前提
- Python 3.11 以降
- 推奨: [uv](https://github.com/astral-sh/uv) もしくは標準の `venv` + `pip`

### uv を使う場合
```bash
uv sync --frozen
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```
> `uv sync` は `pyproject.toml` と `uv.lock` をもとに仮想環境を自動生成します。

### venv + pip を使う場合
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 使い方
- kNN ベースライン: `python src/demo/knn.py`
- MLP ベースライン: `python src/demo/mlp.py`
- 自己教師あり学習 (MAE):
  ```bash
  python src/ssl/mae.py \
      --epochs 200 \
      --mask-ratio 0.75 \
      --latent-dim 8 \
      --eval-mode ft
  ```
  - `--eval-mode linear` で線形分類器 (ロジスティック回帰) のみ評価。
  - `--checkpoint-dir` でエンコーダ/デコーダ重みの保存先ディレクトリを指定可能。

## Jupyter Notebook (実習課題)
1. `src/materials/ipynb/1-Sample_iris-DataPlot.ipynb`: Iris の基本統計と可視化
2. `src/materials/ipynb/S2-Sample_iris-knnClassifier.ipynb`: kNN のハイパーパラメータ探索
3. `src/materials/ipynb/S3-Sample_iris-KmeansCluster.ipynb`: クラスタリングによるクラス構造の把握

> Markdown 版 (`src/materials/markdown/*`) は講義資料などへの再利用を想定しています。更新後は `python src/tools/convert_md_to_ipynb.py` でノートブックへ変換してください。

## Self-Supervised Learning (SSL) の拡張方針
- 各手法ごとに `src/ssl/<method>.py` を作成し、共通の CLI インターフェース (`argparse`) を整えることで実験設定を比較しやすくします。
- 拡張候補:
  - SimCLR: タブular データ用のデータ増強ポリシーとコントラスト損失を導入。
  - BYOL / Barlow Twins: 共有エンコーダとターゲットネットワークを扱うユーティリティを `ssl/` 配下で共通化。
- 評価指標は MAE と同様に、線形評価・凍結ヘッド微調整・ランダム初期化ベースラインを揃え、成果を横並びで比較します。
- 生成された重みやメタデータは `artifacts/<method>/` に保存し、再現実験や可視化ノートブックから再利用できる設計にします。

## 動作確認
```bash
python sanity_test.py
```
主要ライブラリのバージョンと Iris データのロード確認を行います。Torch/CUDA の利用可否もここで把握できます。

## 今後のロードマップ (一例)
- SimCLR のタブular対応実装と MAE との比較実験
- SSL で得た特徴量の可視化ノートブック (PCA/UMAP) 追加
- パラメータサーチや学習曲線を記録するための logging フレームワーク導入

## ライセンス
このリポジトリのライセンスは未設定です。公開する場合は適切なライセンスを `LICENSE` ファイルとして追加してください。
