# ディレクトリ構成計画

## 1. 現在の構成
```
.
├── data/                 # 入力データセット (iris.csv, auto_mpg.csv)
├── docs/
│   ├── note/             # メモ（現在空）
│   ├── results/          # 評価結果・図表の格納先
│   ├── spec/             # 要件・設計ドキュメント
│   └── trace/            # 実験ログ・トレース
├── resouce/              # 演習課題資料（PDF/Markdown）
├── pyproject.toml        # プロジェクト設定・依存定義
├── requirements.txt      # 依存パッケージ一覧
└── uv.lock               # 依存ロックファイル
```

## 2. 追加・更新予定
```
.
├── src/
│   └── ml/                       # Python パッケージ（プロジェクト名と合わせる）
│       ├── __init__.py
│       ├── core/                 # 共通ユーティリティ
│       ├── iris/                 # Iris 課題実装
│       ├── auto_mpg/             # Auto MPG 課題実装
│       └── ssl/                  # Masked Autoencoder と転移学習
├── notebooks/
│   ├── iris/                     # 課題 No3 実験ノート
│   └── auto_mpg/                 # 課題 No4・SSL 実験ノート
├── scripts/                      # CLI スクリプト（再現実験、レポート作成）
├── tests/                        # 単体テスト・スモークテスト
├── configs/ (任意)              # 実験設定（YAML/JSON）
├── docs/
│   ├── results/
│   │   ├── figures/              # 可視化画像
│   │   └── tables/               # 指標一覧・比較表
│   └── trace/
│       └── experiments.log       # 実験トレース統合ログ
└── README.md                     # セットアップと実行ガイドを更新予定
```

## 3. 命名と配置ポリシー
- Python パッケージは `src/ml` とし、`pyproject.toml` の `name` に対応させる。
- 課題別コード（Iris, Auto MPG, SSL）はディレクトリ分けし、`core` で共通処理を共有。
- ノートブックは最小限とし、可能な限り `src/ml` の関数を呼び出す形で再現性を担保。
- 生成物（図・表・ログ）は `docs/results` / `docs/trace` に一元管理し、コミット対象とする。
- 追加スクリプト・テストは `scripts/` / `tests/` に配置し、CI やローカル検証を容易にする。

## 4. フェーズ別整備方針
1. **初期整備**: `src/ml` のスケルトン、`notebooks` と `scripts` のルール整備、README 更新。
2. **課題実装フェーズ**: Iris / Auto MPG モジュールとノートブックを追加、`docs/results` に成果物を格納。
3. **SSL 拡張フェーズ**: `src/ml/ssl` を実装し、実験設定・ログ保存枠組みを拡張。
4. **検証・レビュー**: `tests/` でスモークテスト整備、`docs/trace` に実験サマリを残す。
