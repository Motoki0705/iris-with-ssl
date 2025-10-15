# 実装タスクチェックリスト

## 0. タスク管理方針
- 進捗ステータスはチェックボックスで管理し、`[ ]` = 未着手（Backlog）、`[~]` = 進行中（In Progress）、`[x]` = 完了（Done）を想定。更新は担当エージェントが行い、planner と orchestrator が総括する。
- チケット運用は `docs/spec/agents.md` と `docs/tickets/overview.md` に従う。各タスクは対応するチケットIDを持ち、チケットファイルと双方向で同期する。
- 成果物・エビデンスは `docs/results/<ticket-id>.*`・`docs/trace/<ticket-id>.md` に保存し、judge および orchestrator/reviewer が参照できる状態にする。

## レベル1. 基盤整備
- [x] F-01 coder: `src/ml` パッケージ骨組み（core/iris/auto_mpg/ssl）を作成（依存: requirements, architecture）
- [x] F-02 coder: `core/{datasets,features,metrics,visualization,logging}.py` のひな型を用意（依存: F-01）
- [x] F-03 planner: README にセットアップ・実行ガイドの初稿を追加（依存: F-01）
- [x] F-04 tester: ベースラインスモークテスト構成を検討（pytest or CLI）（依存: F-01）

## レベル2. Iris 課題（No3）
- [ ] I-01 coder: `iris/plotting.py` を実装（カスタムマーカー設定・図保存）（依存: F-01, F-02）
- [ ] I-02 coder: `iris/knn_classifier.py` で k パラメータ化・決定境界描画・混同行列出力（依存: I-01）
- [ ] I-03 coder: `iris/kmeans.py` で k=3/4/5 と特徴ペア比較を実装（依存: I-02）
- [ ] I-04 coder: `iris/experiments.py` で一括実行と結果保存フローを構築（依存: I-02, I-03）
- [ ] I-05 tester: Iris ワークフローのスモークテスト（データロード→指標算出）（依存: I-04）
- [ ] I-06 judge: 図・混同行列・考察メモのレビューと改善提案（依存: I-04）

## レベル3. Auto MPG 課題（No4）
- [ ] M-01 coder: `auto_mpg/knn_regressor.py` 実装（特徴組合せと k 比較）（依存: F-01, F-02）
- [ ] M-02 coder: `auto_mpg/linear_models.py` 実装（単回帰・多項式回帰）（依存: M-01）
- [ ] M-03 coder: `auto_mpg/comparison.py` で MSE/R2 ランキングとレポート生成（依存: M-01, M-02）
- [ ] M-04 coder: `auto_mpg/visualization.py` で回帰曲線・残差図を生成（依存: M-01, M-02）
- [ ] M-05 tester: Auto MPG ワークフロースモークテスト（3モデル以上を検証）（依存: M-03, M-04）
- [ ] M-06 judge: Discussion セクション内容のレビューと改善指示（依存: M-03）

## レベル4. SSL 拡張
- [ ] S-01 planner: Masked Autoencoder の詳細設計（潜在次元・層構成）を決定（依存: architecture）
- [ ] S-02 coder: `ssl/masking.py` でマスク戦略とノイズ付与を実装（依存: S-01, F-02）
- [ ] S-03 coder: `ssl/autoencoder.py` と `trainer.py` でモデル定義・学習ループを構築（依存: S-02）
- [ ] S-04 coder: `ssl/transfer.py` で回帰ヘッドの転移学習機能を実装（依存: S-03, M-01）
- [ ] S-05 coder: `ssl/evaluation.py` で再構成/下流指標の比較ロジックを整備（依存: S-04）
- [ ] S-06 tester: SSL 学習＋転移の再現テスト（短縮エポック）（依存: S-05）
- [ ] S-07 judge: ベースライン比較結果のレビューと成果報告チェック（依存: S-05）

## レベル5. ノートブック / スクリプト整備
- [ ] NB-01 coder: `notebooks/iris/` に参照用ノート（モジュール呼び出し形式）を準備（依存: I-04）
- [ ] NB-02 coder: `notebooks/auto_mpg/` に回帰・SSL 実験ノートを整備（依存: M-03, S-05）
- [ ] SC-01 coder: `scripts/run_iris_tasks.py`（CLI 実行）を作成（依存: I-04）
- [ ] SC-02 coder: `scripts/run_auto_mpg_tasks.py`, `run_auto_mpg_ssl.py` を作成（依存: M-03, S-05）
- [ ] SC-03 tester: CLI スクリプトのスモークテストと使用ガイド更新（依存: SC-01, SC-02）

## レベル6. ドキュメント＆レビュー
- [ ] D-01 planner: `docs/results` テンプレート（図・表命名規則）策定（依存: F-02）
- [ ] D-02 planner: `docs/trace/experiments.log` 記録フォーマット定義（依存: F-02）
- [ ] D-03 judge: 成果物レビュー指標（精度基準・可視化品質）を明文化（依存: D-01, D-02）
- [ ] D-04 planner: 定期ステータスレポートで各ロール進捗を共有（依存: 全体）

## 7. リスク・フォローアップ
- SSL 計算コスト → S-03/S-04 を小規模設定で試験後、必要に応じて拡張
- データ拡張の効果不確実性 → S-05 でノイズ強度スイープを計画
- 時間制約 → I/M 系統タスクを優先、SSL は並行して段階実装

## 8. 運用ノート
- planner はチケット作成・クローズ時に本チェックリストの該当項目を更新し、overview の順序と整合させる。
- orchestrator/reviewer（ユーザー）は進捗報告や承認状況に応じて担当エージェントへ指示を出し、必要に応じてチェックボックスへ `[~]`（進行中）や備考を追記する。
- judge 承認後に `[x]` を付与し、成果物パスが `docs/results` と `docs/trace` に存在することを確認する。
- 各タスクの詳細な要件・成果物規定は `docs/spec/agents.md`、`docs/tickets/<ticket-id>.md` を参照する。
