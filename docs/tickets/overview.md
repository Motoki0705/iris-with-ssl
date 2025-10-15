# チケット運用概要

## 1. 目的
`docs/tickets/` は planner が発行するチケットを管理し、各エージェントの担当順序と履歴を記録するリポジトリ内の単一ソースとする。オーケストレータ / reviewer（ユーザー）は本書と planner の指示を参照しながら、coder/tester/judge に具体的な作業指示を出し、進捗を調整する。

## 2. ロール別の操作ガイド
- **planner**: チケットを作成し、本 overview の表に追加。依存関係と担当を調整し、完了後は履歴へ移動。
- **orchestrator / reviewer（ユーザー）**: planner の計画と overview を基にエージェントへ指示。進行状況をヒアリングし、状態列や履歴更新を planner と連携して行う。
- **coder / tester**: 割り当てられたチケットを処理し、ステータスを更新。成果物は `docs/results/<ticket-id>.*` に、ログは `docs/trace/<ticket-id>.md` に保存し、チケットファイルへ要約を追記。
- **judge**: `docs/results` と `docs/trace` を確認し、要件適合性を判断。結果をチケットと overview に反映し、必要ならフォローアップチケットを planner に提案。

## 3. チケットファイル作成ルール
- 各チケットは `docs/tickets/<ticket-id>.md` 形式で作成する。
- 必須項目:
  - 背景・目的
  - 完了条件（チェックリスト推奨）
  - 期待成果物（`docs/results/<ticket-id>.*`, `docs/trace/<ticket-id>.md` 等）
  - 依存関係と参照ドキュメント（例: `docs/spec/*.md`）
- coder/tester は完了時に成果概要と保存先を追記し、judge レビュー待ちであることを明記する。

## 4. チケットライフサイクルとステータス
- **Backlog**（未着手）: planner が登録した直後の状態。表の「状態」列には `Backlog` と記載。
- **In Progress**（進行中）: coder/tester が着手したら `In Progress`。必要に応じて開始日をメモ列に追記。
- **Review**（レビュー待ち）: 成果物が `docs/results` / `docs/trace` に揃い、judge 確認待ちの状態。メモ欄に保存先をリンク形式で記載。
- **Done**（完了）: judge 承認後に `Done` とし、履歴へ移動。再オープンが必要な場合はメモに理由とフォローアップチケットIDを記載。

## 5. 割り当てと順序管理
以下の表で各チケットの担当ロールと実施順序を管理する。基本は上から順に処理するが、planner と orchestrator が優先度に応じて並び替える。

| 順序 | チケットID | タイトル | 担当 | 状態 | メモ |
|------|------------|----------|------|------|------|
| 1 | I-01 | Iris 可視化ユーティリティ実装 | coder | Review | 成果: `docs/results/I-01.md`, トレース: `docs/trace/I-01.md`, 図: `docs/results/figures/I-01_petal_length_petal_width.png`
| 2 | I-02 | Iris KNN 分類パイプライン | coder | Backlog | 依存: I-01。成果: `iris/knn_classifier.py`
| 3 | I-03 | Iris K-means クラスタリング | coder | Backlog | 依存: I-02。成果: `iris/kmeans.py`
| 4 | I-04 | Iris 実験オーケストレーション | coder | Backlog | 依存: I-03。成果: `iris/experiments.py`
| 5 | I-05 | Iris ワークフロー検証テスト | tester | Backlog | 依存: I-04。結果: `docs/results/I-05.md`
| 6 | I-06 | Iris 成果レビュー | judge | Backlog | 依存: I-05。レポート: `docs/judges/I-06.md`
| 7 | M-01 | Auto MPG KNN 回帰パイプライン | coder | Review | 成果: `docs/results/M-01.md` / `docs/trace/M-01.md`
| 8 | M-02 | Auto MPG 線形・多項式回帰 | coder | Review | 成果: `docs/results/M-02.md` / `docs/trace/M-02.md`
| 9 | M-03 | Auto MPG モデル比較レポート | coder | Review | 成果: `docs/results/M-03.md` / `docs/results/discussion/M-03_report.md`
| 10 | M-04 | Auto MPG 回帰可視化ユーティリティ | coder | Backlog | 依存: M-01〜M-03。図: `docs/results/figures/M-04_*.png`
| 11 | M-05 | Auto MPG ワークフロー検証テスト | tester | Backlog | 依存: M-01〜M-04。結果: `docs/results/M-05.md`
| 12 | M-06 | Auto MPG Discussion レビュー | judge | Backlog | 依存: M-01〜M-05。レポート: `docs/judges/M-06.md`

> 更新時は最新順序を保つため、完了チケットは下部の「完了履歴」に移す。進行中の優先度変更は planner と orchestrator が合意の上で順序を入れ替える。I 系列（Level 2）と M 系列（Level 3）は並列進行を想定しているため、空きリソースに応じて割り当て順を柔軟に調整する。

## 6. 履歴管理
- 完了したチケットは完了日時・担当者・judge 判定結果を明記して履歴へ移動。
- 再オープンや派生タスクが発生した場合は履歴に追記し、新しいチケットIDをリンクする。

### 完了履歴
- 2025-10-14 F-01 coder: パッケージ骨組み整備 → 成果: `docs/results/F-01.md`, Trace: `docs/trace/F-01.md`
- 2025-10-14 F-02 coder: コアユーティリティ初期実装 → 成果: `docs/results/F-02.md`, Trace: `docs/trace/F-02.md`
- 2025-10-14 F-03 planner: README セットアップガイド初稿 → 成果: `docs/results/F-03.md`, Trace: `docs/trace/F-03.md`
- 2025-10-14 F-04 tester: ベースラインスモークテスト構成 → 成果: `docs/results/F-04.md`, Trace: `docs/trace/F-04.md`

## 7. 運用フロー
1. planner がチケットを作成し、本表へ登録。orchestrator に共有。
2. orchestrator が coder/tester/judge へ指示を出し、着手時に状態を `In Progress` へ更新。
3. coder/tester が作業し、成果物を `docs/results` / `docs/trace` に保存。チケットファイルと overview のメモ欄を更新。
4. judge がレビューを実施し、結果をチケットと overview に反映。必要に応じてフィードバックや新規チケットを提案。
5. planner が完了判定を確認し、状態を `Done` に変更して履歴へ移動。実施結果を `docs/spec/implementation-tasks.md` に反映する。
