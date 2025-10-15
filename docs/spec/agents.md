# エージェント責任とワークフロー

## 1. 役割と責任
- **orchestrator / reviewer（ユーザー）**
  - planner の指示および `docs/tickets/overview.md` を参照しつつ、各エージェント（coder/tester/judge）へ具体的な作業依頼を出す。
  - reviewer として judge と協調し、成果物が仕様を満たしているか確認し、必要に応じてフィードバックや追加チケットを発行する。
  - チケットの状態更新が必要な場合は planner と連携し、overview に反映されるよう調整する。

- **planner**
  - `docs/spec/requrements.md` を正典として全体進行を管理し、タスクをチケット化する。
  - チケットには目的・完了条件・成果物形式（`docs/results` / `docs/trace`）を明記し、`docs/spec/implementation-tasks.md` と同期させる。
  - チケットは `docs/tickets/<ticket-id>.md` に作成し、`docs/tickets/overview.md` に担当順序と履歴を追記する。
  - 依存関係や優先度を調整し、各エージェントの進捗を把握して次のアクションを提示する。
- **coder**
  - planner のチケットを受け取り、仕様に沿って実装・修正・再構成を行う。
  - 変更内容を `docs/trace` に記録し、成果（コード、図、指標など）の要約をチケット名と紐づけて `docs/results` に保存。
  - チケットファイルへ成果概要・保存先を追記し、overview の状態欄を更新する。
  - 実装中に仕様の不明点があれば planner にフィードバックし、仕様調整を依頼する。
- **tester**
  - planner のチケットに基づき、テスト戦略の立案・スクリプト作成・スモーク/回帰テストの実行を担当。
  - テスト結果や再現手順を `docs/results` にまとめ、実行ログ・設定値を `docs/trace` に記録。
  - チケットファイルと overview に進捗・結果・レビュー待ちステータスを記録する。
  - 問題が見つかった場合はチケットに対して報告し、必要なら逆チケット（フォローアップ）を planner に提案する。
- **judge**
  - `docs/results` と `docs/trace` の成果を確認し、`docs/spec` の要件（要件、アーキテクチャ、タスク）を満たしているか評価。
  - 判定結果と改善事項をチケットに記録しつつ、評価サマリを `docs/judges/<ticket-id>.md` にレポートとして残す。
  - 完了と判断したチケットをクローズし、成果を次フェーズへ反映させる。差し戻しやフォローアップが必要な場合は planner に報告し、対応チケットを提案する。

## 2. ワークフロー
1. planner がチケットを発行し、対象エージェント（coder/tester）と成果物要件を指定する。
2. coder/tester はチケットを受領して作業を実施。成果をまとめたファイルを `docs/results/<ticket-id>.*` などの命名で配置し、変更点やログを `docs/trace/<ticket-id>.md` 等に記録。
3. 作業完了後、チケットに成果物パスと要約を報告し、レビュー待ち状態へ移行。
4. judge が `docs/results` と `docs/trace` を確認し、`docs/spec` 群に照らして要求を満たすか評価。合否・改善点を `docs/judges/<ticket-id>.md` に記録し、問題があればチケットにコメント、必要な再作業を planner にフィードバック。
5. judge の承認後、planner がチケットをクローズし、必要なドキュメント更新（タスクチェックリストの反映など）を行う。

## 3. 記録ルール
- `docs/results`: チケット単位のアウトプット。図・表・指標サマリ・考察などを Markdown/PNG/CSV で保存し、冒頭にチケットIDと担当者、実行日を記載する。
- `docs/trace`: 実行ログ、設定値、学習パラメータ、変更ファイル一覧など追跡情報を残す。再現性確保のためコマンド履歴や Git ハッシュも記録する。
- `docs/judges`: judge 判定の公式ログ。各レビューは `docs/judges/<ticket-id>.md` に保存し、判定日、承認/差戻しステータス、主要指摘、フォローアップ要求を記載する。
- チケット更新時は planner が `docs/spec/implementation-tasks.md` のチェックボックスを最新状態に反映し、進捗可視化を行う。

## 4. コミュニケーション
- 仕様の不確実性や課題が発生した場合は即時 planner にエスカレーション。
- reviewer（judge）からの指摘は、対応担当者（coder/tester）が新チケットまたは既存チケットの再オープンとして処理する。
- 定期的に planner が進捗レビューを行い、必要に応じて優先度・担当の再割り当てを行う。
