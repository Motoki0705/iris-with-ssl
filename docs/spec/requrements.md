# プロジェクト要件（SSOT）

DOC-ID: RQ-<project>-<nnn>  
Version: 0.1.0  
Owner: planner (@<name>)  
Status: Draft  <!-- Draft | Active | Deprecated -->
Last-Updated: YYYY-MM-DD
<!-- 本書は要件の Single Source of Truth（SSOT）です。変更は必ずチケット経由で反映し、他文書は本書を参照します。 -->

## 1. 目的と範囲（Purpose & Scope）
- 目的: <なぜ実施するか（1–3 行）>
- スコープ: <対象データ/機能/成果物>
- 非スコープ: <含めないもの>
- ステークホルダー: <orchestrator/reviewer, planner, coder, tester, judge>
  <!-- 役割の詳細や手順は agents.md を参照（ここでは列挙のみ） -->

## 2. データと実行環境（Data & Environment）
- データセット:
  - <path/to/data1.csv>（checksum: <sha256>）
  - <path/to/data2.csv>（checksum: <sha256>）
- Python/依存: Python <3.11> / <pandas, scikit-learn, ...>
- 決定性: seed=<42>, split=<holdout/k-fold>, preprocessing=<scaler/imputer 等>
- 設定の正典: `configs/<project>/*.{yml,json}`
  <!-- 実行条件は原則 configs/ で一元管理し、コード直書きを避ける -->

## 3. 機能要件（FR）
<!-- 必要な数だけ以下のセクションを複製して使います。1 要件 = 1 ID。 -->
### FR-<area>-<id>: <要件タイトル（簡潔に）>
- 背景/根拠: <Why（1–2 行）>
- 入力: <データ/設定/前提条件>
- 処理: <主要ステップの要点（箇条書き）>
- 出力: <生成物と形式（CSV/PNG/MD 等）>
- 受入基準（Acceptance）:
  - <指標と閾値／必須図表／保存先>  <!-- 例: accuracy ≥ 0.90、混同行列 PNG を保存 -->
- 成果物（Artifacts）:
  - `docs/results/<ticket-id>.*`（表・図・要約）
  - `docs/trace/<ticket-id>.md`（実行条件：日時・コマンド・git hash）
- リンク: Tickets[<ID,...>] / Modules[<src/...>] / Tests[<tests/...>]

## 4. 非機能要件（NFR）
- 再現性:
  - 乱数・分割・前処理を固定し、設定値は `configs/` に保存
  - 環境（Python/ライブラリ/OS）を明記
- ログ/トレース:
  - 指標・ハイパラ・図表は `docs/results/` に保存
  - 実行条件（日時・コマンド・git hash）は `docs/trace/` に保存
- コード品質（任意）: 型/formatter/linter の方針
- パフォーマンス（任意）: 最大実行時間/メモリの目安

## 5. 指標と評価（Metrics & Evaluation）
- 指標: <Accuracy | MSE | R2 | MAE ...>  <!-- 必要なら定義や式を記載 -->
- 評価手順: <holdout/k-fold、反復回数、乱数固定、可視化必須物（学習曲線/残差図 等）>
- レポーティング:
  - 表: `docs/results/tables/<name>.csv`
  - 図: `docs/results/figures/<name>.png`

## 6. 検証計画（Verification & Validation）
- スモーク: <短時間で I/O と主要指標を確認>  
  例）`pytest -k "<keyword>"` / `python scripts/<run_xx>.py --smoke`
- 回帰: <基準 CSV/JSON と一致、または許容差内>
- レビュー: judge が `docs/results` と `docs/trace` を確認し合否判定
  <!-- レビュー体制や承認手順の詳細は agents.md を参照 -->

## 7. 成果物と命名規約（Artifacts & Naming）
- Results: `docs/results/<ticket-id>.*`（Markdown/CSV/PNG など）
- Trace: `docs/trace/<ticket-id>.md`
- （任意の細分）:
  - `docs/results/tables/`（集計表）
  - `docs/results/figures/`（図）

## 8. トレーサビリティ（Traceability Matrix）
| RQ-ID | Ticket-ID | Module (src/) | Test (tests/) | Results (docs/results) | Trace (docs/trace) |
|------|-----------|---------------|---------------|-------------------------|--------------------|
| FR-… | <T-…>     | <path.py>     | <test_x.py>   | <tables/…/figures/…>    | <T-….md>          |

## 9. 変更管理（Change Control）
- 変更要求: <起票先（チケット）/ レビュア（judge）/ 承認フロー>
- バージョン管理: 本書の更新履歴を冒頭に追記し、関連チケットへリンク
- 同期: 承認後、`implementation-tasks.md` など派生文書を同期

## 10. 未決事項（Open Issues）
- [ ] <未決1>（担当: <name> / 期限: YYYY-MM-DD）
- [ ] <未決2>（担当: <name> / 期限: YYYY-MM-DD）
