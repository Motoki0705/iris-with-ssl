**主要ポイント**
- マルチエージェント体制をTier1（orchestrator）、Tier2（planner/coder/judge）、Tier3（tester/researcher）の階層で定義し、指示と支援の流れを分離。
- orchestratorがチケット状態`state`を管理し、reportsを基にplanner・coder・judgeへ適切にディスパッチするのが中核的役割。
- plannerはSSOTである`requirements.md`を保守しつつticketsを発行し、必要時にresearcherを呼び外部情報で判断品質を高める。
- coderとtesterは実装・検証を分担し、成果（reports）と変更ログ（trace）を整備してjudgeの最終評価に備える。
- 役割間の受け渡し物と完了条件を明確化したハンドオフ契約により、チケット進行と意思決定を一貫管理する。
