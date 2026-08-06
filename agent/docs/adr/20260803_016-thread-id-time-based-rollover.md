# ADR-016: thread_id の時間駆動ロールオーバー（ADR-015 への例外追加）

- **日付**: 2026-08-03
- **ステータス**: 採択

## 背景

ADR-015 により、`Agent`（`utils/workflow.py`）は単一の `thread_id` を内部管理し、外部から渡される `AgentInvocation` の内容に一切左右されない設計となっている。常駐エージェントは単一 `thread_id` で長期間稼働し続けるが、これにより次の問題が生じる。

- LangGraph の `DeepAgentState.messages` は `DeltaChannel`（差分連鎖・祖先チェーンで状態復元する仕組み）を使っており、checkpoint の連鎖は「差分が数珠つなぎになったチェーン」になる。チェーン途中の古い checkpoint 行を削除すると、それより新しい checkpoint の状態復元が壊れるため、削除できるのは常にチェーンの末端（＝ thread 全体）のみ
- `deepagents.create_deep_agent()` が既定で組み込む `SummarizationMiddleware` はモデルへ送るメッセージのみを圧縮し、`state["messages"]`（checkpoint に保存される生メッセージ列）自体は書き換えない。したがって要約が起きても DB 上の checkpoint データ量は際限なく増え続ける
- 結果として、単一 `thread_id` 固定運用のままでは、thread を区切って丸ごと削除する（`adelete_thread` 相当の操作）以外に checkpoint 肥大化を止める手段がない

調査の詳細は `docs/experimental/20260803_deepagents_migration/research.md` を参照。

## 決定

ADR-015 の核心（`thread_id` は `Agent` が内部管理し、外部から渡される `AgentInvocation` の内容に左右されない）は維持したうえで、**`Agent` 自身の判断で、`thread_id` 生成からの経過時間のみを基準に、無条件に新しい `thread_id` へロールオーバーしてよい**という例外を追加する。

- ロールオーバーの発火条件は `thread_id` に埋め込まれた `uuid7` 部分の生成時刻からの経過日数のみとし、`deepagents` の圧縮イベント（`_summarization_event`・`cutoff_index`）には一切依存しない。圧縮検知という関心事と thread 世代交代という関心事を分離する
- ロールオーバー時は旧 `thread_id` の会話全体を `Agent` 自身が要約し、生ログを仮想ファイルシステム（`StoreBackend`）へ退避したうえで、要約を新 `thread_id` の会話冒頭へ引き継ぐ
- 旧 `thread_id` の checkpoint 自体の削除は `Agent` の責務に含めず、運用者が手動実行するメンテナンスノートブック（`scripts/04_maintenance.ipynb`）に委ねる

`thread_id` は `f"{agent_id}:{uuid.uuid7()}"` の形式で払い出す。`agent_id` を prefix に含めることで、複数エージェントが同一 checkpointer・store を共有しても、起動時の直近 thread_id 検索やメンテナンスノートブックでのグルーピングをエージェント単位で絞り込める。

## 影響・備考

- 詳細な設計・実装記録は `docs/experimental/20260803_deepagents_migration/` を参照
- ADR-015 本文（`docs/adr/20260731_015-agent-thread-id-ownership.md`）は改訂しない。本 ADR はその例外追加として扱う
- 対象は常駐エージェント（`agent_bot.py`）のみ。mlflow serving 側（`pack.py`）は本タスクの対象外（`agents.build_agent()` 廃止に伴い動作を保留）
