# ADR-015: Agent の thread_id を内部管理に回帰

- **日付**: 2026-07-31
- **ステータス**: 採択

## 背景

Discord 連携機能の実装(コミット `eef7bda`)により、`Agent`(`utils/workflow.py`)はコンストラクタで受け取った固定 `thread_id` を使う方式から、`on_received()` で受け取る `AgentInvocation["config"]` の `thread_id` を都度使う方式へ変更されていた。これは `DiscordChannel` が Discord のサーバー(ギルド)ごとに異なる `thread_id` を生成し、単一の `Agent` インスタンスで複数サーバーの会話をスレッド分離するための仕組みだった。

その後の `DispatcherService`/`DispatcherChannel`(予定発信機能)の設計・実装(`docs/experimental/20260729_dispatcher_service/`)でも、この「config 経由で thread_id・guild_id・channel_id を引き継ぐ」前提の上に、予定をスレッド単位でフィルタする機能(登録済み予定の一覧・解除を呼び出し元スレッドのものに限定する)が正式要件として積み上げられた。

しかし、今後 `DiscordChannel` 以外の入力チャンネル(例: Gmail 連携)を追加する計画に対し、この設計には次の問題があった。

- 外部から渡される `AgentInvocation` の内容(config の有無・値)次第で `Agent` の会話コンテキスト(thread_id)が変わってしまい、チャンネルを跨いだ連続的なコンテキストを維持できない
- 新しいチャンネルを追加するたびに、そのチャンネルが「config にどんな thread_id を設定するか」を意識する必要があり、チャンネル実装と `Agent` の内部状態管理が密結合してしまう

## 決定

`thread_id`(および LangGraph の `RunnableConfig` 全般)を、`Agent` が自身のコンストラクタ引数として保持する内部管理領域とし、外部から渡される `AgentInvocation` の内容に一切左右されない設計に戻す(`eef7bda` 以前の方式への回帰)。

これに伴い、config 経由で thread_id・guild_id・channel_id を運搬していた仕組みを一貫して撤去した。

| 変更対象 | 内容 |
|---|---|
| `AgentInvocation`(`utils/absclass.py`) | `config` メンバーを完全に除去し、`input` のみを持つ型にする |
| `Agent`(`utils/workflow.py`) | コンストラクタで `thread_id: str \| None = None` を受け取り、`_consume()` は常にこの固定値から組み立てた config でグラフを呼ぶ |
| `DiscordChannel`(`services/discord.py`) | `Receiver` の実装(`on_received()`)を廃止。guild ごとに thread_id を割り振り直す展開処理は、Agent が単一 thread_id で動く以上不要になったため |
| `DispatcherService`(`services/dispatcher.py`) | thread_id によるスレッド単位の予定フィルタ機能(`cancel_dispatch`/`list_dispatch` の `thread_id` 引数)を廃止 |
| `tools/dispatcher.py` | ツールが呼び出し元 config から thread_id・guild_id・channel_id を継承するロジックを廃止 |

guild_id・channel_id(発信先の特定に必要な情報)は、config ではなく `DiscordChannel` が発信するメッセージ本文の先頭にメタデータブロックとして埋め込む方式に変更した(`guild_id: {guild_id}\nchannel_id: {channel_id}\n---\n{本文}`)。LLM が本文からこれを読み取り、`tools/discord.py` のツール呼び出し(`channel_id` 引数)に使う。

今後複数チャンネル間で会話コンテキストを分離したい場合は、`Agent` を複数インスタンス生成しチャンネル側からルーティングする方式で対応する方針とする(本 ADR のスコープ外)。

## 影響・備考

- `docs/experimental/20260729_dispatcher_service/requirements.md` の R004・R005・R019・R021・R025・R027(thread_id ベースのスレッド分離・予定フィルタに関する要件)は、本決定により廃止された。当該ファイルに廃止注記を追記済み
- 本変更の詳細な設計・実装記録は `docs/experimental/20260731_agent_thread_id_rollback/` を参照
- Discord の複数サーバー対応は、当面テナント分離されない(すべてのサーバーの会話が単一の `Agent` インスタンス・単一 thread_id に統合される)。マルチテナント化が必要になった場合は、`Agent` の複数インスタンス化とチャンネル側のルーティングで対応する(将来課題)
- ADR-016(`docs/adr/20260803_016-thread-id-time-based-rollover.md`)により、本決定の核心(外部から渡される内容に左右されない)は維持したまま、「`Agent` 自身の判断で、`thread_id` 生成からの経過時間を基準に新しい `thread_id` へロールオーバーしてよい」という例外が追加された
