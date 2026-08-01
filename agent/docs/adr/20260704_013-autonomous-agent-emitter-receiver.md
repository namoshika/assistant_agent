# ADR-013: 自律型エージェント基盤（Emitter/Receiver/Agent/BroadcastPipe）の導入

- **日付**: 2026-07-04
- **ステータス**: 採択

## 背景

これまでの `agent_server.py`（FastAPI 経由でリクエスト駆動する ChatAgent）に加え、外部イベント（Discord メッセージ、定期実行トリガー等）を起点に自律的に動作する常駐エージェントが必要になった。要件は次の3点。

- 複数の `Agent` を同一プロセス内で並行に動かせる
- 1つの発信元の出力を複数の `Agent` へブロードキャストできる
- `Agent` の応答をさらに別の `Agent` が購読し、連鎖させられる

これらを満たす疎結合な配線の仕組みと、それを使って Discord Bot として常駐するエントリーポイント `agent_bot.py` をあわせて設計した。`agent_server.py` とはプロセスを分離し、リクエスト駆動と常駐駆動の2つの実行形態を共存させる。

## 決定

### アーキテクチャ

「発信できる」（`Emitter`）・「受信できる」（`Receiver`）・「能動的に動作できる」（`ActiveEmitter`）という独立した性質を組み合わせて各クラスを構成する。メッセージは`AgentInvocation`（`input`・`config` を持つ TypedDict、`utils/absclass.py`）で統一して受け渡す。

```
Emitter（発信の共通基底: 購読者を1つ登録し、AgentInvocation を配信できる）
Receiver（受信の共通基底: 外部から AgentInvocation を1件受け取る公開インターフェースを持つ）

ActiveEmitter（Emitter を継承。起動・停止を持つ）
├── CronChannel（一定間隔でトリガーメッセージを発信する）
└── DiscordChannel（ActiveEmitter・Receiver の両方を継承。Discord の発受信を配線する）

Agent（ActiveEmitter・Receiver の両方を継承する具象クラス。LangGraph のグラフを実行する）

BroadcastPipe（Receiver を継承。Emitter 1つと Receiver 複数を1対nでつなぐ。要素数1で1対1接続にも使う）
MergePipe（Receiver を継承。Emitter 複数と Receiver 1つをn対1でつなぐ）
LogWriter（Receiver を継承。受け取ったメッセージを logging 経由で記録する）
```

| クラス | 役割 | 継承する性質 |
|---|---|---|
| `Emitter` | メッセージを発信できるものの共通基底 | - |
| `Receiver` | メッセージを受信できるものの共通基底 | - |
| `ActiveEmitter` | 起動・停止を持つ発信元の共通基底 | `Emitter` |
| `Agent` | 入力を受けて応答を発信する処理単位。LangGraph のグラフを実行する | `ActiveEmitter`・`Receiver` |
| `BroadcastPipe` | `Emitter` 1つと `Receiver` 複数を1対nでつなぐ | `Receiver` |
| `MergePipe` | `Emitter` 複数と `Receiver` 1つをn対1でつなぐ | `Receiver` |
| `LogWriter` | 受け取ったメッセージを logging 経由で記録する | `Receiver` |

1対1接続専用の `Pipe` クラスは別途設けなかった。`BroadcastPipe(src, [dst])`（要素数1の `list[Receiver]`）で表現できるため、独立したクラスを持つ必要がないと判断した。

### `Agent` の実行モデル

- 受け取った新着を1件ずつ順に処理し、同一インスタンス内で並行処理しない（`asyncio.Queue` + 単一の消費タスクで直列化する）
- ただし LLM・ツール呼び出し中に他の `Agent` の処理はブロックされない（`asyncio.create_task()` で各 `Agent` が独立したタスクを持つ）
- 実行するグラフ（`lc_agent`）はコンストラクタで外部から注入され、`Agent` 自身はグラフの構築方法に関与しない
- `ainvoke()` 実行中の例外はループを継続させ、後続メッセージの処理を妨げない。例外発生時は当該インスタンスの識別情報（`thread_id`）と mlflow のトレースIDを含めて `logging` 経由で記録する

### 定期実行（`CronChannel`）

外部ライブラリ（`apscheduler` 等）を使わず、標準ライブラリの `asyncio.sleep()` によるループで実現する。`Agent._consume()` と同じ `asyncio.create_task()` ベースの起動・停止パターンを踏襲する。

```python
async def _run(self) -> None:
    while True:
        await asyncio.sleep(self._interval_seconds)
        self.emit(self._trigger_message)
```

### Discord 連携

- `discord.py`（PyPI: `discord-py`）を採用。Python 向け Discord API ラッパーの事実上の標準であり、非同期ベースの API が `ActiveEmitter` 設計と親和的
- `DiscordChannel` は `ActiveEmitter`・`Receiver` の両方を継承し、`CronChannel` からの受信と Discord サーバーへの展開ロジックを1クラスに集約した。`BroadcastPipe` は「同じメッセージを固定件数の `Receiver` へ配る」役割のみを持たせ、「1件の受信を実行時に決まる可変件数（参加サーバー数）へ展開する」役割は持たせない設計とした
- サーバー（guild）ごとの会話を分離するため、`thread_id` は `guild_id` から `uuid.uuid5(uuid.NAMESPACE_URL, f"discord-guild:{guild_id}")` で決定的に生成する方式とした（`guild_id` をそのまま使わず `discord-guild:` プレフィックスで名前空間を分離し、将来追加され得る他の発信元との `thread_id` 衝突を避けた）
- Bot Token（`AA_DISCORD_BOT_TOKEN`）・結合テスト対象チャンネル（`AA_DISCORD_CHANNEL_ID`）は既存の環境変数命名パターン（`ENV_*`/`AA_*`）に合わせた

### `agent_bot.py`（常駐エントリーポイント）

- `agent_server.py`（`fastapi-cli` 経由で起動する ASGI アプリ）とは別に、`[project.scripts]`（`solbot = "assistant_agent.agent_bot:main"`）で独立プロセスとして起動できるようにした。内部は同期の `main()` から `asyncio.run()` を呼ぶ構成とし、`agents/__init__.py:build_agent()` が同期関数であることと合わせた
- 配線は `CronChannel → DiscordChannel → Agent`（`BroadcastPipe` で接続）とし、`Agent` の応答は `BroadcastPipe` で `LogWriter` へ渡す。ログ出力先は環境変数 `ASSISTANT_AGENT_LOG_PATH`（未指定時 `logs/solbot_history.log`）
- `Agent` のコンストラクタには `agents/__init__.py:build_agent()` で組み立てたコンテキスト（`CommonContext`）を注入する。`agent_server.py`（`LangGraphChatAgent`）と同じ注入パターンを踏襲した
- `ScheduleService`（`CronChannel` を保持）は当初 `ContextRegistry` 経由での取得を想定していたが、他のコンテキスト項目に一切依存しないため `agent_bot.py` 内で直接構築する形に変更した（`ContextRegistry.register()` は不採用）
- Ctrl+C（`KeyboardInterrupt`）時は `CronChannel`・`DiscordChannel`・`Agent` を `stop()` し、終了コード130で終了する

## 影響・備考

- `AgentInvocation` は当初 `input`・`config` を必須キーとする設計だったが、後日ADR-015（`docs/adr/20260731_015-agent-thread-id-ownership.md`）により `config` は除去され、`thread_id` は `Agent` の内部管理へ回帰した。Discord のスレッド分離（本 ADR の決定事項）もあわせて撤去されている。本 ADR は当時の設計判断の記録として残す
- 本決定の詳細な要件・調査記録は `docs/experimental/20260704_autonomous_agent/` を参照
- `Emitter`/`Receiver`/`BroadcastPipe`/`MergePipe` の疎結合な配線パターン自体は、ADR-015 以降も変わらず踏襲されている
