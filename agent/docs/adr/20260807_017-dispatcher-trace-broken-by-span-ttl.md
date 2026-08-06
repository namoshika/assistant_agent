# ADR-017: Dispatcher Channel 起因の実行で mlflow トレースが分断される問題

- **日付**: 2026-08-07
- **ステータス**: 採択

## 背景

Dispatcher Channel（予約イベント）起因のエージェント実行で、mlflow トレースが正しく記録されない現象が確認された。該当トレースは span が `AsyncResponses`（OpenAI Responses API への生呼び出しを捉える `mlflow.openai.autolog()` の leaf span）1個のみとなり、本来その親であるはずの `Agent` / `SampleAgent` / `PatchToolCallsMiddleware.before_agent` / `model` / `ChatOpenAI` の階層が一切記録されていなかった。Discord Channel 起因の実行では同じ構成が正しく記録されており、問題は Dispatcher 起因の実行に限られていた。

mlflow 上で直近50件のトレースを時系列に並べ、span 構成と直前トレースからの経過時間を突合したところ、破損したトレースは必ず「直前のトレースから約2時間前後（7000秒台）経過した後」に発生しており、直後には正常な構成に復帰していた。

| 時刻 (JST) | root span | span数 | 直前との間隔 |
|---|---|---|---|
| 2026-08-07 09:52:17 | `Agent`（正常） | 21 | 117.7秒 |
| 2026-08-07 12:00:53 | `AsyncResponses`のみ | 1 | **7715秒（約2時間8分）** |
| 2026-08-07 12:00:59〜12:01:31 | `AsyncResponses`のみ ×5 | 1 | 4〜15秒間隔 |
| 2026-08-07 13:59:54 | `AsyncResponses`のみ | 1 | **7103秒（約1時間58分）** |
| 2026-08-07 14:00:04〜14:00:55 | `AsyncResponses`のみ ×6 | 1 | 6〜15秒間隔 |
| 2026-08-07 14:01:02 | `Agent`（正常） | 21 | 6.2秒 |

Discord・Dispatcher いずれの実行も `Agent._consume()`（`src/assistant_agent/utils/workflow.py`）という単一のコードパスを共有しており、呼び出し方法（`self._agent.ainvoke(...)`）に分岐は無い。両者の違いはイベント発火頻度のみで、Discord のメッセージ間隔は概ね数分〜数十分に収まるのに対し、Dispatcher の予定間隔は数時間空くことがある。

## 原因

`Agent._consume()` の実装（修正前）は次の構造になっていた。

```python
async def _consume(self) -> None:
    config: RunnableConfig = {"configurable": {"thread_id": self._thread_id}}
    while True:
        with mlflow.start_span("Agent", span_type=SpanType.CHAT_MODEL) as span:
            invocation = await self._queue.get()  # 新着が無ければここで長時間ブロックする
            ...
```

`mlflow.start_span("Agent", ...)` が `await self._queue.get()`（キューが空の間ブロックする待機処理）より外側にあり、新着イベントを待つ間もその待機時間ごと span（トレース）が開いたままの状態として計上されていた。

一方、mlflow のトレーシングSDKには、未完了のままクライアント側インメモリバッファに滞留したトレースを一定時間で破棄する TTL 機構が存在する（`.venv/.../mlflow/environment_variables.py`）。

```python
# How long a trace can be buffered in-memory at client side before being abandoned.
MLFLOW_TRACE_BUFFER_TTL_SECONDS = _EnvironmentVariable("MLFLOW_TRACE_BUFFER_TTL_SECONDS", int, 3600)
```

既定値は 3600秒（1時間）で、本プロジェクトではこの値を上書きしていない。

Dispatcher の予定間隔がこの TTL（1時間）を超えると、`with mlflow.start_span("Agent", ...)` によって開かれたまま待機し続けていたトレースがバッファから破棄（abandon）される。その後ようやくイベントが届いて `self._agent.ainvoke(...)` が実行されても、LangChain/deepagents 側の autolog（`mlflow.langchain.autolog(run_tracer_inline=True)`）が子 span を紐付けようとする親トレースは既に失われているため、`Agent` 以下の階層が記録されない。一方、`mlflow.openai.autolog()` が捉える最下層の生 API 呼び出し（`AsyncResponses`）は親の有無に関係なく独立して記録できるため、孤立した単一 span のトレースとして残る。

Discord は発火間隔が短く TTL を超えることが稀なため、この問題が顕在化していなかった。

## 決定

`invocation = await self._queue.get()` を `with mlflow.start_span("Agent", ...)` より前に移動し、span を開くタイミングを「新着イベントを実際に受信し、処理を開始する瞬間」に変更した。

```python
async def _consume(self) -> None:
    config: RunnableConfig = {"configurable": {"thread_id": self._thread_id}}
    while True:
        invocation = await self._queue.get()
        with mlflow.start_span("Agent", span_type=SpanType.CHAT_MODEL) as span:
            ...
```

これにより、キュー待機中の時間が span/トレースの生存期間に含まれなくなり、Dispatcher の発火間隔が TTL（3600秒）を超えていても、トレースは処理開始時点から新規に生成されるため破棄の影響を受けない。

## 影響・備考

- 副次的な効果として、`stop()`（`self._task.cancel()`）が `await self._queue.get()` の待機中に呼ばれた場合、修正前は空の `Agent` span がキャンセル状態のまま記録されていたが、修正後はそもそも span が開かれないため記録されなくなる。
- `MLFLOW_TRACE_BUFFER_TTL_SECONDS` 自体の値は変更していない（既定3600秒のまま）。ロールオーバー処理や `ainvoke()` 自体の実行時間が TTL を超えるケースは理論上残るが、通常は数秒〜数十秒で完結するため実用上のリスクは低いと判断した。
- 本 ADR は原因調査と対処の記録のみを目的とし、要件定義・設計ドキュメント（`docs/experimental/`）は作成していない。
