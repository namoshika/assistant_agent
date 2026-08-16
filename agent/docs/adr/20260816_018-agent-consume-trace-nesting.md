# ADR-018: Agent._consume() 起因の実行で mlflow トレースのネストが深化する問題

- **日付**: 2026-08-16
- **ステータス**: 採択

## 背景

Discord・Dispatcher Channel 起因のエージェント実行（`Agent._consume()`、`src/assistant_agent/utils/workflow.py`）で、ツール呼び出しを伴うたびに mlflow トレースの span ネストが実行のたびに1段ずつ深くなっていく現象が確認された。
FastAPI 経由（`agent_server.py` の `LangGraphChatAgent.predict_stream_async`）の同種の実行ではこの問題が発生していない。

mlflow サーバー上の実トレースを比較したところ、`traceName: Agent` のトレースは**ツール呼び出しが0回なら正常、1回以上あると必ずネストする**ことを確認した。

- 正常（ツール呼び出しなし）: `Agent → SampleAgent → ... → model → ChatOpenAI`
- ネスト発生（ツール呼び出し2回）: `Agent → SampleAgent → ... → model → tools → TOOL → model → tools → TOOL → model`
  （本来 `SampleAgent` 直下の兄弟であるべき `model`/`tools` が、直前の `tools` の子として記録される）

Discord 実運用のトレース（ツール呼び出し5回）では5段階のネストが確認され、トレースサイズが約10MBに達していた。

## 原因

### 親 span の解決ロジック

`mlflow.langchain.langchain_tracer.MlflowLangchainTracer._get_parent_span()`（`mlflow/langchain/langchain_tracer.py`）は、親 span 候補を次の2系統から取得する。

1. `mlflow.get_current_active_span()`（contextvar 上の「現在アクティブな span」）
2. LangChain が渡す `parent_run_id` に対応する span（本来の親子関係）

両者が食い違う場合 `_resolve_parent_span()` が、「MLflow 側 span が LangChain 側 span の祖先として辿れなければ、LangChain 側（＝直前の兄弟 span）を親として採用する」フォールバックを行う。

### LangGraph のツール並列実行

LangGraph の `ToolNode._afunc`（`langgraph/prebuilt/tool_node.py`）はツール呼び出しを `asyncio.gather(*coros)` で実行する。
ツール呼び出しが1件でも `asyncio.gather` によって独立タスクとしてスケジュールされ、そのタスク内での span の attach（contextvar ベース）は当該タスクのコンテキストにのみ反映される。
これにより、ツールノード実行後に次の `model` ノードの span を作る際、`mlflow.get_current_active_span()` が「直前の tool span」を指したまま残留し、`_resolve_parent_span` のフォールバックによりそれが新たな親として採用される。
ループを繰り返すたびに1階層ずつ深くなる実測結果と一致する。

### `Agent._consume()` の実装（修正前）と `@mlflow.trace` の同型性

```python
with mlflow.start_span("Agent", span_type=SpanType.CHAT_MODEL) as span:
    ...
    result = await asyncio.wait_for(
        self._agent.ainvoke(**merged_invocation, config=config, version="v2"),
        timeout=timeout_seconds,
    )
```

mlflow の `@mlflow.trace`（通常の async 関数向け、`mlflow.tracing.fluent._wrap_function`）の内部実装は `with start_span(...): ... await fn(*args, **kwargs)` であり、上記と構造的に同一である。
したがって `@mlflow.trace` デコレータへの単純な書き換えでは本問題は解決しない。

### FastAPI 側（`predict_stream_async`）で発生しない理由

`predict_stream_async` は async generator であり、`@mlflow.trace` は `_wrap_generator`（`mlflow.tracing.fluent`）を使う。
この実装は `generator.__anext__()` を呼ぶ**その瞬間だけ** `with safe_set_span_in_context(span)` で span をアクティブにし、`yield` で呼び出し元へ制御を返している間は非アクティブに戻す。
mlflow の docstring にも「span should only be "active" at B, C, and E ... Otherwise it will create wrong span tree, or even worse, leak span context and pollute subsequent traces.」と明記されている。

## 対処の経緯（最初の対処が不十分だった理由）

最初の対処として、`Agent._consume()` 内の `ainvoke()` 呼び出しを、`self._agent.astream(...)` を `async for` で逐次消費する形に変更した。
「FastAPI 側は `astream` の `async for` を使っているからネストしない」という理解に基づく対処だったが、これは誤りだった。

```python
async def _run_stream() -> None:
    nonlocal msg_out
    async for _mode, chunk in self._agent.astream(
        **merged_invocation, config=config, stream_mode=["updates"]
    ):
        ...

with mlflow.start_span("Agent", span_type=SpanType.CHAT_MODEL) as span:
    await asyncio.wait_for(_run_stream(), timeout=timeout_seconds)
```

修正を反映したプロセスで実際にトレースを取得し検証したところ、ネストは解消していなかった。
`_wrap_generator` がネストを防いでいた本体は「`astream` を使っていること」ではなく、「`predict_stream_async` 自身が generator 関数であり、mlflow の `@mlflow.trace` がその `__anext__()` 呼び出し1回ごとに span を再アタッチ/デタッチしていること」だった。
上記の修正では `Agent` span を開く `with mlflow.start_span(...)` がループ全体を囲んだままであり、`_wrap_function` と同じ「span を開きっぱなしで await する」構造から変わっていなかったため、根本原因（`ToolNode` の `asyncio.gather` 後の contextvar 分岐）は解消されていなかった。

## `astream` 化と `@mlflow.trace` はそれぞれ独立に必要だった

`@mlflow.trace` デコレータを付けても、関数の中身が `ainvoke` 単発呼び出し（`yield` が最後に1回だけ）であれば、ネストは解消しないことを検証で確認した。

```python
@mlflow.trace(span_type=SpanType.CHAT_MODEL, name="Agent")
async def run_one(self, ...) -> AsyncIterator[dict]:
    result = await self._agent.ainvoke(...)  # ここで await が1回だけ
    yield {"result": result.value["messages"][-1].content}
```

この形で実行すると、`_wrap_generator` は `generator.__anext__()` を1回呼ぶだけで完了してしまい、その1回の呼び出し内部で `ainvoke` がツール呼び出しを何度繰り返そうと、span の再アタッチが起きる機会（generator へ制御が戻るタイミング）が一度も訪れない。
実際に検証したところ、ツール呼び出し2回を伴う実行で通常通りネストが発生した。

`_wrap_generator`（および同等の手動実装）がネストを解消できるのは、「関数呼び出しごとに span を作る」ことと「LangGraph の1ステップごとに `yield` して generator へ制御を戻す（＝`astream` で逐次消費する）」ことの**両方が揃って初めて**成立する。
`@mlflow.trace` デコレータは前者（span 制御の実装）を肩代わりしてくれるだけであり、後者（1ステップごとに区切る構造）は呼び出し側が `astream` を使う形で用意する必要がある。

## 対処の経緯（2回目）: 手動 span 制御は不要だった

`_wrap_generator` と同じ「1チャンク受信の瞬間だけ span をアクティブにする」制御を、mlflow の低レベル API（`start_span_no_context` + `safe_set_span_in_context`）を使って `Agent._consume()` 内に手動実装する対処を一度採用した。
`Agent` クラスを実グラフ・実 mlflow サーバーに対して直接実行するスクリプトで検証し、ツール呼び出し2回・3回を伴う実行でネスト解消を確認した（トレース `tr-70a199e9802f7c954d62affa6350c908`、`tr-7040e643a94a579ccc7dc4c5d3710070`）。

しかしこの実装は、mlflow が `@mlflow.trace` デコレータの内部（`_wrap_generator`）として既に提供している処理を、非公開 API（`mlflow.tracing.provider.safe_set_span_in_context`）を使って手動で再実装したものであり、`with mlflow.start_span(...)` の `__exit__` が保証していた「例外発生時に確実に span を閉じる」処理も自前で再現する必要があった（`try`/`except`/`finally` の分岐が複雑化する要因になった）。

`predict_stream_async` に `@mlflow.trace` を直接付けられるのは、HTTP リクエスト1件につき1回だけ呼ばれる関数だからである。
一方 `Agent._consume()` は `while True` で常駐し続けるループであり、1メッセージ＝1トレースという要件（ADR-017 参照）を満たすには、ループ全体ではなく「1メッセージ分の処理」を関数境界として区切る必要があった。
これを次の決定で解決した。

## 決定

1メッセージ分の処理（rollover・`astream` の逐次消費・span の入出力設定）を `_consume_one()` という非同期 generator メソッドへ切り出し、これに `@mlflow.trace` を直接付けた。
`_consume()` の `while True` ループは、新着メッセージを受け取るたびに `_consume_one()` を `async for` で呼び出すだけになる。

```python
async def _consume(self) -> None:
    while True:
        invocation = await self._queue.get()
        ...
        try:
            msg_out: AnyMessage | None = None
            async with asyncio.timeout(timeout_seconds):
                async for part in self._consume_one(invocation, received_context):
                    if updated := self._extract_last_message(part):
                        msg_out = updated
        except TimeoutError:
            ...
        except Exception:
            ...
        self.emit(AgentInvocation(input={"messages": [msg_out]}, context=received_context))

@mlflow.trace(span_type=SpanType.CHAT_MODEL, name="Agent")
async def _consume_one(
    self, invocation: AgentInvocation, received_context: CommonContext
) -> AsyncIterator[Any]:
    self._thread_id, config = await self._rollover_strategy.invoke(...)
    mlflow.update_current_trace(metadata={...})
    span = mlflow.get_current_active_span()
    span.set_inputs({"messages": [chat_msg_in]})
    async for part in self._agent.astream(
        **merged_invocation, config=config, stream_mode=["updates"], version="v2"
    ):
        yield part
    span.set_outputs({"messages": [chat_msg_out]})
```

- `mlflow.get_current_active_span()` は `@mlflow.trace` 配下（`_wrap_generator` が span をアタッチしている間）であれば、`_consume_one()` の先頭・末尾いずれでも同一の span オブジェクトを返す。
  `span.set_inputs`/`set_outputs`/`set_attribute` は手動実装版と同じ形で呼べる。
- `mlflow.update_current_trace(...)` も同様にデコレータ配下でそのまま呼べる（手動実装版で必要だった `safe_set_span_in_context` によるラップは不要）。
- タイムアウト処理・例外処理（`TimeoutError`/`Exception` を捕捉して `AIMessage` へ変換する）は `_consume()` 側に残す。
  `_wrap_generator` は例外発生時に span を `ERROR` にして再送出するだけで、独自のエラーメッセージへの変換は行わないため、この責務は呼び出し元が持つ必要がある。
- 1チャンクごとの `msg_out` 抽出ロジック（`part["type"] == "updates"` の判定と `node_output["messages"][-1]` の取得）は `_consume_one()`（span 出力用）と `_consume()`（emit 用）の双方で必要なため、`_extract_last_message()` という `@staticmethod` に切り出して重複を避けた。

`Agent` クラスを実グラフ・実 mlflow サーバーに対して直接実行するスクリプトで再検証し、ツール呼び出し3回を伴う実行で `model`/`tools` span が `LangGraph` 直下のフラットな兄弟として記録されること、`mlflow.trace.user`/`mlflow.trace.session` メタデータが正しく設定されること、reasoning ブロックを含む応答が `emit()` 時に維持されることを確認した（トレース `tr-815c0768868a4b333f01a321f7a845d2`、`state: OK`）。

## 影響・備考

- ADR-017（`docs/adr/20260807_017-dispatcher-trace-broken-by-span-ttl.md`）とは別の問題。
  ADR-017 は span TTL によるトレース分断への対処（span を開くタイミングをキュー待機の外に出す）であり、既に反映済み。
  今回のネスト問題はその修正後もツール呼び出しを伴う実行に限り残っていた。
- 本対処は mlflow・LangGraph 双方の非公開実装詳細（`ToolNode` の `asyncio.gather`、mlflow トレーサーの `_get_parent_span`/`_resolve_parent_span` のフォールバック、`_wrap_generator` の span 生存期間制御）に依存する。
  ただし採用した `@mlflow.trace` デコレータ自体は mlflow の公開 API であり、内部実装（`_wrap_generator`）の詳細に直接依存するコードは書いていない。
- **`asyncio.CancelledError`（`stop()` によるタスクキャンセル）発生時、`_wrap_generator` は `_end_stream_span()` を呼ばずにトレースの送出を打ち切る。**
  実機検証で、`_consume_one()` の `astream` 消費中に `stop()` を呼ぶと、該当トレースが mlflow サーバー上に `IN_PROGRESS` のまま記録され、`OK`/`ERROR` いずれの完了状態にもならないことを確認した。
  最初に採用しかけた `start_span_no_context` + `finally: span.end(...)` の手動実装であればこのケースでも `ERROR` 状態で確実に記録できたが、非公開 API 依存と実装の複雑さを避けるため、この既知のトレードオフを許容して `@mlflow.trace` 方式を採用した。
  `stop()` は通常 Ctrl+C 等のプロセス終了時のみ呼ばれ（ADR-013 参照）、処理中のトレースが記録されないことの実害は限定的と判断した。
- 「単体テストが通ったこと」を実機で直る根拠にせず、必ず実際にプロセスを起動してトレースを目視確認する、という検証手順の教訓を得た。

### 既知の未解決問題: サブエージェント内部のネスト

`deepagents.middleware.subagents`（`src/assistant_agent/subagents.py` で使用）の `task` ツール（`atask`）は、サブエージェントを `await subagent.ainvoke(subagent_state, subagent_config)` という単一の `await` で起動する。
この呼び出しは `Agent._consume()` の `astream` ループの**外側**（親グラフの `ToolNode._afunc` が `asyncio.gather` で実行する1コルーチンの内部）で行われるため、今回の対処（1チャンクごとの span 再アタッチ）の対象範囲に含まれない。

実際に `SubAgentMiddleware` を使い、親エージェントが `task` ツールでサブエージェントを1回呼び、サブエージェント内部でツールを2回呼ぶ構成を検証したところ、サブエージェント内部（`test-researcher` 以下）の `model`/`tools` は本対処の前後を問わず同一の形でネストしていた（本 ADR の修正が原因ではなく、既存の別問題）。

```
[CHAT_MODEL] Agent
  [CHAIN] LangGraph
    [CHAIN] model
    [CHAIN] tools
      [TOOL] task
        [CHAIN] test-researcher
          [CHAIN] model
            [CHAIN] tools
              [TOOL] sub_tool_a
              [CHAIN] model          <- ここ以降、サブエージェント内部でネストが再発する
                [CHAIN] tools
                  [TOOL] sub_tool_b
                  [CHAIN] model
    [CHAIN] model
```

親エージェントの `Agent` 直下（`LangGraph` 配下の `model`/`tools`）は正しくフラットになっており、「ネストすべき親子関係（`task` がサブエージェント全体の親であること）」が誤って潰される副作用は確認されなかった。
未解決なのは「サブエージェント内部の `model`/`tools` の連鎖」のみ。
対処するには `atask`（`deepagents` パッケージ側のコード）を同様に `astream` ベースへ変更する必要があるが、サードパーティパッケージの改修が必要になるため、本 ADR のスコープ外として記録に留める。

**FastAPI 経由（`predict_stream_async`）でも同一の問題が起きることを実証した。**
`atask` の `await subagent.ainvoke(...)` は、外側の消費方法（`astream` の `updates`/`messages` モードか `ainvoke` 単発か）によらず、完了するまで外側へ制御を一切返さない。
`_wrap_generator` の「`__anext__()` が呼ばれるたびに span を再アタッチする」制御は、外側の generator に制御が戻ってきたタイミングでのみ働くため、`ainvoke` の内部（サブエージェント実行中）には原理的に介入できない。

`predict_stream_async` と同一の構造（`@mlflow.trace(span_type=SpanType.AGENT)` を付けた関数が `astream(stream_mode=["updates"], version="v2")` を `async for` で逐次消費する）を再現し、`SubAgentMiddleware` 経由でサブエージェントを呼ぶ検証を行ったところ、`predict_stream_async` 相当の親レベル（`LangGraph` 配下の `model`/`tools`）はフラットなままだったが、`task` ツール配下（`test-researcher` 以下）は本 `Agent._consume()` の場合と同様にネストした（トレース `tr-c81e9702650094aab53fbda0e43b78d1`）。

なお `predict_stream_async` 自体は `stream_mode=["messages"]`（LLM トークン単位のストリーミング）を使っており、`GenericFakeChatModel` では tool_calls のみを持つ空 content の `AIMessage` をストリーミングできない制約（`ValueError: No generations found in stream.`）があるため、`predict_stream_async` の実装をそのまま fake モデルで動かすことはできなかった。
上記の検証は `stream_mode=["updates"]` を使う点のみ `predict_stream_async` と異なるが、ネスト解消の機序（`_wrap_generator` による span 再アタッチ）に関わる部分は同一のため、結論の一般性は損なわれない。
