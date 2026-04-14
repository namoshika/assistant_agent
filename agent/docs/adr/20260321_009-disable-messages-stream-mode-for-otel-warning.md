# ADR-009: stream_mode="messages" を無効化して OpenTelemetry 警告を抑制

- **日付**: 2026-03-21
- **ステータス**: 採択（暫定）

## 背景

### 警告の内容

`scripts/hoge.py`（`mlflow.langchain.autolog()` + `predict()` 呼び出し）の実行時に
以下の警告がトークン数分だけ繰り返し出力され、ノートブックの可読性を著しく低下させていた。

```
Invalid type dict in attribute 'token' value sequence.
Expected one of ['bool', 'str', 'bytes', 'int', 'float'] or None
```

この警告は OpenTelemetry (`opentelemetry/attributes/__init__.py`) が
スパン属性値のシーケンス内に `dict` 型を検出したときに出力する。

同様の現象が 2025/04 に報告 ([GitHub Issues](https://github.com/mlflow/mlflow/issues/15330)) されているが、未だ未解決。

### 因果関係

以下の連鎖により発生する。

1. **`stream_mode=["messages"]` の指定**
   `LangGraphWrapper.predict_stream` が `self._agent.stream(..., stream_mode=["updates", "messages"])` を呼ぶと、
   LangGraph (`langgraph/pregel/main.py`) が `StreamMessagesHandler` を `run_manager` の
   callback として登録する。

2. **LLM のストリーミングモードへの強制切り替え**
   `StreamMessagesHandler` は `_StreamingCallbackHandler`
   (`langchain_core/tracers/_streaming.py`) のサブクラスである。
   `BaseChatModel._should_stream()` (`langchain_core/language_models/chat_models.py`) は
   handlers に `_StreamingCallbackHandler` が含まれると `True` を返す。
   このため、エージェント内部のモデルノード (`langchain/agents/factory.py`) が
   `model_.invoke()` を呼んでいても、LLM は `_stream()` (= `converse_stream`) で実行される。

3. **`on_llm_new_token` へ `list[dict]` が渡される**
   `BaseChatModel.stream()` は各ストリーミングチャンクに対して
   `run_manager.on_llm_new_token(chunk.message.content, chunk=chunk)` を呼ぶ。
   ChatBedrockConverse (Claude) はコンテンツをテキスト・ツール呼び出し等の
   content blocks (`list[dict]`) で返すため、`token` 引数に `list[dict]` が渡される。

4. **MLflow トレーサーが OTel スパン属性として登録**
   MLflow の `langchain_tracer.py` は `on_llm_new_token` を受け取り、
   `SpanEvent(attributes={"token": token})` として OTel に渡す。
   `token` の値が `list[dict]` のまま文字列変換されないのが mlflow 側のバグである。

5. **OpenTelemetry が警告を出力**
   OTel の `_clean_attribute()` (`opentelemetry/attributes/__init__.py:80-89`) は
   シーケンス要素が `_VALID_ATTR_VALUE_TYPES = (bool, str, bytes, int, float)` 以外の場合に
   警告を出力して属性を破棄する。

### 発生バージョン確認時の環境

- mlflow 3.10.1
- langchain-core (LangChain v0.3 系)
- langchain-aws (ChatBedrockConverse)
- opentelemetry-sdk

## 決定

`src/assistant_agent/utils/mlflow.py` の `predict_stream` にて、
`stream_mode` から `"messages"` をコメントアウトして除外する。

現状 `predict_stream` で `"messages"` モードの出力（トークン単位のデルタ）は
使用されていないため、機能への実質的な影響はない。

## 影響・備考

- mlflow の upstream バグ修正後に `stream_mode=["updates", "messages"]` へ戻す。
- 関連する暫定対処: ADR-007（mlflow の複数コンテンツ対応制限への対処）
