# ADR-010: mlflow AgentServer を使用せず独自の OpenAI Chat Completions API 互換サーバーを作成

- **日付**: 2026-03-28
- **ステータス**: 採択

## 背景

エージェントの動作確認・試験用に、GitHub 等で公開されているチャットボット OSS（Open WebUI 等）から
接続できる Playground 環境を設けたい。

### mlflow AgentServer の制約

mlflow には `AgentServer` が用意されており、エージェントをサーバーとして起動できる。
しかし、既存のチャットボット OSS はほぼすべて OpenAI Chat Completions API に準拠して設計されており、
AgentServer はこの API と互換性がない。

| 項目 | mlflow AgentServer | OpenAI Chat Completions API |
|---|---|---|
| エンドポイント | `POST /invocations` | `POST /v1/chat/completions` |
| メッセージ形式 | `ChatAgentRequest` / `ChatAgentResponse` | OpenAI 固有スキーマ |
| モデル一覧 API | なし | `GET /v1/models` |

スキーマについては mlflow の型（`ChatAgentMessage` 等）が OpenAI の型と部分的に互換を持つ設計になっているが、
エンドポイントの形式や必要な API が揃っていないため、チャットボット OSS からそのまま接続することはできない。

### API 形式の選択

現時点で採用する API 形式として、以下を比較した。

| 形式 | 説明 | 採用 |
|---|---|---|
| Chat Completions API | OpenAI の旧来の API。ほぼすべてのクライアントが対応しており実績豊富 | ✓ |
| Responses API | OpenAI の新 API。オープン化されており将来的に主流になる見込み | 将来対応 |

現時点では実績・互換性を優先し、Chat Completions API を採用する。
Responses API は OSS クライアントの対応が揃い次第、将来的に追加対応する。

## 決定

mlflow AgentServer を使用せず、FastAPI を用いて独自の OpenAI Chat Completions API 互換サーバーを実装する。

### 実装方針

- `LangGraphChatAgent`（mlflow の `ChatAgent` サブクラス）を受け取り、既存の `FastAPI` インスタンスへルートを追加する
- mlflow が提供する型（`ChatAgentMessage`, `ChatUsage`, `ChatMessage`, `TextContentPart`）を最大限転用し、独自クラス定義を最小化する

### 現行実装の OpenAI API との差異

詳細は `docs/experimental/20260327_apiserver/design.md` を参照。主な差分は以下のとおり。

- `temperature`, `max_tokens` 等のパラメータは受け付けるがエージェントへは渡さない
- `choices` は常に 1 件のみ
- `usage` はエージェントが返す場合のみ含まれる（省略可）
- `finish_reason` は `"stop"` 固定

## 影響・備考

- Responses API への対応時は、同様のアダプター層を追加することで対応できる見込み
- Open WebUI との疎通は `docs/experimental/20260327_apiserver/poc/` の PoC で確認済み
