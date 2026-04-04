# ADR-003: `agent.py` を関心毎に分割

- **日付**: 2026-03-10
- **ステータス**: 採択

## 背景

`agent.py` に LLM 設定・ツール定義・グラフ構築・コンテキスト初期化が混在し肥大化していた。
このままでは今後の機能拡張や、各コンポーネントの独立したテストが困難であるため、関心の分離 (SRP) を行った。

## 決定

`agent.py` を以下の 3 モジュールに分割した:

| ファイル | 責務 |
|---|---|
| `connector.py` | LLM 取得 (`get_llm`) とツール定義 (`get_tools`) |
| `context.py` | `ContextSchema` の定義と `build_context()` によるセッション初期化 |
| `graph.py` | `build_graph()` によるエージェントグラフ構築 |

- ツールがグローバル変数でストアを参照していた設計を廃止し、`ToolRuntime[ContextSchema]` 経由でコンテキストを受け取るパターンへ移行した
- `agent.py` は 9 行に縮小し、3 モジュールを組み合わせて `mlflow.models.set_model()` に登録するだけの薄いエントリポイントとした

## 影響・備考

- `connector.py` / `context.py` / `graph.py` はそれぞれ独立してユニットテスト可能になった
- MLflow の `log_model` に渡すエントリポイントは引き続き `agent.py` のまま変更なし
