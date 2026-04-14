# ADR-011: `assistant_agent` パッケージのモジュール構成

- **日付**: 2026-04-05
- **ステータス**: 採択
- **置き換え**: ADR-003 (agent.py の関心分離)

## 背景

ADR-003 では `agent.py` を `connector.py` / `context.py` / `graph.py` の 3 モジュールに分割したが、
その後の機能追加（Obsidian Vault レトリーバー・FastAPI サーバー・ローダー層など）により構成が大きく変化した。
現状のモジュール構成を ADR として記録し、設計判断の根拠を明示する。

## 決定

### パッケージ構成

```
src/assistant_agent/
├── agents.py            # モデル生成・エージェント組み立てのエントリポイント
├── agent_server.py      # FastAPI サーバー（OpenAI 互換 API として公開）
├── entities.py          # SQLAlchemy ORM エンティティ定義
├── evaluate.py          # 評価スクリプト
├── graph.py             # ContextSchema 定義・LangGraph グラフ構築
├── pack.py              # MLflow モデル登録エントリポイント
├── tools.py             # エージェントツール定義（get_tools / 各 @tool 関数）
├── loader/
│   └── obsidian.py      # Obsidian Vault → PostgreSQL ローダー (VaultLoader / VaultDb)
├── retriever/
│   └── obsidian_llama.py  # LlamaIndex を使った Vault レトリーバー (ObsidianLlamaRetriever)
└── utils/
    ├── absclass.py      # 抽象基底クラス (DocumentRetriever)
    ├── mlflow.py        # MLflow ラッパー (LangGraphChatAgent / LangGraphResponsesAgent)
    └── serving.py       # OpenAI Chat Completions API 互換ディスパッチャ (ChatCompletion)
```

### 各モジュールの責務

| モジュール | 責務 |
|---|---|
| `agents.py` | `get_model()` で LLM・埋め込みモデルを生成し、`build_agent()` でストア・グラフを組み立てて `LangGraphChatAgent` を返す |
| `agent_server.py` | FastAPI アプリを起動し、`ChatCompletion.bind()` で `/v1/chat/completions` を公開する。ローカル開発・テスト用サーバー |
| `entities.py` | `ObsidianVaultBase` / `ObsidianVaultRawEntity` など SQLAlchemy ORM クラスを定義 |
| `graph.py` | `ContextSchema`（`llm` + `obsidian_store`）を定義し、`build_graph()` で LangGraph エージェントを構築する |
| `pack.py` | `agents.build_agent()` を呼んで `mlflow.models.set_model()` に登録する薄いエントリポイント。ADR-003 時代の `agent.py`（`connector` / `context` / `graph` を直接組み合わせて `set_model()` を呼んでいた）の役割を引き継ぐ |
| `tools.py` | `get_tools()` が返す `@tool` 関数群。`ToolRuntime[ContextSchema]` 経由でコンテキストを受け取る |
| `loader/obsidian.py` | `VaultLoader`（Vault ディレクトリ → Document リスト変換）と `VaultDb`（PostgreSQL への洗い替え同期）|
| `retriever/obsidian_llama.py` | `ObsidianLlamaRetriever`。LlamaIndex IngestionPipeline でチャンク管理し、ベクター検索・ID 引き当て・バックリンク取得を提供する |
| `utils/absclass.py` | `DocumentRetriever` 抽象クラス。`search_documents` / `sync_chunks` を規定 |
| `utils/mlflow.py` | `LangGraphChatAgent`（`ChatAgent` サブクラス）・`LangGraphResponsesAgent`（`ResponsesAgent` サブクラス）の MLflow ラッパー |
| `utils/serving.py` | `ChatCompletion` ディスパッチャ。OpenAI Chat Completions API 互換エンドポイントを FastAPI に生やす |

### ADR-003 からの主な変更点

| 変更点 | 詳細 |
|---|---|
| `agent.py` を廃止し `pack.py` へ改名 | ADR-003 時代の `agent.py` は `connector` / `context` / `graph` を直接組み合わせて `mlflow.models.set_model()` に登録する薄いエントリポイントだった。モジュール名を整理した際に `pack.py` へ改名し、組み立て処理は `agents.py` へ委譲した |
| `connector.py` / `context.py` を廃止 | `connector.py` の責務は `agents.py` の `get_model()` へ移動。`context.py` の `ContextSchema` は `graph.py` へ移動し、`build_session()` の役割は `build_agent()` に統合された |
| `graph.py` に `ContextSchema` を集約 | グラフ構築とコンテキスト定義を同一ファイルに置くことで、依存関係が明確になった |
| `agents.py` をオーケストレーション層として新設 | モデル生成・ストア初期化・グラフ組み立てを一箇所に集約。`pack.py` と `agent_server.py` の両エントリポイントから共通利用される |
| `loader/` サブパッケージを追加 | Vault の取り込み・DB 同期は推論パスと独立しているため分離 |
| `retriever/` サブパッケージを追加 | ADR-005 に従い、レトリーバーはアプリ固有実装として `src/assistant_agent/retriever/` に配置 |
| `utils/serving.py` を追加 | ADR-010 に従い、OpenAI 互換サーバー層を `utils/` に配置し `agent_server.py` から利用 |

## 影響・備考

- MLflow エントリポイントは ADR-003 時代の `agent.py` から `pack.py` へ改名された。`log_model` に渡すパスは `pack.py` を指定する
- `utils/` 配下は複数エージェントプロジェクトへの共通部品候補として設計（ADR-004 の方針を継承）
- `tools.py` は `ToolRuntime[ContextSchema]` 経由でコンテキストを受け取る設計を維持（ADR-003 の判断を継承）
