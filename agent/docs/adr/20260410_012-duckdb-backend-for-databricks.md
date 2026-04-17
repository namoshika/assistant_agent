# ADR-012: バックエンドの複数対応と DuckDB バックエンドの追加

- **日付**: 2026-04-10
- **ステータス**: 採択

## 背景

エージェントを Databricks 上で動作できるようにしたい。本プロジェクトはこれまでローカル環境での動作を前提に開発を進めてきたが、Databricks 製品上での稼働を実現し、この製品上で PoC する際のベースとして活用できる状態を目指す。

現状のバックエンド DB には PostgreSQL を採用している。Databricks においても PostgreSQL は **Lakebase** として提供されているが、**2026/04 時点では ap-northeast-1 リージョンで未提供**であり、使用が難しい。

## 代替バックエンドの検討

Lakebase が使えない環境向けに以下の選択肢を検討した。

| 選択肢 | 評価 |
|---|---|
| `SimpleVectorStore` / `SimpleDocumentStore` | RAG のデータフィルタ機能が不足。メタデータ条件による絞り込みや `backlink_filter` の実装が PostgreSQL と比較して困難 |
| `ChromaVectorStore` | 同様にフィルタ機能が不足。加えてサーバープロセスが別途必要 |
| DuckDB | 単一ファイルで動作する組み込み OLAP DB。LlamaIndex 公式の VectorStore・DocumentStore が整備されており、`backlink_filter` の実装も可能 |

## 決定

**バックエンドを複数対応できる設計とし、DuckDB バックエンドを追加する。**

DuckDB を採用した理由は以下のとおり。

- サーバープロセスが不要（単一ファイルで動作する組み込み DB）で、Databricks ノートブックやジョブからそのまま利用できる
- `llama-index-vector-stores-duckdb` / `llama-index-storage-docstore-duckdb` により、LlamaIndex の IngestionPipeline・VectorStoreIndex・バックリンク検索を PostgreSQL と同等の操作感で使用できることを PoC で確認済み
- `duckdb-engine` により SQLAlchemy 経由の ORM 操作も可能で、`backlink_filter` を DuckDB 向けに実装できることを確認済み

## 備考

- DuckDB バックエンドの位置づけは **PoC・デモ向け**であり、本番適用を前提としない
- Databricks 本番環境への適用は Lakebase の ap-northeast-1 提供開始を待つことになる（時期未定）
- 本番適用を急ぐ場合は独自 ETL パイプラインの作り込みが必要だが、LlamaIndex の豊富な機能で迅速に検証できるというコンセプトから逸脱するため現時点では採用しない
- DuckDB バックエンドの調査・PoC の詳細は `docs/experimental/20260408_duckdb_store_context/` を参照
