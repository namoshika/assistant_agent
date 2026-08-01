# ADR-014: StoreConnector / Retriever 層の非同期化

- **日付**: 2026-07-08
- **ステータス**: 採択

## 背景

`store.py` の `PostgresStoreConnector.get_vector_store()` は、`PGEngine.from_engine()` が `AsyncEngine` 専用であり `get_engine()` が返す同期 `Engine` をラップできないという制約のため、`PGEngine.from_connection_string()` で別途接続を二重生成していた。`get_engine()` を `AsyncEngine` 化し `PGEngine.from_engine()` に直接渡すことで、この二重生成を解消したい。

事前調査で `.venv` 内 `langchain_postgres` のソースを確認したところ、`PGEngine.from_engine(engine, loop=None)` は `loop` を渡さない場合、`PGVectorStore.create_sync()` 等が内部で使用する `_run_as_sync()` が `Exception("Engine was initialized without a background loop and cannot call sync methods.")` を送出する仕様であることが判明した。一方 `PGEngine.from_connection_string()` は内部で `create_async_engine()` し、専用のバックグラウンドスレッド + event loop を生成・保持することで同期メソッドの呼び出しをサポートしている。

したがって `get_engine()` を単純に `AsyncEngine` 化して `from_engine()` に渡すだけでは、現状の `PGVectorStore.create_sync()` 呼び出しが破綻する。`get_vector_store()` 側も非同期 API（`PGVectorStore.create()` + `await`）に切り替える必要があり、これに伴い呼び出し元の `VaultObsidianRetriever`/`VaultSampleRetriever` の各メソッド、`VaultUtils`、LangChain `@tool` 関数、関連する単体・結合テスト全体を非同期化するフルスコープの改修が必要と判明した。

## 決定

`StoreConnector`（`utils/absclass.py`）の `get_engine()` の戻り値型を `Engine` から `AsyncEngine` に変更し、関連する層を一貫して非同期化する。

- `PostgresStoreConnector.get_engine()`: `create_async_engine(url.set(drivername="postgresql+asyncpg"))` でキャッシュ生成する。接続文字列自体（環境変数 `ENV_PG_CONNECTION_STRING`）は `postgresql://` 形式のまま受け取り、内部でドライバのみ `postgresql+asyncpg` に差し替える。非同期対応ドライバ `asyncpg` は `langchain-postgres` の依存として既に導入済みであることを確認済み
- `PostgresStoreConnector.get_vector_store()`: `PGEngine.from_engine(self.get_engine())` を使うよう変更し、`PGEngine.from_connection_string()` による二重生成をやめる。`PGVectorStore.create_sync()` から `PGVectorStore.create()`（非同期）への切り替えに伴い、`get_vector_store()` 自体を `async def` にする
- `VaultUtils`（`entities/base.py`）: `sync_docs()`/`sync_chunks()` を `AsyncSession`/`AsyncConnection` を使った非同期メソッドに変更する
- `VaultObsidianRetriever`/`VaultSampleRetriever`: `initialize()`, `search_documents()`, `get_documents_by_ids()`, `get_backlinks()`, `sync_chunks()` を `async def` 化し、`Session(self._sa_engine)` を `AsyncSession(self._sa_engine)` + `async with` に変更する
- LangChain ツール（`tools/obsidian.py`, `tools/sample.py`）: ツール関数を `async def` 化し `await retriever.xxx()` に変更する。`langchain.tools.tool` デコレータは関数が `async def` であれば自動的に非同期ツールとして登録され、LangGraph 実行時（`ainvoke` 系）から `await` されるため、呼び出し側の変更は不要
- テスト: `pytest-asyncio` を新規導入し、影響する単体・結合テスト（`test_store.py`, `services/*`, `tools/*` 等）を非同期対応（`async def test_xxx` + `await`）に書き換える

## 影響・備考

- 本決定の詳細な調査記録（`PGEngine` の同期/非同期実装比較、`AsyncSession` への置き換え方法等）は `docs/experimental/20260708_async_engine/` を参照
- `pytest-asyncio` を新規依存として追加した
