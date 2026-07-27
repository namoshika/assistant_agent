# Requirements: {機能名}

<!-- 機能名はタスクの内容を端的に表す名詞句で記載する。例: "DuckDB StoreConnector" -->

## 目的

<!-- 何を・なぜ実装するか。対象ファイル・クラス名と、実装の動機・背景を2〜4文で記載する。
     設計検討の過程（比較した代替案、途中の議論）は含めない。 -->

> 記載例:
>
> `src/assistant_agent/store.py` に `DuckDBStoreConnector` を追加する。
> DuckDB は単一ファイルで動作する組み込み OLAP DB であり、Databricks 上でもローカルでも依存を最小化したい場面に適する。

## 事前調査

<!-- 設計前に行った調査・検証の概要を記載する。PoC や実験ノートブックがある場合はパスを示す。
     調査が不要だった場合は「なし」と記載する。 -->

> 記載例:
>
> 設計前に以下のフェーズで調査・検討を行った。結果は `./poc/` に記録されている。
>
> | フェーズ | 内容 |
> |---|---|
> | Phase 1 | DuckDB で CRUD の基本動作を実証（duckdb ネイティブ・SQLAlchemy 経由） |
> | Phase 2 | `VaultObsidianRetriever` が必要とする処理が DuckDB で動作するか調査 |
> | Phase n | PostgreSQL / DuckDB 共通化方針を決定 |

## 実装要件

<!-- 実装が満たすべき要件を、変更対象コンポーネントごとに箇条書きで記載する。
     「何をするか」を中心に記載し、「どう実装するか」は design.md に委ねる。
     ただし制約・除外事項・注意点はここで明示する。

     以下は書かない（design.md に委ねる）:
     - 理由・経緯（「なぜこの設計にしたか」「検討した代替案との比較」）
     - 現状描写（変更前の実装がどうなっているかの説明。design.md の「現状」節の役割）
     - 実装方法の先回り（具体的な API・関数名の選定は design.md で行う）
     - 機能要件と全文重複する記載（要件番号を参照するだけに留める）
     - 自明な言い換え（既述の内容から当然導ける帰結の再掲） -->

### {コンポーネント名 1}（{ファイル名}）

<!-- コンポーネント（クラス・モジュール）ごとにセクションを分ける。 -->

> 記載例:
>
> - `StoreConnector` を継承し、`get_vector_store()` / `get_docstore()` / `get_engine()` を実装する
> - インメモリモード（`persist_dir=None`）とファイル永続化モード（`persist_dir` 指定）の両方に対応する
> - LlamaIndex と SQLAlchemy Engine は同一 DuckDB ファイルへの同時接続が不可のため、**ファイルを分けて管理する**
>   - LlamaIndex 用: `{persist_dir}/{name}_llamaindex.duckdb`
>   - SQLAlchemy 用: `{persist_dir}/entity.duckdb`
> - `close()` を実装し、内部で `CHECKPOINT` を実行する（WAL の安全なフラッシュのため）
> - `persist()` は不要（ファイル永続化モードでは書き込み時に自動永続化されるため）

### {コンポーネント名 2}（{ファイル名}）

> 記載例:
>
> - `DocumentFields` をインターフェース的な基底クラスとし、`document_metadata` の列定義は具体クラスに持たせる
> - DB の違いによる実装差異（`backlink_filter` など）はサブクラスのクラスメソッドとして定義する
> - 列順は現行と同じ `document_id, document_metadata, content, path` を維持する

## 使用パッケージ

<!-- 新たに追加・削除するパッケージを記載する。変更がない場合は「変更なし」と記載する。 -->

> 記載例:
>
> - `duckdb` (>=1.0)
> - `llama-index-vector-stores-duckdb`
> - `duckdb-engine` (SQLAlchemy dialect)
