# Design: {機能名}

<!-- 機能名はタスクの内容を端的に表す名詞句で記載する。例: "DuckDB StoreContext" -->

---

## 概要

### 実装する機能

<!-- 何を・なぜ実装するか。目的・背景・達成したいゴールを2〜4文で記載する。 -->

> 記載例:
>
> DuckDB をバックエンドとする `DuckDBStoreContext` を追加し、PostgreSQL なしでローカル・Databricks 環境の両方で動作する RAG パイプラインを実現する。
>
> あわせてエンティティ層を DB 種別に依存しない設計へ整理し、PostgreSQL / DuckDB を同一インターフェースで切り替えられるようにする。

### モジュール構成と責務

<!-- 変更・追加が及ぶモジュールのディレクトリツリーを示し、各モジュールの責務を 1 行で記載する。 -->

> 記載例:
>
> ```
> src/agent_assistant/
> ├── entities/
> │   ├── base.py        # 共通基底クラス・backlink_filter 抽象定義
> │   ├── postgres.py    # PostgreSQL 用ミックスイン・テーブルクラス
> │   └── duckdb.py      # DuckDB 用ミックスイン・テーブルクラス
> └── utils/
>     └── store_factory.py  # StoreContext 実装群（DuckDBStoreContext 追加）
> ```
>
> | モジュール | 責務 |
> |---|---|
> | `entities/base.py` | DB 非依存の列定義と `backlink_filter` の抽象インターフェース |
> | `entities/postgres.py` | PostgreSQL 固有の JSONB 型と `backlink_filter` 実装 |
> | `utils/store_factory.py` | VectorStore / DocumentStore の生成を DB 種別ごとに隠蔽 |

---

## 変更対象ファイル

<!-- 変更・新規作成・削除するファイルを列挙する。変更種別は「新規作成 / 改修 / 削除」のいずれかを記載する。 -->

> 記載例:
>
> | ファイル | 変更種別 |
> |---|---|
> | `src/agent_assistant/entities.py` | 削除（ディレクトリへ移行） |
> | `src/agent_assistant/entities/base.py` | 新規作成（共通基底クラス） |
> | `src/agent_assistant/utils/store_factory.py` | 改修（DuckDBStoreContext 追加） |
> | `tests/integration/test_store_factory.py` | 新規作成（DuckDB テスト追加） |

---

## 1. {変更対象}: `{ファイルパス}`

<!-- 変更対象ごとにセクションを設ける。番号は変更対象ファイルテーブルの順番と対応させる。 -->

### 現状

<!-- 変更前のコード・クラス構成・依存関係など、現在の状態を記載する。変更の動機が伝わるよう、問題点や制約も添える。 -->

> 記載例:
>
> `src/agent_assistant/entities.py` に以下が定義されている。
>
> ```python
> class ObsidianVaultEntity:      # 基底（JSONB 列定義込み）
> class ObsidianVaultBase(DeclarativeBase): ...
> class ObsidianVaultRawEntity(ObsidianVaultBase, ObsidianVaultEntity): ...
> ```
>
> `backlink_filter` に PostgreSQL 固有の `JSONB.contains` が直接使われており、DuckDB では動作しない。

### 変更方針

<!-- 変更の具体的な内容を記載する。方針の説明 + 変更後コードの両方を示す。
     コード変更が大きい場合はビフォー/アフターを並べる。注意点・制約・設計上の意図も記載する。 -->

> 記載例:
>
> `entities.py` を削除し `entities/` ディレクトリへ移行する。`backlink_filter` を抽象メソッドとして基底クラスに定義し、DB 固有実装は各サブクラスに委譲する。
>
> ```python
> # entities/base.py（変更後）
> from abc import abstractmethod
> from sqlalchemy.sql.elements import ColumnElement
>
> class ObsidianVaultEntity:
>     @classmethod
>     @abstractmethod
>     def backlink_filter(cls, document_id: str) -> ColumnElement[bool]:
>         raise NotImplementedError
> ```
>
> **注意:** `@classmethod` + `@abstractmethod` の二重デコレータは実行時・pyright ともにエラーなし（検証済み）。

### テスト（単体）

<!-- 単体テストで確認すべき観点と方針を記載する。追加・削除がない場合もその旨を明示する。 -->

> 記載例:
>
> 追加・削除なし（`entities/` は純粋なデータモデル定義のため単体テストの対象外）。

> 追加ありの場合の例:
>
> `tests/unit/test_store_factory.py` の Chroma 関連テスト4件をすべて削除し、関連インポートも削除する。

### テスト（結合）

<!-- 結合テストで確認すべき観点・テストメソッド名・フィクスチャを記載する。 -->

> 記載例:
>
> `tests/integration/test_entities.py`（新規作成）で検証する。
>
> - `test_backlink_filter_01`（PostgreSQL）
>   - 観点1: `backlink_filter` が `document_metadata["forward_links"]` に対してフィルタできること
> - `test_backlink_filter_02`（DuckDB）
>   - 観点1: `backlink_filter` が `document_metadata["forward_links"]` に対してフィルタできること

---

## 2. {変更対象}: `{ファイルパス}`

<!-- 上記と同じ構成で繰り返す。変更対象が増えるたびにセクションを追加する。 -->

### 現状

<!-- （記載する） -->

### 変更方針

<!-- （記載する） -->

### テスト（単体）

<!-- （記載する） -->

### テスト（結合）

<!-- （記載する） -->

---

## 設計チェック

<!-- 設計全体を横断する注意点・制約・検証済み事項をまとめる。
     個別ファイルに書くと埋もれる「型チェックの挙動」「パッケージの制約」「将来の拡張への考慮」などを記載する。 -->

> 記載例:
>
> - `document_metadata: Mapped[dict]`（`mapped_column()` なし）を基底に置くことで pyright が各サブクラスの `row.document_metadata` アクセスを認識できる（検証済み）
> - `postgres.py` / `duckdb.py` でクラス名が重複するが、`DeclarativeBase` が異なるため `metadata` は独立し衝突しない（検証済み）
> - `DuckDBVectorStore` はノードのメタデータに list 型を受け付けない（`str / int / float / None` のみ）。`forward_links`（`list[str]`）はメタデータから除外して渡す
