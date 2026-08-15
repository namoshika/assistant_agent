# Design: {機能名}

<!-- 機能名はタスクを端的に表す名詞句。規約は .claude/rules/design_policy.md を参照。
     Python コード例は ruff の line-length に従う。 -->

## 概要

### 実装する機能

<!-- 何を・なぜ実装するか。目的・ゴールを2〜4文。 -->

> 例: DuckDB バックエンドの `DuckDBStoreConnector` を追加し、PostgreSQL なしでも RAG パイプラインを動かせるようにする。

### モジュール構成と責務

<!-- 変更が及ぶモジュールのディレクトリツリーと、各モジュールの責務を1行で。 -->

> 例:
> | モジュール | 責務 |
> |---|---|
> | `entities/base.py` | DB 非依存の列定義と抽象インターフェース |

## 変更対象ファイル

<!-- 変更・新規作成・削除するファイルを列挙し、変更種別を添える。 -->

> 例:
> | ファイル | 変更種別 |
> |---|---|
> | `src/assistant_agent/store.py` | 改修（DuckDBStoreConnector 追加） |

## 1. {変更対象}: `{ファイルパス}`

<!-- 変更対象ごとにセクションを設ける。番号は変更対象ファイル一覧の順と対応させる。 -->

### 変更方針

> 例:
> `AGENT_ID` の値をモジュール定数として保持せず、呼び出し元から `build_lc_agent` の引数として受け取る形にする。
>
> 削除: モジュール定数 `AGENT_ID`
>
> ```python
> AGENT_ID = "assistant-sample"  # thread_id の prefix・StoreBackend の namespace に使う識別子
> ```
>
> `build_lc_agent` はこれまで `AGENT_ID` を直接参照していたが、呼び出し元から `agent_id` を受け取る形にする。
>
> 変更: `build_lc_agent` のシグネチャに `agent_id: str` を追加し、内部で `AGENT_ID` を参照していた箇所をこの引数に置き換える
>
> ```python
> def build_lc_agent(checkpointer: BaseCheckpointSaver, store: BaseStore, llm: BaseChatModel, agent_id: str) -> CompiledStateGraph:
>     backend = StoreBackend(store=store, namespace=lambda _rt: (agent_id, "filesystem"))
>     ...
> ```

### テスト（単体）

> 例:
> `tests/unit/agents/test_sample.py`
>
> - `test_receive_01`:
>   - 観点1 (修正): `build_lc_agent` の呼び出しに `agent_id` 引数を追加する
>
> `tests/unit/test_agent_bot.py`
>
> 変更対象なし（`agent_bot` の import のみに依存）

### テスト（結合）

<!-- 単体と同じ形式（ファイルパス→テストメソッド→観点の箇条書き）。 -->

> 例:
> `tests/integration/utils/test_workflow.py`
>
> - `old_thread_id`（フィクスチャ）:
>   - (修正): `thread_id` の prefix 組み立てに使う定数の参照先を変更する
> - `TestDefaultRolloverStrategy.test_invoke_01`:
>   - 観点1・2 (変更なし、参照置き換えのみ): 検証観点自体は変更しない

## 2. {変更対象}: `{ファイルパス}`

<!-- 上記と同じ構成で繰り返す。 -->

## 使用パッケージ

<!-- 使用するサードパーティパッケージ。変更なしなら「変更なし」。 -->

## 設計チェック

<!-- 設計全体を横断するトレードオフ・将来課題。個別ファイル節に書くと埋もれる横断的判断を記載する。 -->
