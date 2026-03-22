# ADR-004: ディレクトリ構成の変更

- **日付**: 2026-03-10
- **ステータス**: 採択

## 背景

習作から実用 KB エージェントへの転換を見据え、役割に応じたディレクトリ分離を行った。
また、当初 `agent-rag` という名前のサブプロジェクトとして扱われていたが、「RAG」に限定されない汎用的なアシスタントとしての発展性を持たせるため名称を汎用化し、あわせて `scripts/` の Notebook から `src/` パッケージを扱いやすくするための editable install 化も行った。

## 決定

**パッケージ名・名前空間の変更:**

- `pyproject.toml` のパッケージ名を `agent-rag` から `agent-assistant` へ変更した
- `src/` 直下・`src/libs/` 配下に散在していた各モジュールを `src/agent_assistant/` 配下へ移動し、名前空間を `agent_assistant.*` に統一した
- `src/libs/utils.py` (中身は主に MLflow 連携の `LangGraphWrapper`) を責務を明確にする目的で `src/agent_assistant/utils/mlflow.py` へリネームした

**ディレクトリ役割の定義:**

| ディレクトリ | 役割 |
|---|---|
| `scripts/` | メンテナンス系 Notebook。アドホックな作業による更新が高頻度で発生する |
| `src/agent_assistant/utils/` | 複数エージェント開発での共通部品として使うことを見越した抽象クラス・汎用モジュール |
| `src/agent_assistant/*` | アプリ固有の実装。設計が定まったコードを格納する |

**editable install 化:**

`scripts/` から `src/` を import しやすくするため、src 配下を `agent_assistant` パッケージ化し editable install された構成として扱う方針とした:

- 相対インポート (`from ..absclass import ...`) を絶対インポート (`from agent_assistant.utils import absclass`) に統一した
- `pyproject.toml` の `[tool.pytest.ini_options]` から `pythonpath = ["src"]` を削除した (`uv run` による editable install で解決するため不要)

`pyproject.toml` の開発依存に `ipywidgets` を追加した。

## 影響・備考

- プロジェクト内のインポートはすべて `agent_assistant.*` を基準とする絶対インポートへ統一された
- `scripts/` の Notebook は editable install 済み環境 (`uv run jupyter`) で実行する
- `pytest` は `pythonpath` 設定なしで動作する (`uv run pytest`)
