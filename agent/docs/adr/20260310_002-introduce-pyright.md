# ADR-002: 静的解析ツール (pyright) の導入

- **日付**: 2026-03-10
- **ステータス**: 採択

## 背景

コードの品質と保守性を高めるため、静的解析ツール (pyright) を導入した。
導入に際し、pyright が検出したインポートパスの誤りや型の不一致を修正した。

## 決定

- mlflow のインポートパスを正確なサブモジュールパスへ修正した

  | 旧 | 新 |
  |---|---|
  | `from mlflow.pyfunc import ResponsesAgent` | `from mlflow.pyfunc.model import ResponsesAgent` |
  | `from mlflow.types.responses import Message` | `from mlflow.types.responses_helpers import Message, OutputItem` |

- ライブラリ側の型定義が不完全なために pyright が誤検出する箇所には `# pyright: ignore[...]` を追加してサプレッションした

## 影響・備考

- `# pyright: ignore` を付与した箇所は型安全性を一部妥協している。対象ライブラリの型定義が整備された時点で除去を検討する
