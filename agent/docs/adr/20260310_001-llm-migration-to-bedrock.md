# ADR-001: LLM を Gemini から Bedrock の Claude へ変更

- **日付**: 2026-03-10
- **ステータス**: 採択 (暫定・再検討予定)

## 背景

Gemini API (`ChatGoogleGenerativeAI`) の応答に十数分以上の遅延が発生し、エージェントの動作確認や開発が困難な状況となった。
問題の解消を待つ間も開発を継続するため、一時的に Amazon Bedrock 上の Claude モデルへ切り替えることとした。

## 決定

- LLM を `ChatGoogleGenerativeAI` から `ChatBedrock` (`global.anthropic.claude-haiku-4-5-20251001-v1:0`) へ変更した
- LLM 認証情報を以下の環境変数から取得するよう変更した

  | 環境変数 | 扱い |
  |---|---|
  | `AA_GEMINI_API_KEY` | 必須 (埋め込み表現取得で引き続き使用) |
  | `ENV_GEMINI_MODEL_ID` | 不要 (未使用に変化) |
  | `AWS_ACCESS_KEY_ID` | 必須 (新規追加) |
  | `AWS_SECRET_ACCESS_KEY` | 必須 (新規追加) |
  | `AWS_DEFAULT_REGION` | 任意 (新規追加 (省略時: `us-east-1`)) |

- Gemini API 遅延への対応中に試行錯誤した名残として、`evaluate.py` の `is_japanese` 評価スコアラーをコメントアウトした状態とした
- `pyproject.toml`: `langchain[aws]` extra を追加、開発依存に `boto3`・`litellm` を追加した

## 影響・備考

- AWS 認証情報 (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`) が起動の必須環境変数となった
- `is_japanese` スコアラーが無効のため、評価時に日本語判定が行われない状態である
- Gemini API の問題が解消した時点で LLM・スコアラーの選択を再検討する
