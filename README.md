# Hello Agent World
AI エージェント開発を学習する用の砂場。

## 実行・デプロイ手順

```sh
# スコープ作成
databricks secrets create-scope assistant_agent --profile DEFAULT

# シークレット値の登録
databricks secrets put-secret assistant_agent aws_access_key_id --string-value <AWS_ACCESS_KEY_ID>
databricks secrets put-secret assistant_agent aws_secret_access_key --string-value <AWS_SECRET_ACCESS_KEY>
databricks secrets put-secret assistant_agent aws_default_region --string-value <AWS_DEFAULT_REGION>
databricks secrets put-secret assistant_agent gemini_api_key --string-value <ENV_GEMINI_API_KEY>
databricks secrets put-secret assistant_agent dbx_pat_agent --string-value <ENV_DATABRICKS_TOKEN>

# アプリデプロイ
databricks bundle deploy -t dev --profile DEFAULT
databricks bundle run agent -t dev --profile DEFAULT
databricks bundle run chatbot -t dev --profile DEFAULT
```
