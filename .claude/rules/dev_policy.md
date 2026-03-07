---
paths:
  - src/*
---
# ソース方針

## ディレクトリ

- `src/`: ソースコード
- `src/libs/`: 共通ソースコード (ローカルと Databricks の両環境を念頭に置く)

## 主要なモジュール

| ファイル | 役割 |
|---|---|
| `src/01_import_data.ipynb` | データ登録ワークフロー |
| `src/02_make_agent.ipynb` | エージェント開発・動作確認用プレイグラウンド |
| `src/03_evaluate.ipynb` | 評価ワークフロー |
| `src/agent.py` | エージェント本体 |
| `src/evaluate.py` | エージェント評価の基準定義 |
| `src/libs/absclass.py` | 抽象インターフェース |
| `src/libs/retriever.py` | サンプル RAG 用のベクターストア |
| `src/libs/utils.py` | 細かい各種実装 |
| `src/libs/chunkstore/` | ベクターストアとのコネクタ (ChunkReader/Writer) |
| `src/libs/chunker/` | ドキュメントのチャンキング実装 (Chunker) |
| `src/libs/documentstore/` | ChunkReader/Writer と Chunker を組み合わせた検索・登録オーケストレーター |
