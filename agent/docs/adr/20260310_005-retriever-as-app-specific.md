# ADR-005: 外部データ取得層をアプリ固有の要素へ変更

- **日付**: 2026-03-10
- **ステータス**: 採択 (一部再検討予定)

## 背景

習作から実用 KB エージェントへ転換するにあたり、外部データ取得を担う ChunkStore / DocumentStore の実装が汎用モジュール (`utils/`) に置かれていた。
これらはアプリ固有のスキーマ (カラム定義・埋め込みモデル) を持つため、アプリ固有の実装 (`retriever/`) として切り出し、再定義した。

## 決定

**ファイル移動:**

| 旧パス | 新パス |
|---|---|
| `src/assistant_agent/obsidian.py` | `src/assistant_agent/loader/obsidian.py` |
| `src/assistant_agent/utils/documentstore/obsidian.py` | `src/assistant_agent/retriever/obsidian.py` |
| `src/assistant_agent/utils/documentstore/markdown.py` | `src/assistant_agent/retriever/markdown.py` |

**ChunkStore の再定義:**

- `utils/retriever.py` に散在していた `ObsidianChunkStore` / `MonthlyNewsChunkStore` を削除し、各 `retriever/` ファイルに対応する ChunkStore クラス (`ObsidianChunkStore`, `MarkdownChunkStore`) として定義し直した
- `ChunkStore.__init__` の Embedding API キーを `os.getenv` 直接呼び出しから `emb_api_key: SecretStr` 引数として受け取るよう変更した (依存性注入によるテスタビリティ向上)

**機能廃止:**

- `search_knowledge` ツールと `MonthlyNewsChunkStore` を削除した: 月次ニュースはサンプルデータで面白みが薄く、高度な実装を試みる余地も少なかった
- `ObsidianDocumentStore.get_backlinks()` を一時廃止した: 設計整理時の効率化のため一旦外した (再追加予定)

## 影響・備考

- `get_backlinks` 廃止中: バックリンク検索が現時点では使えない。再追加時は `retriever/obsidian.py` に実装し、`connector.py` のツール登録も合わせて行う
- `ChunkStore` の DI 化により、テスト時はモック実装を注入してユニットテスト可能になった
