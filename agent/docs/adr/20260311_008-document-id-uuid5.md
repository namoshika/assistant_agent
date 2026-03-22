# ADR-008: Document の識別子を document_id (UUID v5) へ変更

- **日付**: 2026-03-11
- **ステータス**: 採択

## 背景

エージェントが Obsidian vault のドキュメントを参照する際、識別子として `path` (vault 相対パス) のみを使用していた。

リンク先ノートを取得する際は、LLM が wikilink から拡張子を補完して生成したパスをツール側で後方一致検索していた。
しかし、ファイルパスにスペースや全角文字が含まれる場合に動作が不安定となる問題があった。

## 決定

`document_id` を導入し、完全一致によるドキュメント取得を可能にする。 `document_id` は `path` から UUID v5 で決定論的に生成。

```python
import uuid

def path_to_document_id(path: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, path))
```

- 同一パスなら常に同じ UUID が得られるため、ドキュメント再インポート時も ID は変わらない
- UUID をサーバー側で生成・取得する (`RETURNING`) 必要がなく、実装がシンプルになる

## 影響

- 既存の DB テーブルは破壊的変更のため再作成が必要 (vault の再インポートで対応)
