import datetime
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import yaml
from langchain_core.document_loaders.base import BaseLoader
from langchain_core.documents import Document

FRONT_MATTER_DELIMITER = "---\n"


def path_to_document_id(path: str) -> str:
    """ファイルの絶対パス文字列から document_id (UUID5) を生成する."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, path))


class MarkdownLoader(BaseLoader):
    """単一の Markdown ファイルをロードする.

    DirectoryLoader の loader_cls 契約 (`loader_cls(str(path), **loader_kwargs)`) に
    合わせ、ファイルパスを位置引数、追加メタデータを **extra_metadata で受け取る。
    """

    def __init__(self, file: str | Path, **extra_metadata: Any) -> None:
        """Construct MarkdownLoader."""
        self._file = Path(file)
        self._extra_metadata = extra_metadata

    def lazy_load(self) -> Iterator[Document]:
        """Markdown ファイルを読み込み、フロントマターを metadata に取り込んで返す."""
        text = self._file.read_text(encoding="utf-8")
        raw_front_matter, body = self.split_front_matter(text)
        meta: dict = dict(self._extra_metadata)
        if raw_front_matter is not None:
            fm: dict = yaml.safe_load(raw_front_matter) or {}
            for k, v in fm.items():
                meta[k] = v.isoformat() if isinstance(v, (datetime.date, datetime.datetime)) else v
        resolved = self._file.resolve()
        meta.setdefault("file_path", str(resolved))
        meta.setdefault("file_name", resolved.name)
        yield Document(id=path_to_document_id(str(resolved)), page_content=body, metadata=meta)

    @staticmethod
    def split_front_matter(text: str) -> tuple[str | None, str]:
        """先頭・終端を `---` で囲むフロントマター記法で text を分割する.

        区切りが見つかった場合は (フロントマターの YAML 文字列, 本文) を返す。
        見つからない場合は (None, text) を返し、フロントマター無しか否かの
        判定を呼び出し元に委ねる。改行コードは CRLF/LF のどちらも受け付ける。
        """
        normalized = text.replace("\r\n", "\n")
        head, delimiter, rest = normalized.partition(FRONT_MATTER_DELIMITER)

        # フロントマターの始端無し、または先頭に余計な文字列がある場合はフロントマター無しとみなす
        if head != "" or not delimiter:
            return None, text
        # フロントマターの終端が見つからない場合もフロントマター無しとみなす
        raw_front_matter, delimiter, body = rest.partition(f"\n{FRONT_MATTER_DELIMITER}")
        if not delimiter:
            return None, text

        # フロントマターの終端が見つかった場合は (フロントマター, 本文) を返す
        return raw_front_matter, body
