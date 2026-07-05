import datetime
import re
import uuid
from pathlib import Path
from typing import Any, Iterator

import yaml
from langchain_core.document_loaders.base import BaseLoader
from langchain_core.documents import Document

_FRONT_MATTER_RE = re.compile(r"^---\r?\n(.*?)\r?\n---\r?\n", re.DOTALL)


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
        m = _FRONT_MATTER_RE.match(text)
        meta: dict = dict(self._extra_metadata)
        body = text
        if m:
            fm: dict = yaml.safe_load(m.group(1)) or {}
            for k, v in fm.items():
                meta[k] = v.isoformat() if isinstance(v, (datetime.date, datetime.datetime)) else v
            body = text[m.end() :]
        resolved = self._file.resolve()
        meta.setdefault("file_path", str(resolved))
        meta.setdefault("file_name", resolved.name)
        yield Document(id=path_to_document_id(str(resolved)), page_content=body, metadata=meta)
