import datetime
import re
import uuid
from pathlib import Path
from typing import Any, Iterable

import yaml
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document

_FRONT_MATTER_RE = re.compile(r"^---\r?\n(.*?)\r?\n---\r?\n", re.DOTALL)


def path_to_document_id(path: str) -> str:
    """ファイルの絶対パス文字列から document_id (UUID5) を生成する."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, path))


class MarkdownReader(BaseReader):
    def lazy_load_data(
        self, file: Path, extra_info: dict | None = None, **kwargs: Any
    ) -> Iterable[Document]:
        """Markdown ファイルを読み込み、フロントマターを metadata に取り込んで返す."""
        text = file.read_text(encoding="utf-8")
        m = _FRONT_MATTER_RE.match(text)
        meta: dict = dict(extra_info or {})
        body = text
        if m:
            fm: dict = yaml.safe_load(m.group(1)) or {}
            for k, v in fm.items():
                meta[k] = v.isoformat() if isinstance(v, (datetime.date, datetime.datetime)) else v
            body = text[m.end() :]
        file_path = meta.get("file_path") or str(file.resolve())
        yield Document(id_=path_to_document_id(file_path), text=body, metadata=meta)
