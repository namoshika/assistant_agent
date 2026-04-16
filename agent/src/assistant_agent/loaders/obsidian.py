import datetime
import re
import uuid
from pathlib import Path
from typing import Any, Iterable

from llama_index.core import Document
from llama_index.core.readers.base import BaseReader
from obsidian_parser import Vault

_FRONT_MATTER_REGEX = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)


def path_to_document_id(path: str) -> str:
    """Vault ルートを基準とした相対パスから document_id (UUID5) を生成する."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, path))


class VaultLoader(BaseReader):
    """Vault ディレクトリをロードし、forward_links を含む Document リストを返す.

    obsidianmd-parser で Document を生成する。
    forward_links の各値はリンク先ノートの document_id (UUID5)。
    """

    def __init__(self, vault_path: Path) -> None:
        """Construct VaultLoader."""
        self._vault_path = vault_path

    def lazy_load_data(self, *args: Any, **load_kwargs: Any) -> Iterable[Document]:
        """Vault ディレクトリからドキュメントをロードし、メタデータを付与する."""
        vault_path = self._vault_path.resolve()
        vault = Vault(vault_path)
        docs = []
        for note in vault.notes:
            rel_path = str(note.path.relative_to(vault_path))
            raw_text = note.path.read_text(encoding="UTF-8")
            # datetime.date / datetime.datetime は LlamaIndex のメタデータフィルターや
            # DB (DuckDB / PostgreSQL JSONB) への格納時に型エラーが発生するため ISO 文字列に変換する
            metadata = {
                k: v.isoformat() if isinstance(v, (datetime.date, datetime.datetime)) else v
                for k, v in note.frontmatter.items()
            }
            metadata["path"] = rel_path
            metadata["forward_links"] = self._extract_forward_links(rel_path, vault)
            docs.append(
                Document(
                    id_=path_to_document_id(rel_path),
                    text=_FRONT_MATTER_REGEX.sub("", raw_text),
                    metadata=metadata,
                )
            )
        return docs

    def _extract_forward_links(self, rel_path: str, vault: Vault) -> list[str]:
        """Document のwikilinks を document_id (UUID5) に解決して返す."""
        note = vault.get_note(rel_path)
        if note is None:
            return []
        result = []
        for link in note.wikilinks:
            target = vault.get_note(link.target)
            if target is not None:
                target_path = str(target.path.relative_to(vault.path))
                result.append(path_to_document_id(target_path))
        return result
