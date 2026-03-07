import datetime
from pathlib import Path

from langchain_community.document_loaders import ObsidianLoader
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from obsidian_parser import Vault


class VaultLoader(BaseLoader):
    """Obsidian Vault をロードし、forward_links を含む Document リストを返す。

    ObsidianLoader (LangChain) で Document を生成し、
    obsidianmd-parser で forward_links (vault 相対パスリスト) を補完する。
    forward_links の各パスは raw テーブルの path PK と同一形式。
    """

    def __init__(self, vault_path: str | Path) -> None:
        self._vault_path = Path(vault_path)

    def load(self) -> list[Document]:
        docs = ObsidianLoader(str(self._vault_path), collect_metadata=True).load()
        vault = Vault(self._vault_path)
        for doc in docs:
            rel_path = str(Path(doc.metadata["path"]).relative_to(self._vault_path))
            doc.metadata |= {
                "path": rel_path,
                "created": datetime.datetime.fromtimestamp(doc.metadata["created"]).isoformat(),
                "last_modified": datetime.datetime.fromtimestamp(doc.metadata["last_modified"]).isoformat(),
                "last_accessed": datetime.datetime.fromtimestamp(doc.metadata["last_accessed"]).isoformat(),
                "forward_links": self._extract_forward_links(rel_path, vault),
            }
        return docs

    def _extract_forward_links(self, rel_path: str, vault: Vault) -> list[str]:
        """wikilinks を vault 相対パスに解決して返す。"""
        note = vault.get_note(rel_path)
        if note is None:
            return []
        result = []
        for link in note.wikilinks:
            target = vault.get_note(link.target)
            if target is not None:
                result.append(str(target.path.relative_to(vault.path)))
        return result
