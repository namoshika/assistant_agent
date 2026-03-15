import hashlib
import datetime
import uuid
from pathlib import Path
from langchain_community.document_loaders import ObsidianLoader
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from obsidian_parser import Vault
from sqlalchemy import Engine, delete, insert
from sqlalchemy.orm import Session

from agent_assistant.model import (
    ObsidianVaultEntity,
    ObsidianVaultRawEntity,
)


def path_to_document_id(path: str) -> str:
    """vault 相対パスから document_id (UUID5) を生成する。"""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, path))


class VaultLoader(BaseLoader):
    """Obsidian Vault をロードし、forward_links を含む Document リストを返す。

    ObsidianLoader (LangChain) で Document を生成し、
    obsidianmd-parser で forward_links (document_id リスト) を補完する。
    forward_links の各値はリンク先ノートの document_id (UUID5)。
    """

    def __init__(self, vault_path: str | Path) -> None:
        self._vault_path = Path(vault_path)

    def load(self) -> list[Document]:
        docs = ObsidianLoader(str(self._vault_path), collect_metadata=True).load()
        vault = Vault(self._vault_path)
        for doc in docs:
            rel_path = str(Path(doc.metadata["path"]).relative_to(self._vault_path))
            full_body = Path(doc.metadata["path"]).read_text()
            doc.id = path_to_document_id(rel_path)
            # ObsidianLoader が None を "None" 文字列に変換するため元に戻す
            doc.metadata = {
                k: (None if v == "None" else v) for k, v in doc.metadata.items()
            }
            doc.metadata |= {
                "document_id": doc.id,
                "path": rel_path,
                "hash": hashlib.sha256(full_body.encode()).hexdigest(),
                "created": datetime.datetime.fromtimestamp(
                    doc.metadata["created"]
                ).isoformat(),
                "last_modified": datetime.datetime.fromtimestamp(
                    doc.metadata["last_modified"]
                ).isoformat(),
                "last_accessed": datetime.datetime.fromtimestamp(
                    doc.metadata["last_accessed"]
                ).isoformat(),
                "forward_links": self._extract_forward_links(rel_path, vault),
            }
        return docs

    def _extract_forward_links(self, rel_path: str, vault: Vault) -> list[str]:
        """wikilinks を document_id (UUID5) に解決して返す。"""
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


class PgVault:
    """Obsidian vault ノートを PostgreSQL に同期するクラス。"""

    @staticmethod
    def sync(
        documents: list[Document],
        sa_engine: Engine,
        raw_entity: type[ObsidianVaultEntity] = ObsidianVaultRawEntity,
    ) -> None:
        """raw テーブルを引数 documents の内容で洗い替えする。"""
        rows = [
            {
                "document_id": doc.id,
                "document_metadata": doc.metadata,
                "content": doc.page_content,
                "hash": doc.metadata["hash"],
                "path": doc.metadata["path"],
            }
            for doc in documents
        ]
        assert rows is not None
        with Session(sa_engine) as session:
            session.execute(delete(raw_entity))
            session.execute(insert(raw_entity), rows)
            session.commit()
