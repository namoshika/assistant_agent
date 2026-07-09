import pytest
from langchain_core.documents import Document

from assistant_agent.entities import postgres
from assistant_agent.entities.base import VaultUtils
from assistant_agent.services import VaultSampleRetriever
from assistant_agent.store import PostgresStoreConnector


@pytest.mark.integration
async def test_search_documents_01(
    pg_conn: PostgresStoreConnector,
    pg_retriever_smpl: VaultSampleRetriever,
    pg_entity_smpl: type[postgres.DocumentFields],
    docs_smpl: list[Document],
):
    """指定したクエリに類似するチャンク Document を取得できるか確認.

    前提: VaultUtils.sync + sync_chunks 済み

    観点1: 1件以上の結果が返り、対象ドキュメントの document_id・file_path を含むチャンクがある
    """
    # 試験準備
    raw_entity = pg_entity_smpl
    await VaultUtils.sync_docs(docs_smpl, pg_conn.get_engine(), raw_entity)
    await pg_retriever_smpl.sync_chunks()
    doc = docs_smpl[0]

    # 試験実施・結果検証
    # 観点1
    results = await pg_retriever_smpl.search_documents(doc.page_content[:30], top_k=10)
    assert len(results) >= 1
    matched = next((r for r in results if r.metadata.get("document_id") == doc.id), None)
    assert matched is not None
    assert matched.metadata["file_path"] == doc.metadata["file_path"]
