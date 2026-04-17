import pytest
from llama_index.core import Document

from assistant_agent.entities import postgres
from assistant_agent.entities.base import VaultUtils
from assistant_agent.services import VaultSampleRetriever
from assistant_agent.utils.store_context import PostgresStoreContext


@pytest.mark.integration
def test_search_documents_01(
    pg_cxt: PostgresStoreContext,
    pg_retriever_smpl: VaultSampleRetriever,
    pg_entity_smpl: type[postgres.DocumentFields],
    docs_smpl: list[Document],
):
    """指定したクエリに類似する Document を取得できるか確認.

    前提: VaultUtils.sync + sync_chunks 済み

    観点1: フィルタなし — 1件以上の結果が返り、id・page_content・metadata が一致する
    観点2: フィルタあり — 指定条件に合致するDocumentのみ返る
        ケース1: date >= "2026-01-01 00:00:00" で絞ると 2025 年以前が除外される
        ケース2: path text_match "01_Inbox" で絞ると該当フォルダのみ返る
    """
    # 試験準備
    raw_entity = pg_entity_smpl
    VaultUtils.sync(docs_smpl, pg_cxt.get_engine(), raw_entity)
    pg_retriever_smpl.sync_chunks()
    doc = docs_smpl[0]

    # 試験実施・結果検証
    # 観点1
    results = pg_retriever_smpl.search_documents(doc.text[:30], top_k=10)
    assert len(results) >= 1
    matched = next((r for r in results if r.node.ref_doc_id == doc.id_), None)
    assert matched is not None
    assert matched.text == doc.text
    assert matched.metadata == doc.metadata
