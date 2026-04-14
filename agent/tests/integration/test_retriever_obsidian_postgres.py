import pytest
from langchain_core.documents import Document
from llama_index.core.vector_stores.types import FilterOperator, MetadataFilter, MetadataFilters

from assistant_agent.loader.obsidian import VaultDb, path_to_document_id
from assistant_agent.retriever.obsidian_llama import ObsidianLlamaRetriever
from assistant_agent.utils.store_factory import PostgresStoreContext


@pytest.mark.integration
def test_search_documents_01(
    pg_cxt: PostgresStoreContext,
    pg_obsidian_retriever: ObsidianLlamaRetriever,
    pg_entity: type,
    vault_docs: list[Document],
):
    """指定したクエリに類似する Document を取得できるか確認.

    前提: VaultDb.sync + sync_chunks 済み

    観点1: フィルタなし — 1件以上の結果が返り、id・page_content・metadata が一致する
    観点2: フィルタあり — 指定条件に合致するDocumentのみ返る
        ケース1: date >= "2026-01-01 00:00:00" で絞ると 2025 年以前が除外される
        ケース2: path text_match "01_Inbox" で絞ると該当フォルダのみ返る
    """
    # 試験準備
    raw_entity = pg_entity
    VaultDb.sync(vault_docs, pg_cxt.get_engine(), raw_entity)
    pg_obsidian_retriever.sync_chunks()
    doc = vault_docs[0]

    # 試験実施・結果検証
    # 観点1
    results = pg_obsidian_retriever.search_documents(doc.page_content[:30], top_k=10)
    assert len(results) >= 1
    matched = next((r for r in results if r.id == doc.id), None)
    assert matched is not None
    assert matched.page_content == doc.page_content
    assert matched.metadata == doc.metadata

    # 観点2 ケース1: date フィルタ
    date_filter = MetadataFilters(
        filters=[
            MetadataFilter(key="date", value="2026-01-01 00:00:00", operator=FilterOperator.GTE)
        ]
    )
    results_date = pg_obsidian_retriever.search_documents(
        doc.page_content[:30], top_k=10, filters=date_filter
    )
    assert all(r.metadata["date"] >= "2026-01-01 00:00:00" for r in results_date)

    # 観点2 ケース2: path フィルタ
    path_filter = MetadataFilters(
        filters=[MetadataFilter(key="path", value="01_Inbox", operator=FilterOperator.TEXT_MATCH)]
    )
    results_path = pg_obsidian_retriever.search_documents("project", top_k=10, filters=path_filter)
    assert len(results_path) >= 1
    assert all("01_Inbox" in r.metadata["path"] for r in results_path)


@pytest.mark.integration
def test_get_documents_by_ids_01(
    pg_cxt: PostgresStoreContext,
    pg_obsidian_retriever: ObsidianLlamaRetriever,
    pg_entity: type,
    vault_docs: list[Document],
):
    """document_ids に対応する Document を取得できるか確認.

    観点1: 既存 id を指定すると id・page_content・metadata の全項目が一致する
    観点2: 存在しない document_id を指定すると空リストが返る
    """
    # 試験準備
    raw_entity = pg_entity
    doc = vault_docs[0]
    VaultDb.sync([doc], pg_cxt.get_engine(), raw_entity)

    # 試験実施
    assert doc.id is not None
    results = pg_obsidian_retriever.get_documents_by_ids([doc.id])

    # 結果検証
    # 観点1
    assert len(results) == 1
    assert results[0].id == doc.id
    assert results[0].page_content == doc.page_content
    assert results[0].metadata == doc.metadata
    # 観点2
    assert pg_obsidian_retriever.get_documents_by_ids(["nonexistent-uuid"]) == []


@pytest.mark.integration
def test_get_backlinks_01(
    pg_cxt: PostgresStoreContext,
    pg_obsidian_retriever: ObsidianLlamaRetriever,
    pg_entity: type,
    vault_docs: list[Document],
):
    """指定した document_id を参照する Document 一覧を取得できるか確認.

    観点1: リンク元のみ返り、リンク先は含まれない
    観点2: 返ってきた Document の id・page_content・metadata が登録値と一致する
    観点3: 存在しない document_id を指定すると空リストが返る
    """  # noqa: E501
    # 試験準備
    raw_entity = pg_entity
    doc_a = next(
        (d for d in vault_docs if d.metadata.get("forward_links")),
        None,
    )
    if doc_a is None:
        pytest.skip("forward_links を持つDocumentが vault_docs にない")
    link_tgt_id = doc_a.metadata["forward_links"][0]
    doc_b = next((d for d in vault_docs if d.id == link_tgt_id), None)
    if doc_b is None:
        pytest.skip("forward_links のリンク先が vault_docs にない")
    VaultDb.sync([doc_a, doc_b], pg_cxt.get_engine(), raw_entity)

    # 試験実施
    results = pg_obsidian_retriever.get_backlinks(link_tgt_id)

    # 結果検証
    # 観点1
    assert len(results) == 1
    # 観点2
    assert results[0].id == doc_a.id
    assert results[0].page_content == doc_a.page_content
    assert results[0].metadata == doc_a.metadata
    # 観点3
    assert pg_obsidian_retriever.get_backlinks(path_to_document_id("nonexistent_target.md")) == []


@pytest.mark.integration
def test_sync_chunks_01(
    pg_cxt: PostgresStoreContext,
    pg_obsidian_retriever: ObsidianLlamaRetriever,
    pg_entity: type,
    vault_docs: list[Document],
):
    """Vault テーブルと Chunk テーブルの同期ができるか確認.

    操作対象外 Document (noise) を含む状態で各操作を実行し、
    操作した Document のチャンクのみが変化し noise のチャンクが不変であるか確認。

    観点1: 追加後 sync_chunks の後、target と noise が正しい全項目で取得できる
    観点2: 更新後 sync_chunks の後、target が新コンテンツで取得でき、noise は不変
    観点3: 削除後 sync_chunks の後、target が取得できなくなり、noise は不変
    """
    # 試験準備
    raw_entity = pg_entity
    noise_doc = vault_docs[0]
    target_doc = vault_docs[1]
    assert noise_doc.id is not None
    assert target_doc.id is not None

    # --- ステップ1: 追加 ---
    VaultDb.sync([noise_doc, target_doc], pg_cxt.get_engine(), raw_entity)
    pg_obsidian_retriever.sync_chunks()

    # 観点1
    results = pg_obsidian_retriever.get_documents_by_ids([noise_doc.id, target_doc.id])
    noise_result = next(r for r in results if r.id == noise_doc.id)
    target_result = next(r for r in results if r.id == target_doc.id)
    assert noise_result.page_content == noise_doc.page_content
    assert noise_result.metadata == noise_doc.metadata
    assert target_result.page_content == target_doc.page_content
    assert target_result.metadata == target_doc.metadata

    # --- ステップ2: 更新 ---
    new_content = target_doc.page_content + " 更新版"
    updated_target = Document(
        id=target_doc.id,
        page_content=new_content,
        metadata=target_doc.metadata,
    )
    VaultDb.sync([noise_doc, updated_target], pg_cxt.get_engine(), raw_entity)
    pg_obsidian_retriever.sync_chunks()

    # 観点2
    target_result = pg_obsidian_retriever.get_documents_by_ids([target_doc.id])[0]
    assert target_result.page_content == updated_target.page_content
    assert target_result.metadata == updated_target.metadata
    noise_result = pg_obsidian_retriever.get_documents_by_ids([noise_doc.id])[0]
    assert noise_result.page_content == noise_doc.page_content
    assert noise_result.metadata == noise_doc.metadata

    # --- ステップ3: 削除 ---
    VaultDb.sync([noise_doc], pg_cxt.get_engine(), raw_entity)
    pg_obsidian_retriever.sync_chunks()

    # 観点3
    assert pg_obsidian_retriever.get_documents_by_ids([target_doc.id]) == []
    noise_result = pg_obsidian_retriever.get_documents_by_ids([noise_doc.id])[0]
    assert noise_result.page_content == noise_doc.page_content
    assert noise_result.metadata == noise_doc.metadata
