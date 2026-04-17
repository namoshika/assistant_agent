import pytest
from llama_index.core import Document
from llama_index.core.vector_stores.types import FilterOperator, MetadataFilter, MetadataFilters

from assistant_agent.entities import postgres
from assistant_agent.entities.base import VaultUtils
from assistant_agent.loaders.obsidian import path_to_document_id
from assistant_agent.services import VaultObsidianRetriever
from assistant_agent.utils.store_context import PostgresStoreContext


@pytest.mark.integration
def test_search_documents_01(
    pg_cxt: PostgresStoreContext,
    pg_retriever_obs: VaultObsidianRetriever,
    pg_entity_obs: type[postgres.ObsidianFields],
    docs_obs: list[Document],
):
    """指定したクエリに類似する Document (id_ + metadata のみ) を取得できるか確認.

    前提: VaultUtils.sync + sync_chunks 済み

    観点1: フィルタなし — 1件以上の結果が返り、対象の document_id と
        metadata["file_path"] が含まれる
    観点2: フィルタあり — 指定条件に合致する Document のみ返る
        ケース1: date >= "2026-01-01 00:00:00" で絞ると 2025 年以前が除外される
        ケース2: path text_match "01_Inbox" で絞ると該当フォルダのみ返る
    """
    # 試験準備
    raw_entity = pg_entity_obs
    VaultUtils.sync(docs_obs, pg_cxt.get_engine(), raw_entity)
    pg_retriever_obs.sync_chunks()
    doc = docs_obs[0]

    # 試験実施・結果検証
    # 観点1
    results = pg_retriever_obs.search_documents(doc.text[:30], top_k=10)
    assert len(results) >= 1
    matched = next((r for r in results if r.id_ == doc.id_), None)
    assert matched is not None
    assert matched.metadata["file_path"] == doc.metadata["file_path"]

    # 観点2 ケース1: date フィルタ
    date_filter = MetadataFilters(
        filters=[
            MetadataFilter(key="date", value="2026-01-01 00:00:00", operator=FilterOperator.GTE)
        ]
    )
    results_date = pg_retriever_obs.search_documents(doc.text[:30], top_k=10, filters=date_filter)
    assert all(d.metadata["date"] >= "2026-01-01 00:00:00" for d in results_date)

    # 観点2 ケース2: path フィルタ
    path_filter = MetadataFilters(
        filters=[
            MetadataFilter(key="file_path", value="01_Inbox", operator=FilterOperator.TEXT_MATCH)
        ]
    )
    results_path = pg_retriever_obs.search_documents("project", top_k=10, filters=path_filter)
    assert len(results_path) >= 1
    assert all("01_Inbox" in d.metadata["file_path"] for d in results_path)


@pytest.mark.integration
def test_get_documents_by_ids_01(
    pg_cxt: PostgresStoreContext,
    pg_retriever_obs: VaultObsidianRetriever,
    pg_entity_obs: type,
    docs_obs: list[Document],
):
    """document_ids に対応する Document を取得できるか確認.

    観点1: 既存 id を指定すると id, text, metadata の全項目が一致する
    観点2: 存在しない document_id を指定すると空リストが返る
    """
    # 試験準備
    raw_entity = pg_entity_obs
    doc = docs_obs[0]
    VaultUtils.sync([doc], pg_cxt.get_engine(), raw_entity)

    # 試験実施
    assert doc.id_ is not None
    results = pg_retriever_obs.get_documents_by_ids([doc.id_])

    # 結果検証
    # 観点1
    assert len(results) == 1
    assert results[0].id_ == doc.id_
    assert results[0].text == doc.text
    assert results[0].metadata == doc.metadata
    # 観点2
    assert pg_retriever_obs.get_documents_by_ids(["nonexistent-uuid"]) == []


@pytest.mark.integration
def test_get_backlinks_01(
    pg_cxt: PostgresStoreContext,
    pg_retriever_obs: VaultObsidianRetriever,
    pg_entity_obs: type,
    docs_obs: list[Document],
):
    """指定した document_id を参照する Document 一覧を取得できるか確認.

    観点1: リンク元のみ返り、リンク先は含まれない
    観点2: 返ってきた Document の id, text, metadata が登録値と一致する
    観点3: 存在しない document_id を指定すると空リストが返る
    """  # noqa: E501
    # 試験準備
    raw_entity = pg_entity_obs
    doc_a = next(
        (d for d in docs_obs if d.metadata.get("forward_links")),
        None,
    )
    if doc_a is None:
        pytest.skip("forward_links を持つDocumentが vault_docs にない")
    link_tgt_id = doc_a.metadata["forward_links"][0]
    doc_b = next((d for d in docs_obs if d.id_ == link_tgt_id), None)
    if doc_b is None:
        pytest.skip("forward_links のリンク先が vault_docs にない")
    VaultUtils.sync([doc_a, doc_b], pg_cxt.get_engine(), raw_entity)

    # 試験実施
    results = pg_retriever_obs.get_backlinks(link_tgt_id)

    # 結果検証
    # 観点1
    assert len(results) == 1
    # 観点2
    assert results[0].id_ == doc_a.id_
    assert results[0].text == doc_a.text
    assert results[0].metadata == doc_a.metadata
    # 観点3
    assert pg_retriever_obs.get_backlinks(path_to_document_id("nonexistent_target.md")) == []


@pytest.mark.integration
def test_sync_chunks_01(
    pg_cxt: PostgresStoreContext,
    pg_retriever_obs: VaultObsidianRetriever,
    pg_entity_obs: type,
    docs_obs: list[Document],
):
    """Vault テーブルと Chunk テーブルの同期ができるか確認.

    操作対象外 Document (noise) を含む状態で各操作を実行し、
    操作した Document のチャンクのみが変化し noise のチャンクが不変であるか確認。

    観点1: 追加後 sync_chunks の後、target と noise が正しい全項目で取得できる
    観点2: 更新後 sync_chunks の後、target が新コンテンツで取得でき、noise は不変
    観点3: 削除後 sync_chunks の後、target が取得できなくなり、noise は不変
    """
    # 試験準備
    raw_entity = pg_entity_obs
    noise_doc = docs_obs[0]
    target_doc = docs_obs[1]
    assert noise_doc.id_ is not None
    assert target_doc.id_ is not None

    # --- ステップ1: 追加 ---
    VaultUtils.sync([noise_doc, target_doc], pg_cxt.get_engine(), raw_entity)
    pg_retriever_obs.sync_chunks()

    # 観点1
    results = pg_retriever_obs.get_documents_by_ids([noise_doc.id_, target_doc.id_])
    noise_result = next(r for r in results if r.id_ == noise_doc.id_)
    target_result = next(r for r in results if r.id_ == target_doc.id_)
    assert noise_result.text == noise_doc.text
    assert noise_result.metadata == noise_doc.metadata
    assert target_result.text == target_doc.text
    assert target_result.metadata == target_doc.metadata

    # --- ステップ2: 更新 ---
    new_content = target_doc.text + " 更新版"
    updated_target = Document(
        id_=target_doc.id_,
        text=new_content,
        metadata=target_doc.metadata,
    )
    VaultUtils.sync([noise_doc, updated_target], pg_cxt.get_engine(), raw_entity)
    pg_retriever_obs.sync_chunks()

    # 観点2
    target_result = pg_retriever_obs.get_documents_by_ids([target_doc.id_])[0]
    assert target_result.text == updated_target.text
    assert target_result.metadata == updated_target.metadata
    noise_result = pg_retriever_obs.get_documents_by_ids([noise_doc.id_])[0]
    assert noise_result.text == noise_doc.text
    assert noise_result.metadata == noise_doc.metadata

    # --- ステップ3: 削除 ---
    VaultUtils.sync([noise_doc], pg_cxt.get_engine(), raw_entity)
    pg_retriever_obs.sync_chunks()

    # 観点3
    assert pg_retriever_obs.get_documents_by_ids([target_doc.id_]) == []
    noise_result = pg_retriever_obs.get_documents_by_ids([noise_doc.id_])[0]
    assert noise_result.text == noise_doc.text
    assert noise_result.metadata == noise_doc.metadata
