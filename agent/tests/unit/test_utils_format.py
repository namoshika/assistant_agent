from llama_index.core import Document

import assistant_agent.utils.format as fmt


def test_format_doc_ids_01():
    """Document リストから document_id と path の一覧文字列を返せるか確認.

    観点1: 戻り値が文字列である
    観点2: document_id と path が含まれ、page_content は含まれない
    """
    # 試験準備
    docs = [
        Document(
            id_="doc-id-1",
            text="ノート本文",
            metadata={"file_path": "notes/idea.md"},
        ),
        Document(
            id_="doc-id-2",
            text="別のノート",
            metadata={"file_path": "folder/meeting.md"},
        ),
    ]

    # 試験実施
    result = fmt.format_doc_ids(docs)

    # 結果検証
    # 観点1
    assert isinstance(result, str)
    # 観点2
    assert "doc-id-1" in result
    assert "doc-id-2" in result
    assert "notes/idea.md" in result
    assert "folder/meeting.md" in result
    assert "ノート本文" not in result
    assert "別のノート" not in result


def test_format_doc_list_01():
    """ContentsWithFrontmatter リストから整形済み文字列を返せるか確認.

    観点1: 戻り値が文字列である
    観点2: title, contents, frontmatter のキー値, forward_links の id と basename が含まれる
    観点3: forward_links なし（キー自体なし）でも正常に文字列を返す
    """
    # 試験準備
    doc_with_links = fmt.ContentsWithFrontmatter(
        title="note.md",
        contents="ノート本文",
        frontmatter={
            "file_path": "folder/note.md",
            "tags": "YAML 日本語テキスト",
            "forward_links": {"linked-id-1": "linked1.md", "linked-id-2": "linked2.md"},
        },
    )
    doc_without_links = fmt.ContentsWithFrontmatter(
        title="plain.md",
        contents="リンクなし本文",
        frontmatter={"file_path": "folder/plain.md"},
    )

    # 試験実施
    result = fmt.format_doc_list([doc_with_links, doc_without_links])

    # 結果検証
    # 観点1
    assert isinstance(result, str)
    # 観点2
    assert "note.md" in result
    assert "ノート本文" in result
    assert "YAML 日本語テキスト" in result
    assert "folder/note.md" in result
    assert "linked-id-1" in result
    assert "linked-id-2" in result
    assert "linked1.md" in result
    assert "linked2.md" in result
    # 観点3
    assert "リンクなし本文" in result
