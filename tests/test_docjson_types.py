from mypkg.core.docjson_types import (
    ContentBlock,
    ContentBlockType,
    DocumentDocJSON,
    DocumentMetadata,
    InlineImageData,
    Section,
)


def test_content_block_roundtrip_without_bbox_or_diagram():
    block = ContentBlock(
        id="p1",
        type=ContentBlockType.PARAGRAPH,
        doc_index=1,
        text="본문",
    )
    serialized = block.to_dict()

    assert "bbox" not in serialized
    assert "diagram" not in serialized

    restored = ContentBlock.from_dict(serialized)
    assert restored.id == block.id
    assert restored.type == ContentBlockType.PARAGRAPH
    assert restored.doc_index == 1
    assert restored.text == "본문"


def test_section_to_dict_without_bbox_controls():
    section = Section(
        id="s1",
        number="1",
        title="제목",
        doc_index=0,
    )
    section.blocks = [
        ContentBlock(id="p0", type="paragraph", doc_index=0, text="본문"),
        ContentBlock(id="p1", type="paragraph", doc_index=1, text="추가"),
    ]
    child = Section(
        id="s1_1",
        number="1.1",
        title="소제목",
        doc_index=1,
    )
    child.span = [1, 2]
    section.subsections.append(child)
    section.span = [0, 3]

    payload = section.to_dict()
    assert "bbox" not in payload["blocks"][0]
    assert payload["blocks"][0]["id"] == "p0"
    # doc_index=1 블록은 자식 섹션 span에 포함되므로 제외됨
    assert all(block["id"] != "p1" for block in payload["blocks"])


def test_document_docjson_metadata_toggle():
    docjson = DocumentDocJSON(
        version="1.0",
        metadata=DocumentMetadata(title="테스트"),
        sections=[],
        inline_images=[InlineImageData(rid="r1")],
        include_metadata=False,
    )
    payload = docjson.to_dict()
    assert "metadata" not in payload
    assert payload["version"] == "1.0"

