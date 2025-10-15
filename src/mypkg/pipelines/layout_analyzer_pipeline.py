
# mypkg/pipeline/layout_analyzer_pipeline.py
# Sanitized JSON을 받아 레이아웃 분석(섹션, 블록, 메타데이터)을 수행하고 DocJSON을 생성하는 파이프라인.
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from mypkg.components.analyzer.document_metadata_analyzer import DocumentMetadataAnalyzer
from mypkg.components.analyzer.layout_assembler import (
    assign_blocks_to_sections,
    build_inline_image_blocks,
    build_paragraph_blocks,
)
from mypkg.components.analyzer.list_table_analyzer import (
    analyze_lists,
    analyze_tables,
    emit_list_table_components_from_sanitized,
)
from mypkg.components.analyzer.section_analyzer import iter_sections
from mypkg.core.config.docjson_config import DocJsonConfig
from mypkg.core.docjson_types import (
    ContentBlock,
    DocumentDocJSON,
    DocumentMetadata,
    InlineImageData,
    Section,
)
from mypkg.core.io import (
    base_dir_from_sanitized,
    component_paths_from_sanitized,
    docjson_output_path_from_sanitized,
    load_available_components_from_sanitized,
    load_json,
    meta_path_from_sanitized,
    resolve_sanitized_path,
    save_blocks_components_from_sanitized,
    save_json,
)
from mypkg.components.sanitizer.table_sanitizer import TableSanitizer
from mypkg.core.parser import ParagraphRecord, TableRecord


def _blocks_from_dicts(blocks: List[Dict[str, Any]]) -> List[ContentBlock]:
    return [ContentBlock.from_dict(item) for item in blocks or []]


def run_pipeline(
    sanitized_path: str | Path,
    out_docjson_path: str | Path,
    emit_components: bool = True,
    use_components: bool = True,
    doc_version: str | None = None,
    tenant_key: str | None = None,
    docjson_config: DocJsonConfig | None = None,
) -> DocumentDocJSON:
    """레이아웃 파이프라인 실행."""

    sanitized_path = resolve_sanitized_path(sanitized_path)
    basename = base_dir_from_sanitized(sanitized_path).name  # {base_name}/{version}/_sanitized → basename

    out_path = Path(out_docjson_path) if out_docjson_path else None
    if out_path is None or not out_path.suffix:
        out_docjson_path = docjson_output_path_from_sanitized(sanitized_path, basename)
    else:
        out_docjson_path = out_path.resolve()
    Path(out_docjson_path).parent.mkdir(parents=True, exist_ok=True)

    raw_ir = json.loads(sanitized_path.read_text(encoding="utf-8"))
    paragraphs = [ParagraphRecord(**p) for p in raw_ir.get("paragraphs", [])]
    tables_ir = [TableRecord(**t) for t in raw_ir.get("tables", [])]

    config = docjson_config or DocJsonConfig()
    sections: List[Section] = config.build_sections(paragraphs)
    heading_idx = {section.doc_index for section in iter_sections(sections)}

    if emit_components:
        emit_list_table_components_from_sanitized(paragraphs, tables_ir, sanitized_path, basename)

    components: Dict[str, Any] = (
        load_available_components_from_sanitized(sanitized_path, basename) if use_components else {}
    )

    if "lists" in components:
        consumed = set(components.get("consumed", []))
        list_blocks = []
        for idx, info in enumerate(components.get("lists", []) or []):
            doc_index = info.get("doc_index")
            text = info.get("text") or ""
            block_id = f"list_{doc_index if isinstance(doc_index, int) else idx}"
            list_blocks.append(
                ContentBlock(
                    id=block_id,
                    type="list",
                    doc_index=doc_index,
                    text=text,
                )
            )
    else:
        list_block_dicts, consumed = analyze_lists(paragraphs)
        list_blocks = _blocks_from_dicts(list_block_dicts)

    sanitizer = TableSanitizer()
    if "tables" in components:
        table_blocks = _blocks_from_dicts(components["tables"])
    else:
        table_blocks = _blocks_from_dicts(analyze_tables(tables_ir, sanitizer, paragraphs))

    inline_images_payload = components.get("inline_images")
    if isinstance(inline_images_payload, dict):
        inline_images_payload = inline_images_payload.get("inline_images", [])
    if inline_images_payload is None:
        inline_images_payload = raw_ir.get("inline_images", [])
    inline_image_blocks = build_inline_image_blocks(inline_images_payload)

    paragraph_blocks = build_paragraph_blocks(
        paragraphs,
        skip_docidx=consumed,
        heading_idx=heading_idx,
    )

    all_blocks: List[ContentBlock] = (
        paragraph_blocks + list_blocks + table_blocks + inline_image_blocks
    )
    all_blocks.sort(key=lambda block: (block.doc_index, block.id))
    assign_blocks_to_sections(sections, all_blocks)

    blocks_payload = {"blocks": [block.to_dict() for block in all_blocks]}
    save_blocks_components_from_sanitized(blocks_payload, sanitized_path, basename)

    if config.include_metadata:
        metadata = DocumentMetadataAnalyzer(raw_ir, basename, Path(sanitized_path)).analyze()
        save_json(metadata.to_dict(), meta_path_from_sanitized(sanitized_path, basename))
    else:
        metadata = DocumentMetadata()

    docjson = DocumentDocJSON(
        version=(doc_version or "0.1.0"),
        metadata=metadata,
        blocks=all_blocks,
        sections=sections,
        inline_images=[
            InlineImageData.from_dict(item)
            if not isinstance(item, InlineImageData)
            else item
            for item in inline_images_payload or []
        ],
        include_metadata=config.include_metadata,
    )

    save_json(docjson.to_dict(), out_docjson_path)
    return docjson
