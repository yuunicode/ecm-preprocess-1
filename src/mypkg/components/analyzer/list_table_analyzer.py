"""리스트/테이블 컴포넌트 분석기.

sanitizer 단계에서 바로 활용할 수 있도록 리스트/테이블 요약을 생성하고,
필요 시 `_comp` 디렉터리에 저장하는 도우미 함수를 제공한다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

from mypkg.core.io import (
    save_list_components_from_sanitized,
    save_table_components_from_sanitized,
)
from mypkg.components.sanitizer.table_sanitizer import TableSanitizer
from mypkg.core.parser import TableRecord, ParagraphRecord


def format_list_component(title: str, items: List[str]) -> str:
    """리스트 출력을 문자열로 합치는 기본 포맷터."""

    lines: List[str] = []
    if title:
        lines.append(f"제목: {title}")
    for idx, item in enumerate(items, start=1):
        clean = (item or "").strip()
        if not clean:
            continue
        lines.append(f"{idx}. {clean}")
    return "\n".join(lines)


def _is_list_bullet(paragraph: Dict[str, Any]) -> bool:
    """스타일이 'List Bullet'인지 확인한다."""

    style = (paragraph.get("style") or "").strip()
    return style == "List Bullet"


def _sort_by_doc_index(paragraphs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """doc_index 기준으로 문단을 정렬한다."""

    return sorted(paragraphs, key=lambda p: (p.get("doc_index") is None, p.get("doc_index", 0)))


def analyze_lists(
    paragraphs: List[Dict[str, Any]],
    formatter: Callable[[str, List[str]], str] | None = None,
) -> Tuple[List[Dict[str, Any]], set[int]]:
    """List Bullet 스타일 문단을 묶어 리스트 컴포넌트를 생성한다."""

    fmt = formatter or format_list_component
    ordered_paras = _sort_by_doc_index(paragraphs)
    by_doc_index = {p.get("doc_index"): p for p in ordered_paras if p.get("doc_index") is not None}
    components: List[Dict[str, Any]] = []
    consumed: set[int] = set()

    i = 0
    while i < len(ordered_paras):
        para = ordered_paras[i]
        if not _is_list_bullet(para):
            i += 1
            continue

        group: List[Dict[str, Any]] = []
        start_idx = i
        last_doc_index = None

        while i < len(ordered_paras):
            candidate = ordered_paras[i]
            if not _is_list_bullet(candidate):
                break
            doc_index = candidate.get("doc_index")
            if last_doc_index is not None and doc_index is not None:
                if doc_index != last_doc_index + 1:
                    break
            group.append(candidate)
            if doc_index is not None:
                consumed.add(doc_index)
                last_doc_index = doc_index
            i += 1

        if not group:
            i += 1
            continue

        first_doc_index = group[0].get("doc_index")
        title = ""
        if isinstance(first_doc_index, int):
            prev_para = by_doc_index.get(first_doc_index - 1)
            if prev_para:
                title = (prev_para.get("text") or "").strip()

        items = [(p.get("text") or "").strip() for p in group]
        formatted_text = fmt(title, items)
        component_id = first_doc_index if isinstance(first_doc_index, int) else len(components)

        components.append({
            "id": f"list_{component_id}",
            "type": "list",
            "doc_index": first_doc_index,
            "text": formatted_text,
            "list_data": {
                "ordered": False,
                "items": items,
            },
        })

    return components, consumed


def analyze_tables(tables: List[TableRecord], sanitizer: TableSanitizer, paragraphs: List[ParagraphRecord] = []) -> List[Dict[str, Any]]:
    """테이블 정보를 ContentBlock(dict) 형태로 변환한다."""
    sanitized_tables = sanitizer.apply(tables, paragraphs)
    blocks: List[Dict[str, Any]] = []
    for table in sanitized_tables:
        doc_index = table.get("doc_index", -1)
        blocks.append({
            "id": f"table_{table.get('tid', doc_index)}",
            "type": "table",
            "doc_index": doc_index,
            "text": table.get("preceding_text"),
            "table_data": table,
            "list_data": None,
        })
    return blocks


def emit_list_table_components_from_sanitized(
    paragraphs: List[ParagraphRecord],
    tables: List[TableRecord],
    sanitized_path: str | Path,
    basename: str,
) -> Dict[str, str]:
    """리스트/테이블 컴포넌트를 생성해 `_comp` 경로에 저장한다."""

    sanitizer = TableSanitizer()
    lists, consumed = analyze_lists(paragraphs)
    table_blocks = analyze_tables(tables, sanitizer, paragraphs)
    list_path = save_list_components_from_sanitized(
        {"lists": lists, "consumed": sorted(consumed)}, sanitized_path, basename
    )
    table_path = save_table_components_from_sanitized(
        {"tables": table_blocks}, sanitized_path, basename
    )
    return {"list_path": str(list_path), "table_path": str(table_path)}
