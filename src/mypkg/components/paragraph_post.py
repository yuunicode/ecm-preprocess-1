"""
sanitized JSON에서 문단 경로(path)와 본문(context)을 추출하는 후처리 스크립트.

헤딩 스타일(heading 1, heading 2, 소제목 등)을 계층 구조로 변환하고,
같은 헤딩 아래의 일반 문단은 하나의 context로 묶어서 path/context 쌍을 생성한다.
테이블은 아직 별도로 처리하지 않으며, 구간을 나누는 경계로만 활용한다.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Set
from mypkg.components.parser.xml_parser import build_table_context_rows


def header_entries_to_strings(rows: List[List[Dict[str, Any]]]) -> List[str]:
    """헤더-셀 매핑 결과를 사람이 읽기 쉬운 문자열 목록으로 변환한다."""
    contexts: List[str] = []
    for row_entries in rows:
        parts: List[str] = []
        for entry in row_entries:
            header_text = (entry.get("header") or "").strip()
            cell_text = (entry.get("cell") or "").strip()
            if header_text and cell_text:
                parts.append(f"{header_text} : {cell_text}")
            elif header_text:
                parts.append(header_text)
            elif cell_text:
                parts.append(cell_text)
        joined = " | ".join(part for part in parts if part)
        if joined:
            contexts.append(joined)
    return contexts


# 숫자 패턴(heading 1, heading 2 등)을 추출하기 위한 정규식
HEADING_PATTERN = re.compile(r"(\d+)")

# 공백 정규화용 패턴

WHITESPACE_RE = re.compile(r"\s+")
# 붙어 있는 단어를 완화하기 위한 패턴 목록
GLUED_PATTERNS = [
    re.compile(r"(?<=\w)(?=when[A-Z])"),
    re.compile(r"(?<=\w)(?=as[A-Z])"),
    re.compile(r"(?<=\w)(?=and[A-Z])"),
    re.compile(r"(?<=\w)(?=or[A-Z])"),
    re.compile(r"(?<=\w)(?=Yes\b)"),
    re.compile(r"(?<=\w)(?=No\b)"),
    re.compile(r"(?<=[a-z])(?=[A-Z])"),
]
IMAGE_TOKEN_PATTERN = re.compile(r"\[image:[^\]]+\]")
TABLE_TOKEN_PATTERN = re.compile(r"\[table:[^\]]+\]")


def normalize_text(text: str) -> str:
    """문단/헤딩 텍스트의 공백과 붙어 있는 단어를 정리한다."""
    if not text:
        return ""
    tokens: Dict[str, str] = {}

    def _capture(match: re.Match[str]) -> str:
        placeholder = f"__IMG_TOKEN_{len(tokens)}__"
        tokens[placeholder] = match.group(0)
        return placeholder

    protected = IMAGE_TOKEN_PATTERN.sub(_capture, text)
    cleaned = WHITESPACE_RE.sub(" ", protected).strip()
    for pattern in GLUED_PATTERNS:
        cleaned = pattern.sub(" ", cleaned)
    cleaned = WHITESPACE_RE.sub(" ", cleaned).strip()
    for placeholder, original in tokens.items():
        cleaned = cleaned.replace(placeholder, original)
    return cleaned


def strip_inline_tokens(text: str) -> str:
    """image/table 토큰을 제거한 실질 텍스트를 반환한다."""
    without_images = IMAGE_TOKEN_PATTERN.sub("", text)
    without_tables = TABLE_TOKEN_PATTERN.sub("", without_images)
    return without_tables.strip()


def last_non_sub_level(stack: List[Dict[str, Any]]) -> int:
    """소제목을 제외한 가장 최근 헤딩 레벨을 찾는다."""
    for entry in reversed(stack):
        if not entry["is_subheading"]:
            return entry["level"]
    return 0


def heading_entry(
    style: Optional[str],
    text: Optional[str],
    stack: List[Dict[str, Any]],
) -> Optional[Tuple[int, bool]]:
    """헤딩 스타일이면 (레벨, 소제목 여부)를 반환하고 아니면 None."""
    if not style:
        return None
    if text and "[image:" in text:
        return None
    style_stripped = style.strip()
    lower = style_stripped.lower()
    if "소제목" in style_stripped or "타이틀" in style_stripped or "title" in lower:
        base = last_non_sub_level(stack)
        level = max(base + 1, 1)
        return level, True
    if "heading" in lower:
        match = HEADING_PATTERN.search(lower)
        level = int(match.group(1)) if match else 1
        return level, False
    return None


def classify_list_style(style: Optional[str]) -> str:
    """리스트 스타일을 bullet/continue/none으로 분류한다."""
    if not style:
        return "none"
    lowered = style.strip().lower()
    if "list" not in lowered:
        return "none"
    if "continue" in lowered or "continued" in lowered:
        return "continue"
    return "start"


def iter_events(
    paragraphs: Iterable[Dict[str, Any]],
    tables: Iterable[Dict[str, Any]],
) -> Iterable[Tuple[int, str, Dict[str, Any]]]:
    """문단과 테이블을 doc_index 기준으로 정렬하기 위한 이벤트 스트림 생성."""
    for para in paragraphs:
        yield (para.get("doc_index") or 0, "paragraph", para)
    for table in tables:
        yield (table.get("doc_index") or 0, "table", table)


def build_contexts(data: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """sanitized JSON으로부터 문단 및 테이블 컨텍스트 목록을 생성한다."""
    paragraphs = data.get("paragraphs", []) or []
    tables = data.get("tables", []) or []

    events = sorted(iter_events(paragraphs, tables), key=lambda item: item[0])

    paragraph_contexts: List[Dict[str, Any]] = []
    table_no_border_contexts: List[Dict[str, Any]] = []
    table_image_contexts: List[Dict[str, Any]] = []
    table_header_contexts: List[Dict[str, Any]] = []
    table_others_contexts: List[Dict[str, Any]] = []
    heading_stack: List[Dict[str, Any]] = []
    buffer: List[Dict[str, Any]] = []
    current_path: Optional[str] = None
    list_group: Optional[Dict[str, Any]] = None

    def flush_buffer() -> None:
        """누적된 본문을 contexts 목록에 저장한다."""
        nonlocal buffer, current_path
        if not buffer:
            return
        buffer_snapshot = list(buffer)
        buffer.clear()
        context_text = " ".join(item["text"] for item in buffer_snapshot if item["text"]).strip()
        if not context_text:
            return
        path_str = current_path or ""
        doc_indices = sorted({item["doc_index"] for item in buffer_snapshot if item["doc_index"] is not None})
        bold_texts: List[str] = []
        seen: set[str] = set()
        for item in buffer_snapshot:
            for emph in item["bold_texts"]:
                norm = emph.strip()
                if not norm:
                    continue
                if norm in seen:
                    continue
                seen.add(norm)
                bold_texts.append(norm)
        paragraph_contexts.append(
            {
                "path": path_str,
                "context": context_text,
                "doc_indices": doc_indices,
                "bold_texts": bold_texts,
            }
        )

    def flush_list_group() -> None:
        """list bullet/continue 시퀀스를 contexts에 반영한다."""
        nonlocal list_group
        if not list_group:
            return
        text_parts = list_group.get("texts") or []
        context_text = " ".join(part for part in text_parts if part).strip()
        if context_text:
            paragraph_contexts.append(
                {
                    "path": list_group.get("path") or "",
                    "context": context_text,
                    "doc_indices": sorted(list(list_group.get("doc_indices") or [])),
                    "bold_texts": list_group.get("bold_texts") or [],
                }
            )
        list_group = None

    def update_current_path() -> None:
        """현재 헤딩 스택으로 path 문자열을 갱신한다."""
        nonlocal current_path
        current_path = " > ".join(entry["text"] for entry in heading_stack) if heading_stack else ""

    def preprocess_table_cells(table: Dict[str, Any]) -> List[List[Dict[str, Any]]]:
        """텍스트+이미지 셀을 텍스트로 변환한 뒤 셀 정보를 반환한다."""
        data_matrix = table.get("data") or []
        processed_rows: List[List[Dict[str, Any]]] = []
        for r_idx, row in enumerate(data_matrix):
            processed_row: List[Dict[str, Any]] = []
            if not row:
                processed_rows.append(processed_row)
                continue
            for c_idx, cell in enumerate(row):
                if not isinstance(cell, dict):
                    processed_row.append(
                        {
                            "text": "",
                            "clean_text": "",
                            "images": [],
                            "has_image": False,
                            "has_text": False,
                            "row": r_idx,
                            "col": c_idx,
                        }
                    )
                    continue
                raw_text = cell.get("text") or ""
                normalized_text = normalize_text(raw_text)
                images = list((cell.get("images") or []))
                clean_text = strip_inline_tokens(normalized_text)
                has_inline_text = bool(clean_text)
                if images and has_inline_text:
                    image_tokens = " ".join(f"[image:{image_id}]" for image_id in images)
                    normalized_text = f"{normalized_text} {image_tokens}".strip()
                    images = []
                    clean_text = strip_inline_tokens(normalized_text)
                cell["text"] = normalized_text
                cell["images"] = images
                processed_row.append(
                    {
                        "text": normalized_text,
                        "clean_text": clean_text,
                        "images": images,
                        "has_image": bool(images),
                        "has_text": bool(clean_text),
                        "row": r_idx,
                        "col": c_idx,
                    }
                )
            processed_rows.append(processed_row)
        return processed_rows

    def extract_table_image_contexts(
        table: Dict[str, Any],
        path: str,
        processed_rows: Optional[List[List[Dict[str, Any]]]] = None,
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        if processed_rows is None:
            processed_rows = preprocess_table_cells(table)
        if not processed_rows:
            return results

        def format_image_tokens(images: List[str]) -> str:
            tokens = [f"[image:{rid}]" for rid in images]
            return " ".join(tokens).strip()

        # Pre-processing pass, as requested by the user.
        # Cells with both images and text are converted to text-only cells
        # by inlining the image tokens. This makes them available for subsequent pairing.
        for row in processed_rows:
            for cell in row:
                if cell["has_image"] and cell["has_text"]:
                    token_str = format_image_tokens(cell["images"])
                    # New format: "text [image:id]" with a space, no parentheses.
                    new_text = f"{cell['clean_text']} {token_str}".strip()
                    cell["text"] = new_text
                    cell["clean_text"] = new_text
                    cell["has_image"] = False  # Mark as text-only for pairing logic
                    cell["images"] = []

        # Re-calculate row_info after the pre-processing pass
        row_info: List[Dict[str, Any]] = []
        for row in processed_rows:
            has_image_row = any(cell.get("has_image") for cell in row)
            has_text_row = any(cell.get("has_text") for cell in row)
            combined_text_parts = [cell["clean_text"] for cell in row if cell["has_text"]]
            combined_raw_parts = [cell["text"] for cell in row if cell["has_text"]]
            row_info.append({
                "text": " ".join(part for part in combined_text_parts if part).strip(),
                "raw_text": " ".join(part for part in combined_raw_parts if part).strip(),
                "has_image": has_image_row,
                "has_text": has_text_row,
            })

        used_cells: set[Tuple[int, int]] = set()
        used_rows: set[int] = set()
        doc_idx = table.get("doc_index")
        tid = table.get("tid")

        def make_entry(context_text: str, images: List[str]) -> Dict[str, Any]:
            unique_images = list(dict.fromkeys(images))
            context_text = context_text.strip()
            return {
                "path": path or "",
                "context": context_text,
                "doc_index": doc_idx,
                "bold_texts": [],
                "table_tid": tid,
                "images": unique_images,
            }

        allow_pairing = not (table.get("is_rowheader") and table.get("is_colheader"))

        if allow_pairing:
            # Case 3 (same cell) is now handled by the pre-processing pass.
            # The logic below will now operate on the modified cell data.

            # 케이스 1: 이미지 셀 왼쪽, 텍스트 셀 오른쪽
            for row in processed_rows:
                if not row:
                    continue
                r_idx = row[0]["row"]
                for idx, cell in enumerate(row):
                    coord = (r_idx, idx)
                    if cell["has_image"] and not cell["has_text"] and coord not in used_cells:
                        text_parts: List[str] = []
                        raw_parts: List[str] = []
                        for j in range(idx + 1, len(row)):
                            right_cell = row[j]
                            right_coord = (r_idx, j)
                            if right_coord in used_cells:
                                continue
                            if right_cell["has_text"] and not right_cell["has_image"]:
                                text_parts.append(right_cell["clean_text"])
                                raw_parts.append(right_cell["text"])
                        if text_parts:
                            combined_clean = " ".join(text_parts).strip()
                            combined_raw = " ".join(part for part in raw_parts if part).strip()
                            base_text = combined_raw or combined_clean
                            token_str = format_image_tokens(cell["images"])
                            context_text = base_text
                            if token_str:
                                # New format: space instead of " : "
                                context_text = f"{token_str} {base_text}" if base_text else token_str
                            entry = make_entry(context_text, cell["images"])
                            results.append(entry)
                            used_cells.add(coord)
                            used_rows.add(r_idx)
                            for j in range(idx + 1, len(row)):
                                right_cell = row[j]
                                if right_cell["has_text"] and not right_cell["has_image"]:
                                    used_cells.add((r_idx, j))
                                    used_rows.add(r_idx)

            # 케이스 2: 텍스트 행과 이미지 행이 번갈아 나오는 경우
            if not table.get("is_rowheader"):
                r = 0
                total_rows = len(processed_rows)
                while r < total_rows - 1:
                    if r in used_rows or (r + 1) in used_rows:
                        r += 1
                        continue
                    text_info = row_info[r]
                    image_info = row_info[r + 1]
                    if text_info["has_text"] and not text_info["has_image"] and image_info["has_image"] and not image_info["has_text"]:
                        text_row = processed_rows[r]
                        image_row = processed_rows[r + 1]
                        paired = False
                        max_len = max(len(text_row), len(image_row))
                        for col_idx in range(max_len):
                            t_cell = text_row[col_idx] if col_idx < len(text_row) else None
                            i_cell = image_row[col_idx] if col_idx < len(image_row) else None
                            if not t_cell or not i_cell:
                                continue
                            if not t_cell["has_text"]:
                                continue
                            if not i_cell["has_image"]:
                                continue
                            images = list(dict.fromkeys(i_cell["images"]))
                            if not images:
                                continue
                            token_str = format_image_tokens(images)
                            base_text = t_cell["text"].strip()
                            context_text = base_text
                            if token_str:
                                # New format: space instead of " : "
                                context_text = f"{token_str} {base_text}" if base_text else token_str
                            entry = make_entry(context_text, images)
                            results.append(entry)
                            used_cells.add((t_cell["row"], t_cell["col"]))
                            used_cells.add((i_cell["row"], i_cell["col"]))
                            paired = True
                        if paired:
                            used_rows.add(r)
                            used_rows.add(r + 1)
                    r += 1

        return [ctx for ctx in results if ctx.get("context")] 

    def table_cells_to_text(table: Dict[str, Any]) -> str:
        """테이블 셀 텍스트를 단순 연결하여 하나의 문자열로 반환한다."""
        parts: List[str] = []
        for row in table.get("data", []) or []:
            for cell in row or []:
                cell_text = normalize_text((cell or {}).get("text") or "")
                if cell_text:
                    parts.append(cell_text)
        return " ".join(parts).strip()

    for _, kind, payload in events:
        if kind == "paragraph":
            style = payload.get("style")
            text = (payload.get("text") or "").strip()
            entry = heading_entry(style, text, heading_stack)
            text = (payload.get("text") or "").strip()
            if entry is not None:
                level, is_subheading = entry
                flush_buffer()
                flush_list_group()
                while heading_stack and heading_stack[-1]["level"] >= level:
                    heading_stack.pop()
                if text:
                    heading_stack.append(
                        {
                            "level": level,
                            "text": normalize_text(text),
                            "is_subheading": is_subheading,
                        }
                    )
                update_current_path()
            else:
                classification = classify_list_style(style)
                normalized_text = normalize_text(text) if text else ""
                doc_idx = payload.get("doc_index")
                emphasized_raw = payload.get("emphasized") or []
                bold_norm = [normalize_text(e) for e in emphasized_raw if e]

                if classification == "none":
                    flush_list_group()
                    if normalized_text:
                        flush_buffer()
                        buffer.append(
                            {
                                "text": normalized_text,
                                "doc_index": doc_idx,
                                "bold_texts": bold_norm,
                            }
                        )
                        flush_buffer()
                else:
                    flush_buffer()
                    if normalized_text:
                        if list_group is None or classification == "start":
                            flush_list_group()
                            doc_set: Set[int] = set()
                            if doc_idx is not None:
                                doc_set.add(doc_idx)
                            bold_seen: Set[str] = set()
                            bold_list: List[str] = []
                            for norm in bold_norm:
                                if not norm:
                                    continue
                                if norm in bold_seen:
                                    continue
                                bold_seen.add(norm)
                                bold_list.append(norm)
                            list_group = {
                                "path": current_path or "",
                                "texts": [normalized_text],
                                "doc_indices": doc_set,
                                "bold_texts": bold_list,
                                "_bold_seen": bold_seen,
                            }
                        else:
                            list_group["path"] = current_path or list_group.get("path") or ""
                            list_group["texts"].append(normalized_text)
                            doc_set = list_group.get("doc_indices")
                            if isinstance(doc_set, set) and doc_idx is not None:
                                doc_set.add(doc_idx)
                            elif doc_idx is not None:
                                doc_set = {doc_idx}
                                list_group["doc_indices"] = doc_set
                            bold_seen = list_group.get("_bold_seen")
                            if bold_seen is None or not isinstance(bold_seen, set):
                                bold_seen = set()
                                list_group["_bold_seen"] = bold_seen
                            for norm in bold_norm:
                                if not norm:
                                    continue
                                if norm in bold_seen:
                                    continue
                                bold_seen.add(norm)
                                list_group.setdefault("bold_texts", []).append(norm)
                update_current_path()
        else:
            flush_buffer()
            flush_list_group()
            update_current_path()
            processed_rows = preprocess_table_cells(payload)
            if not payload.get("has_borders", True):
                preceding = normalize_text(payload.get("preceding_text") or "")
                table_text = table_cells_to_text(payload)
                if table_text or preceding:
                    doc_idx = payload.get("doc_index")
                    table_no_border_contexts.append(
                        {
                            "path": current_path or "",
                            "context": table_text,
                            "preceding_text": preceding,
                            "doc_index": doc_idx,
                            "bold_texts": [],
                            "table_tid": payload.get("tid"),
                        }
                    )
            image_contexts = extract_table_image_contexts(payload, current_path or "", processed_rows)
            if image_contexts:
                table_image_contexts.extend(image_contexts)
            
            if payload.get("has_borders", True) and not payload.get("is_rowheader") and not payload.get("is_colheader"):
                cell_values = []
                for row in payload.get("data", []):
                    for cell in row or []:
                        cell_text = normalize_text((cell or {}).get("text") or "")
                        if cell_text:
                            cell_values.append(cell_text)
                
                concatenated_text = " ".join(cell_values).strip()
                if concatenated_text:
                    table_others_contexts.append(
                        {
                            "path": current_path or "",
                            "context": concatenated_text,
                            "doc_index": payload.get("doc_index"),
                            "table_tid": payload.get("tid"),
                        }
                    )
            if payload.get("is_rowheader") or payload.get("is_colheader"):
                matrix = payload.get("data") or []
                anchors = payload.get("anchors") or []
                header_rows = build_table_context_rows(
                    matrix,
                    anchors,
                    bool(payload.get("is_rowheader")),
                    bool(payload.get("is_colheader")),
                )
                context_lines = header_entries_to_strings(header_rows)
                if context_lines:
                    for idx, line in enumerate(context_lines):
                        table_header_contexts.append(
                            {
                                "path": current_path or "",
                                "doc_index": payload.get("doc_index"),
                                "table_tid": payload.get("tid"),
                                "context": line,
                                "table_html": payload.get("table_html"),
                                "row_index": idx,
                            }
                        )

    flush_buffer()
    flush_list_group()

    return paragraph_contexts, table_no_border_contexts, table_image_contexts, table_header_contexts, table_others_contexts


def parse_args() -> argparse.Namespace:
    """CLI 실행 시 필요한 인자를 파싱한다."""
    parser = argparse.ArgumentParser(description="sanitized JSON을 VDB 적재용 path/context 형태로 변환한다.")
    parser.add_argument("--input", required=True, type=Path, help="*_sanitized.json 파일 경로")
    parser.add_argument("--paragraph-output", type=Path, help="문단용 결과 JSON 경로")
    parser.add_argument("--table-no-border-output", type=Path, help="테두리 없는 테이블용 결과 JSON 경로")
    parser.add_argument("--table-image-output", type=Path, help="테이블 이미지용 결과 JSON 경로")
    parser.add_argument("--table-combined-output", type=Path, help="테이블 통합 결과 JSON 경로")
    parser.add_argument("--table-header-output", type=Path, help="테이블 헤더 컨텍스트 JSON 경로")
    parser.add_argument("--table-others-output", type=Path, help="테두리 있는 일반 테이블용 결과 JSON 경로")
    parser.add_argument("--output-dir", type=Path, help="기본 출력 디렉터리 (미지정 시 <version>/_for_vdb)")
    return parser.parse_args()


def main() -> None:
    """CLI 진입점."""
    args = parse_args()
    raw = json.loads(args.input.read_text(encoding="utf-8"))
    (
        para_contexts,
        table_no_border_contexts,
        table_image_contexts,
        table_header_contexts,
        table_others_contexts,
    ) = build_contexts(raw)
    img_by_tid: Dict[str, List[Dict[str, Any]]] = {}
    for entry in table_image_contexts:
        tid = entry.get("table_tid")
        if tid:
            img_by_tid.setdefault(tid, []).append(entry)

    table_token_pattern = re.compile(r"\[table:([^\]]+)\]")
    no_border_for_combined: List[Dict[str, Any]] = []
    for no_border_entry in table_no_border_contexts:
        modified_entry = copy.deepcopy(no_border_entry)
        original_context_text = modified_entry.get("context", "") or ""

        def replace_nested_table_token(match: re.Match[str]) -> str:
            nested_tid = match.group(1)
            if nested_tid in img_by_tid:
                image_contexts_for_tid = img_by_tid[nested_tid]
                combined_image_text = " ".join(
                    img_entry.get("context", "").strip()
                    for img_entry in image_contexts_for_tid
                    if img_entry.get("context")
                ).strip()
                return combined_image_text or match.group(0)
            return match.group(0)

        new_context_text = table_token_pattern.sub(replace_nested_table_token, original_context_text)
        modified_entry["context"] = new_context_text.strip()
        no_border_for_combined.append(modified_entry)

    combined_contexts: List[Dict[str, Any]] = []

    def extend_with_priority(entries: Iterable[Dict[str, Any]], source_label: str) -> None:
        for entry in entries:
            cloned = copy.deepcopy(entry)
            cloned["context_source"] = source_label
            if "context" in cloned and isinstance(cloned["context"], str):
                cloned["context"] = cloned["context"].strip()
            combined_contexts.append(cloned)

    extend_with_priority(table_header_contexts, "header")
    extend_with_priority(table_image_contexts, "image")
    extend_with_priority(no_border_for_combined, "no_border")
    extend_with_priority(table_others_contexts, "others")

    sanitized_path = args.input.resolve()
    base_dir = sanitized_path.parent.parent  # .../beta
    out_dir = Path(args.output_dir).resolve() if args.output_dir else (base_dir / "_for_vdb")
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = sanitized_path.stem
    base_stem = stem.removesuffix("_sanitized")

    para_path = Path(args.paragraph_output) if args.paragraph_output else (out_dir / f"{base_stem}_paragraph_contexts.json")
    para_path.parent.mkdir(parents=True, exist_ok=True)
    para_path.write_text(json.dumps({"contexts": para_contexts}, ensure_ascii=False, indent=2), encoding="utf-8")

    table_path = Path(args.table_no_border_output) if args.table_no_border_output else (out_dir / f"{base_stem}_table_no_border_contexts.json")
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.write_text(json.dumps({"contexts": table_no_border_contexts}, ensure_ascii=False, indent=2), encoding="utf-8")

    table_image_path = Path(args.table_image_output) if args.table_image_output else (out_dir / f"{base_stem}_table_image_contexts.json")
    table_image_path.parent.mkdir(parents=True, exist_ok=True)
    table_image_path.write_text(json.dumps({"contexts": table_image_contexts}, ensure_ascii=False, indent=2), encoding="utf-8")

    table_combined_path = Path(args.table_combined_output) if args.table_combined_output else (out_dir / f"{base_stem}_table_combined_contexts.json")
    table_combined_path.parent.mkdir(parents=True, exist_ok=True)
    table_combined_path.write_text(json.dumps({"contexts": combined_contexts}, ensure_ascii=False, indent=2), encoding="utf-8")

    table_header_path = Path(args.table_header_output) if args.table_header_output else (out_dir / f"{base_stem}_table_header_contexts.json")
    table_header_path.parent.mkdir(parents=True, exist_ok=True)
    table_header_path.write_text(json.dumps({"contexts": table_header_contexts}, ensure_ascii=False, indent=2), encoding="utf-8")

    table_others_path = Path(args.table_others_output) if args.table_others_output else (out_dir / f"{base_stem}_table_others_contexts.json")
    table_others_path.parent.mkdir(parents=True, exist_ok=True)
    table_others_path.write_text(json.dumps({"contexts": table_others_contexts}, ensure_ascii=False, indent=2), encoding="utf-8")

    if (
        not args.paragraph_output
        and not args.table_no_border_output
        and not args.table_image_output
        and not args.table_combined_output
        and not args.table_header_output
        and not args.table_others_output
        and not args.output_dir
    ):
        print(f"[info] paragraph contexts → {para_path}")
        print(f"[info] table no-border contexts → {table_path}")
        print(f"[info] table image contexts → {table_image_path}")
        print(f"[info] table combined contexts → {table_combined_path}")
        print(f"[info] table header contexts → {table_header_path}")
        print(f"[info] table others contexts → {table_others_path}")


if __name__ == "__main__":
    main()
