"""
테이블 컴포넌트 정제기 (Table Sanitizer)

이 모듈은 파서(parser)로부터 추출된 테이블(`TableRecord`) 목록을 받아,
테이블 구조를 분석하고 VDB에 적재하기 좋은 형태로 정제하는 역할을 합니다.

주요 기능:
- **셀 병합 처리**: `gridSpan`(열 병합)과 `vMerge`(행 병합) 속성을 해석하여,
  병합된 모든 셀에 원본 셀의 텍스트, 이미지, 색상 정보를 채워 넣습니다.
- **2D 데이터 행렬 생성**: 정제된 셀 데이터를 `TableCellData` 형식의 2D 리스트(`data`)로 만듭니다.
- **컨텍스트 정보 추가**: 테이블 바로 앞에 위치한 문단 텍스트를 `preceding_text`로 추출하여 포함합니다.
- **헤더 분석**: 테이블의 첫 행과 첫 열의 배경색이 모두 동일한지 여부를 분석하여,
  `is_rowheader`와 `is_colheader` 플래그를 설정합니다.
"""
from typing import List, Dict, Any
from mypkg.core.parser import TableRecord, TableCellRecord, ParagraphRecord
from mypkg.components.parser.xml_parser import render_table_html

class TableSanitizer:
    """테이블 데이터를 정제하고 분석하여 VDB에 적합한 컴포넌트를 생성합니다."""

    def __init__(self, blank_placeholder: str = ""):
        self.blank = blank_placeholder

    def apply(self, tables: List[TableRecord], paragraphs: List[ParagraphRecord]) -> List[Dict[str, Any]]:
        """테이블 레코드 목록을 받아 정제 로직을 적용하고, 처리된 딕셔너리 목록을 반환합니다."""
        out: List[Dict[str, Any]] = []
        all_paragraphs = {p.doc_index: p for p in paragraphs if p.doc_index is not None}

        def combine_text_and_images(text: str, images: List[str]) -> str:
            text = (text or "").strip()
            image_tokens = " ".join(f"[image:{rid}]" for rid in images if rid)
            if text and all(f"[image:{rid}]" in text for rid in images if rid):
                return text
            if text and image_tokens:
                return f"{text} {image_tokens}".strip()
            if image_tokens:
                return image_tokens
            return text

        for t in tables or []:
            preceding_text = None
            if t.doc_index is not None and t.doc_index > 0:
                preceding_para = all_paragraphs.get(t.doc_index - 1)
                if preceding_para:
                    preceding_text = preceding_para.text

            rows = t.rows or []
            col_counts = [sum(max(1, c.gridSpan or 1) for c in row) for row in rows] or [0]
            C = max(col_counts) if col_counts else 0
            R = len(rows)
            if R == 0 or C == 0:
                continue

            matrix = [[self.blank for _ in range(C)] for _ in range(R)]
            image_matrix = [[[] for _ in range(C)] for _ in range(R)]
            style_matrix = [[False for _ in range(C)] for _ in range(R)]
            vflags = [[None for _ in range(C)] for _ in range(R)]
            anchors: List[Dict[str, Any]] = []

            for r_idx, row in enumerate(rows):
                c_idx = 0
                while c_idx < C and vflags[r_idx][c_idx] == "continue":
                    c_idx += 1

                for cell in row or []:
                    colspan = max(1, cell.gridSpan or 1)
                    vmerge = cell.vMerge
                    txt = (cell.text or "").strip()

                    while c_idx < C and vflags[r_idx][c_idx] == "continue":
                        c_idx += 1

                    if c_idx >= C:
                        break

                    inline_imgs = list(getattr(cell, "inline_images", []) or [])
                    combined_text = combine_text_and_images(txt, inline_imgs)
                    matrix[r_idx][c_idx] = combined_text
                    image_matrix[r_idx][c_idx] = list(inline_imgs)
                    style_matrix[r_idx][c_idx] = bool(getattr(cell, "has_style", False))

                    for off in range(colspan):
                        cc = c_idx + off
                        if cc >= C:
                            break
                        if off > 0:
                            matrix[r_idx][cc] = combined_text if combined_text else self.blank
                            image_matrix[r_idx][cc] = list(inline_imgs)
                            style_matrix[r_idx][cc] = bool(getattr(cell, "has_style", False))
                        if vmerge:
                            vflags[r_idx][cc] = vmerge

                    anchors.append({
                        "r": r_idx,
                        "c": c_idx,
                        "text": combined_text,
                        "colspan": max(1, colspan),
                        "vmerge": vmerge,
                        "rowspan": 1,
                    })
                    c_idx += colspan

            for anchor in anchors:
                r0, c0 = anchor["r"], anchor["c"]
                text = anchor["text"]
                colspan = max(1, anchor.get("colspan") or 1)
                vmerge = anchor.get("vmerge")

                rowspan = 1
                if vmerge == "restart":
                    rr = r0 + 1
                    while rr < R:
                        all_cont = all(c0 + off < C and vflags[rr][c0 + off] == "continue" for off in range(colspan))
                        if not all_cont:
                            break
                        rowspan += 1
                        rr += 1
                anchor["rowspan"] = max(1, rowspan)

                base_images = list(image_matrix[r0][c0])
                base_style = style_matrix[r0][c0]
                for rr in range(r0, min(R, r0 + rowspan)):
                    for off in range(colspan):
                        cc = c0 + off
                        if cc < C:
                            matrix[rr][cc] = text
                            image_matrix[rr][cc] = list(base_images)
                            style_matrix[rr][cc] = base_style

            cell_data_matrix: List[List[Dict[str, Any]]] = []
            for r in range(R):
                row_payload: List[Dict[str, Any]] = []
                for c in range(C):
                    row_payload.append({
                        "text": str(matrix[r][c]) if matrix[r][c] is not None else self.blank,
                        "images": image_matrix[r][c],
                        "styled": style_matrix[r][c],
                    })
                cell_data_matrix.append(row_payload)

            is_rowheader = False
            if R > 0 and C > 0:
                first_row_texts = [matrix[0][c] for c in range(C)]
            row_style_flags = [style_matrix[0][c] for c in range(C)]
            style_uniform = all(row_style_flags) and any(row_style_flags)
            long_text = any(len(text.split()) >= 3 for text in first_row_texts if text)
            is_rowheader = style_uniform and not long_text

            is_colheader = False
            if R > 0 and C > 0:
                first_col_texts = [matrix[r][0] for r in range(R)]
                col_style_flags = [style_matrix[r][0] for r in range(R)]
                style_uniform = all(col_style_flags) and any(col_style_flags)
                long_text = any(len(text.split()) >= 3 for text in first_col_texts if text)
                is_colheader = style_uniform and not long_text

            table_html = render_table_html(cell_data_matrix, anchors, is_rowheader, is_colheader)

            out.append({
                "tid": t.tid,
                "doc_index": t.doc_index,
                "preceding_text": preceding_text,
                "rows": R,
                "cols": C,
                "data": cell_data_matrix,
                "is_rowheader": is_rowheader,
                "is_colheader": is_colheader,
                "has_borders": t.has_borders,
                "table_html": table_html,
                "anchors": anchors,
            })

        return out

    def build_components(self, sanitized_tables: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """정제된 테이블 목록을 JSON 출력용 딕셔너리로 감쌉니다."""
        return {"tables": sanitized_tables}
