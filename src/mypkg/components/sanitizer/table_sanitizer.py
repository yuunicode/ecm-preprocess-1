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
        out = []
        all_paragraphs = {p.doc_index: p for p in paragraphs if p.doc_index is not None}

        for t in tables or []:
            # --- 1. 초기 설정 및 테이블 이전 문단 텍스트 추출 ---
            preceding_text = None
            if t.doc_index is not None and t.doc_index > 0:
                preceding_para = all_paragraphs.get(t.doc_index - 1)
                if preceding_para:
                    preceding_text = preceding_para.text

            rows = t.rows
            col_counts = [sum(max(1, c.gridSpan or 1) for c in row) for row in rows] or [0]
            C = max(col_counts) if col_counts else 0
            R = len(rows)

            matrix = [[self.blank for _ in range(C)] for _ in range(R)]
            image_matrix = [[[] for _ in range(C)] for _ in range(R)]
            color_matrix = [[None for _ in range(C)] for _ in range(R)]
            style_color_matrix = [[None for _ in range(C)] for _ in range(R)]
            explicit_color_matrix = [[None for _ in range(C)] for _ in range(R)]
            has_explicit_matrix = [[False for _ in range(C)] for _ in range(R)]
            vflags = [[None for _ in range(C)] for _ in range(R)]  # 행 병합 상태 추적용 플래그

            anchors = []

            # --- 2. 원본 테이블 순회하며 기본 정보 채우기 ---
            for r_idx, row in enumerate(rows):
                c_idx = 0
                # 행 병합으로 건너뛸 셀 인덱스 이동
                while c_idx < C and vflags[r_idx][c_idx] == "continue":
                    c_idx += 1

                for cell in row:  # 이 'cell' 변수는 TableCellRecord 타입의 객체입니다.
                    colspan = max(1, cell.gridSpan or 1)
                    vmerge = cell.vMerge
                    txt = (cell.text or "").strip()

                    while c_idx < C and vflags[r_idx][c_idx] == "continue":
                        c_idx += 1

                    if c_idx >= C:
                        break

                    # 매트릭스에 텍스트, 이미지, 색상 정보 채우기
                    matrix[r_idx][c_idx] = txt
                    image_matrix[r_idx][c_idx] = list(getattr(cell, "inline_images", []) or [])
                    cell_explicit = getattr(cell, "explicit_bg_color", None)
                    cell_style = getattr(cell, "style_bg_color", None)
                    cell_has_explicit = bool(getattr(cell, "has_explicit_bg", False))
                    final_color = cell_explicit if cell_has_explicit else cell_style
                    color_matrix[r_idx][c_idx] = final_color
                    style_color_matrix[r_idx][c_idx] = cell_style
                    explicit_color_matrix[r_idx][c_idx] = cell_explicit
                    has_explicit_matrix[r_idx][c_idx] = cell_has_explicit

                    # 열 병합(colspan) 및 행 병합(vmerge) 플래그 처리
                    for off in range(colspan):
                        cc = c_idx + off
                        if cc >= C:
                            break
                        if off > 0: # 병합된 추가 셀은 빈 값으로 채움
                            matrix[r_idx][cc] = self.blank
                            image_matrix[r_idx][cc] = list(getattr(cell, "inline_images", []) or [])
                            color_matrix[r_idx][cc] = final_color
                            style_color_matrix[r_idx][cc] = cell_style
                            explicit_color_matrix[r_idx][cc] = cell_explicit
                            has_explicit_matrix[r_idx][cc] = cell_has_explicit
                        if vmerge:
                            vflags[r_idx][cc] = vmerge

                    anchors.append({
                        "r": r_idx,
                        "c": c_idx,
                        "text": txt,
                        "colspan": max(1, colspan),
                        "vmerge": vmerge,
                        "rowspan": 1,
                    })
                    c_idx += colspan

            # --- 3. 병합된 셀(앵커 기준)의 내용 채우기 ---
            for a in anchors:
                r0, c0 = a["r"], a["c"]
                text, colspan, vmerge = a["text"], max(1, a["colspan"] or 1), a["vmerge"]

                # 행 병합(rowspan) 계산
                rowspan = 1
                if vmerge == "restart":
                    rr = r0 + 1
                    while rr < R:
                        all_cont = all(c0 + off < C and vflags[rr][c0 + off] == "continue" for off in range(colspan))
                        if not all_cont:
                            break
                        rowspan += 1
                        rr += 1
                a["rowspan"] = max(1, rowspan)
                
                # 계산된 rowspan, colspan에 따라 모든 병합된 셀에 데이터 복사
                base_images = list(image_matrix[r0][c0])
                base_color = color_matrix[r0][c0]
                base_style_color = style_color_matrix[r0][c0]
                base_explicit_color = explicit_color_matrix[r0][c0]
                base_has_explicit = has_explicit_matrix[r0][c0]
                for rr in range(r0, min(R, r0 + rowspan)):
                    for off in range(colspan):
                        cc = c0 + off
                        if cc < C:
                            matrix[rr][cc] = text
                            image_matrix[rr][cc] = list(base_images)
                            color_matrix[rr][cc] = base_color
                            style_color_matrix[rr][cc] = base_style_color
                            explicit_color_matrix[rr][cc] = base_explicit_color
                            has_explicit_matrix[rr][cc] = base_has_explicit

            # --- 4. 최종 데이터 구조 생성 ---
            cell_data_matrix = []
            for r in range(R):
                row_data = []
                for c in range(C):
                    row_data.append({
                        "text": str(matrix[r][c]) if matrix[r][c] is not None else self.blank,
                        "images": image_matrix[r][c],
                        "color": color_matrix[r][c],
                        "style_color": style_color_matrix[r][c],
                        "explicit_color": explicit_color_matrix[r][c],
                        "has_explicit_color": has_explicit_matrix[r][c],
                    })
                cell_data_matrix.append(row_data)

            # --- 5. 헤더 분석 ---
            is_rowheader = False
            if R > 0 and C > 0:
                first_row_colors = color_matrix[0]
                if first_row_colors[0] is not None and all(c == first_row_colors[0] for c in first_row_colors):
                    is_rowheader = True

            is_colheader = False
            if R > 0 and C > 0:
                first_col_colors = [color_matrix[r][0] for r in range(R)]
                if first_col_colors[0] is not None and all(c == first_col_colors[0] for c in first_col_colors):
                    is_colheader = True

            table_html = render_table_html(cell_data_matrix, anchors, is_rowheader, is_colheader)

            t_out = {
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
            }
            out.append(t_out)

        return out

    def build_components(self, sanitized_tables: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """정제된 테이블 목록을 JSON 출력용 딕셔너리로 감쌉니다."""
        return {"tables": sanitized_tables}
