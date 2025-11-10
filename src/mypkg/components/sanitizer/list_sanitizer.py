"""
리스트 컴포넌트 정제기 (List Sanitizer)

이 모듈은 "List Bullet" 또는 "List Number" 스타일을 가진 연속된 문단들을
하나의 리스트 컴포넌트로 그룹화하는 역할을 합니다.

주요 기능:
- **리스트 그룹화**: 연속된 리스트 스타일의 문단을 찾아 하나의 그룹으로 묶습니다.
- **제목 탐색**: 리스트 그룹 바로 앞에 위치한 '소제목2' 스타일의 문단을 리스트의 제목으로 자동 인식합니다.
- **텍스트 포맷팅**: 그룹화된 리스트 아이템들을 보기 쉬운 형식(예: "a. 내용")으로 포맷팅합니다.
- **메타데이터 병합**: 그룹에 포함된 모든 문단의 속성(강조, 수식, 이미지 포함 여부 등)을 리스트 컴포넌트에 통합합니다.
"""
from __future__ import annotations
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

from mypkg.core.parser import ParagraphRecord

# 포맷터 함수의 타입 별칭
FormatFunc = Callable[[str, List[str]], str]

def _alpha_label(index: int) -> str:
    """0부터 시작하는 인덱스를 받아 'a', 'b', ..., 'z', 'aa', 'ab', ... 순서의 알파벳 레이블을 생성합니다."""
    base = ord("a")
    label = []
    idx = index
    while True:
        idx, rem = divmod(idx, 26)
        label.append(chr(base + rem))
        if idx == 0:
            break
        idx -= 1  # 26진법과 유사하지만, 0-25가 a-z에 매핑되므로 다음 자리수로 넘어갈 때 1을 빼줌
    return "".join(reversed(label))


def default_list_formatter(title: str, items: List[str]) -> str:
    """리스트의 제목과 항목들을 받아 기본 형식의 문자열로 포맷팅합니다."""
    lines: List[str] = []
    title_clean = title.strip()
    if title_clean:
        lines.append(f"제목: {title_clean}")

    for idx, item in enumerate(items):
        stripped = item.strip()
        if not stripped:
            continue
        label = _alpha_label(idx)
        lines.append(f"{label}. {stripped}")
    return "\n".join(lines)


class ListSanitizer:
    """
    대표적인 리스트 스타일(예: "List Bullet", "List Number", "List Paragraph", "List Continue")의 문단을
    그룹화하여 단일 리스트 컴포넌트로 변환하는 정제기입니다.
    """

    _LIST_STYLE_PREFIXES = tuple(
        s.lower() for s in ("List Bullet", "List Number", "List Paragraph", "List Continue")
    )
    _TITLE_STYLE_CANDIDATE = "소제목2"
    _PRECEDING_LOOKBACK = 3

    def __init__(self, formatter: FormatFunc | None = None) -> None:
        """
        ListSanitizer를 초기화합니다.

        Args:
            formatter: 리스트의 제목과 항목들을 문자열로 변환할 함수.
                       제공되지 않으면 default_list_formatter가 사용됩니다.
        """
        self.formatter: FormatFunc = formatter or default_list_formatter

    def sanitize(
        self,
        paragraphs: Sequence[ParagraphRecord],
    ) -> Tuple[List[Dict[str, object]], List[int]]:
        """
        문단 목록에서 리스트 그룹을 찾아 정제된 컴포넌트와 처리된 문단 인덱스 목록을 반환합니다.

        Args:
            paragraphs: 전체 문단 레코드 시퀀스.

        Returns:
            - 정제된 리스트 컴포넌트 딕셔너리 목록.
            - 리스트 컴포넌트로 처리되어 소비된 문단의 doc_index 목록.
        """
        groups = self._collect_bullet_groups(paragraphs)
        by_doc_index = {p.doc_index: p for p in paragraphs if p.doc_index is not None}

        components: List[Dict[str, object]] = []
        consumed_indices: List[int] = []

        for group in groups:
            if len(group) <= 1:
                continue

            first_para = group[0]
            first_doc_index = first_para.doc_index
            if first_doc_index is None:
                continue

            # 리스트의 제목 및 주변 컨텍스트 추출
            title_text, preceding_texts = self._collect_preceding_context(first_doc_index, by_doc_index)

            items = [p.text.strip() for p in group if p.text]
            formatted_text = self.formatter(title_text, items)

            # 그룹 내 모든 문단의 메타데이터를 병합
            source_indices, merged_attrs = self._merge_group_attributes(group)

            component_style = self._normalize_style(first_para.style) or "List"

            components.append(
                {
                    "text": formatted_text,
                    "doc_index": first_doc_index,
                    "style": component_style,
                    "preceding_texts": preceding_texts,
                    "source_doc_indices": sorted(source_indices),
                    **merged_attrs,
                }
            )
            
            # 이 그룹에 속한 문단들의 인덱스를 소비된 것으로 기록
            for para in group:
                if para.doc_index is not None:
                    consumed_indices.append(para.doc_index)

        return components, sorted(set(consumed_indices))

    def build_components(
        self,
        sanitized_lists: List[Dict[str, object]],
        consumed_indices: List[int],
    ) -> Dict[str, List]:
        """정제된 리스트 데이터와 소비된 인덱스를 최종 JSON 구조로 감쌉니다."""
        return {"lists": sanitized_lists, "consumed": consumed_indices}

    def _normalize_style(self, style: str | None) -> str:
        """문단 스타일 문자열을 정규화합니다."""
        return (style or "").strip()

    def _is_list_style(self, style: str | None) -> bool:
        """주어진 스타일이 리스트 스타일인지 확인합니다."""
        normalized = self._normalize_style(style)
        if not normalized:
            return False
        lowered = normalized.lower()
        return any(lowered.startswith(prefix) for prefix in self._LIST_STYLE_PREFIXES)

    def _collect_bullet_groups(
        self,
        paragraphs: Iterable[ParagraphRecord],
    ) -> List[List[ParagraphRecord]]:
        """문단 목록을 순회하며 연속된 리스트 스타일의 문단 그룹을 찾습니다."""
        # 문서를 doc_index 순서로 정렬하여 순차적으로 처리
        ordered = sorted(paragraphs, key=lambda p: (p.doc_index is None, p.doc_index or 0))
        
        groups: List[List[ParagraphRecord]] = []
        i = 0
        while i < len(ordered):
            current = ordered[i]
            if not self._is_list_style(current.style):
                i += 1
                continue

            # 리스트 그룹 시작: 연속된 리스트 문단을 수집
            group: List[ParagraphRecord] = []
            last_doc_index = current.doc_index

            while i < len(ordered):
                candidate = ordered[i]
                # 리스트 스타일이 아니면 그룹 종료
                if not self._is_list_style(candidate.style):
                    break
                
                # doc_index가 연속적이지 않으면 그룹 종료 (예: 중간에 다른 문단이 끼어든 경우)
                doc_index = candidate.doc_index
                if last_doc_index is not None and doc_index is not None:
                    if doc_index > last_doc_index + 1:
                        break
                
                group.append(candidate)
                if doc_index is not None:
                    last_doc_index = doc_index
                i += 1

            if group:
                groups.append(group)
            else:
                # 그룹이 만들어지지 않은 경우, 다음 문단으로 이동
                i += 1
        return groups

    def _collect_preceding_context(
        self,
        first_doc_index: int,
        by_doc_index: Dict[int, ParagraphRecord],
    ) -> Tuple[str, List[str]]:
        """
        리스트 그룹의 시작 인덱스를 기준으로 제목 후보와 주변 컨텍스트(직전 문단들)를 수집합니다.
        """
        collected: List[Tuple[int, str]] = []
        title_text = ""

        for offset in range(1, self._PRECEDING_LOOKBACK + 1):
            prev_para = by_doc_index.get(first_doc_index - offset)
            if not prev_para:
                continue

            text = (prev_para.text or "").strip()
            if (
                not text
                or prev_para.image_included
                or text.startswith("[image:")
                or self._is_list_style(prev_para.style)
            ):
                continue

            doc_index = prev_para.doc_index
            if doc_index is None:
                continue

            collected.append((doc_index, text))

            normalized_style = self._normalize_style(prev_para.style)
            if not title_text and normalized_style == self._TITLE_STYLE_CANDIDATE:
                title_text = text

        collected.sort(key=lambda item: item[0])
        preceding_texts = [text for _, text in collected]

        return title_text, preceding_texts

    def _merge_group_attributes(
        self,
        group: List[ParagraphRecord],
    ) -> Tuple[set, Dict[str, object]]:
        """리스트 그룹 내 모든 문단의 속성(source_indices, emphasized 등)을 병합합니다."""
        source_index_set = set()
        emphasized: List[str] = []
        math_texts: List[str] = []
        image_included = False

        for para in group:
            if para.doc_index is not None:
                source_index_set.add(para.doc_index)
            # 문단이 병합된 경우, 원본 source_doc_indices도 모두 포함
            for idx in getattr(para, "source_doc_indices", []) or []:
                if isinstance(idx, int):
                    source_index_set.add(idx)
            
            emphasized.extend(para.emphasized or [])
            math_texts.extend(para.math_texts or [])
            if para.image_included:
                image_included = True

        return source_index_set, {
            "emphasized": emphasized,
            "math_texts": math_texts,
            "image_included": image_included,
        }
