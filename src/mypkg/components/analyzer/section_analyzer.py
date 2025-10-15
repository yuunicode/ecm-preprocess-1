"""
숫자형 제목 기반 섹션 트리 구축기

이 모듈은 문단 리스트(각 문단은 최소한 `doc_index`, `text` 보유)에서
숫자 패턴의 제목(예: "1", "1.1", "1.1.1" …)을 감지하여 섹션 트리를 만듭니다.

핵심 아이디어
- 제목 탐지: 정규식으로 "숫자( . 숫자)* + 공백 + 제목" 형태를 인식
- 스택 기반 트리화: 등장 순서대로 순회하며 현재 열린 섹션 스택을 레벨에 맞춰 닫거나 추가
- 구간(span): 각 섹션은 [start, end) 반열린 구간으로 문서 범위를 갖습니다
  · start = 제목이 놓인 문단의 doc_index
  · end   = 다음에 등장하는 같은/상위 레벨 제목의 doc_index 또는 문서 끝+1

제목 수준(레벨) 지원
- `HEADING_RE`가 "첫 번호 + ( . 번호 ) × 최대 7회"를 허용하므로, 최대 8단계(1~8 레벨)까지 지원합니다.
  예) 1, 1.1, 1.1.1, …, 1.1.1.1.1.1.1.1
- 더 깊게 필요하면 정규식의 `{0,7}`을 늘리면 됩니다.

접두 문구(한글/영문) 지원
- 한글: "제" 접두 또는 "장/절/항" 접미(선택)를 허용합니다. 예) "제 1 장 개요", "1 절 정의"
- 영문: "Section/Sec./Chapter/Chap./Part" 접두 허용. 예) "Section 1.2 Title", "Chapter 3) Text"
"""

# mypkg/components/analyzer/section_analyzer.py
from __future__ import annotations
import re
from typing import List, Iterable, Tuple, Optional, Set, Dict
from mypkg.core.docjson_types import Section

# 제목 감지 정규식
# - ^\s*                : 시작/선행 공백 허용
# - (\d+(?:\s*\.\s*\d+){0,7}) : "숫자" + ("." + "숫자")가 최대 7번 반복 → 총 8단계까지 허용
#   · "1", "1.1", "1. 1", "1.  2.  3" 모두 허용 (점 주변 공백 허용)
# - [.)]?\s+            : 끝에 "." 또는 ")"가 있을 수도 있고(옵션), 그 뒤에 공백이 최소 1칸
# - (.*\S)\s*$         : 실제 제목 본문(양끝 공백 제거, 내용이 1자 이상)
# 예) "1. 개요", "1.1 정의", "2.1.3.4) 세부 항목"
HEADING_RE = re.compile(r"^\s*(\d+(?:\s*\.\s*\d+){0,7})[.)]?\s+(.*\S)\s*$")

# 한글 접두/접미 지원: (제)? <번호>(.번호)* (장|절|항)? 제목
HEADING_RE_KO = re.compile(
    r"^\s*(?:제\s*)?(\d+(?:\s*\.\s*\d+){0,7})\s*(?:장|절|항)?[.)]?\s+(.*\S)\s*$"
)

# 영문 접두 지원: Section|Sec.|Chapter|Chap.|Part + <번호>(.번호)*
HEADING_RE_EN = re.compile(
    r"^\s*(?:section|sec\.|chapter|chap\.|part)\s*(\d+(?:\s*\.\s*\d+){0,7})[.)]?\s+(.*\S)\s*$",
    re.IGNORECASE,
)

def _collapse_spaces(s: str) -> str:
    return re.sub(r"\s{2,}", " ", s).strip()


def _style_levels_from(heading_styles: Optional[Iterable[str]]) -> Dict[str, int]:
    """주어진 스타일 이름 목록을 레벨 맵으로 변환한다."""
    level_map: Dict[str, int] = {}
    if not heading_styles:
        return level_map
    for idx, style in enumerate(heading_styles, start=1):
        if not isinstance(style, str):
            continue
        key = style.strip().lower()
        if not key:
            continue
        m = re.search(r"(\d+)$", key)
        level = int(m.group(1)) if m else idx
        level_map[key] = level
    return level_map


def _next_style_number(level: int, counters: List[int]) -> str:
    """스타일 기반 섹션 번호를 단계별로 증가시키고 문자열을 반환한다."""
    if level <= 0:
        level = 1
    if len(counters) < level:
        counters.extend([0] * (level - len(counters)))
    counters[level - 1] += 1
    for idx in range(level, len(counters)):
        counters[idx] = 0
    return ".".join(str(counters[i]) for i in range(level) if counters[i] > 0)


def _sync_style_counters(number: str, level: int, counters: List[int]) -> None:
    """숫자 패턴으로 추출된 번호를 스타일 카운터에 반영한다."""
    if level <= 0:
        return
    try:
        parts = [int(part) for part in number.split(".") if part]
    except ValueError:
        return
    target_len = max(level, len(parts))
    if len(counters) < target_len:
        counters.extend([0] * (target_len - len(counters)))
    for idx, val in enumerate(parts):
        counters[idx] = val
    for idx in range(len(parts), len(counters)):
        counters[idx] = 0

def _detect_heading(
    text: str,
    style: Optional[str] = None,
    allowed_styles: Optional[Set[str]] = None,
) -> Tuple[str, str, int] | None:
    """
    단일 문단 텍스트가 숫자형 제목인지 감지하여
    (정규화된 번호, 정규화된 제목, 레벨) 을 반환합니다.

    동작 요약
    - HEADING_RE로 매칭 후, 번호 내 점 주변 공백을 제거/정규화
      예: "1. 1  제목" → 번호 "1.1", 제목 "제목"
    - 레벨은 번호를 '.'로 split한 길이
      예: "1" → 1, "1.1" → 2, "1.1.1" → 3
    - 매칭 실패 시 None
    `allowed_styles`가 주어지면, style이 거기에 포함된 문단만 제목으로 간주합니다.
    """
    if allowed_styles is not None:
        normalized_style = (style or "").strip().lower()
        if normalized_style not in allowed_styles:
            return None
    # 1) 순수 숫자형 패턴: 1, 1.1, 1.1.1 ...
    m = HEADING_RE.match(text or "")
    # 2) 한글 접두/접미 패턴: 제1장/1절 등
    if not m:
        m = HEADING_RE_KO.match(text or "")
    # 3) 영문 접두 패턴: Section 1.2 / Chapter 3 ...
    if not m:
        m = HEADING_RE_EN.match(text or "")
    if not m:
        return None
    raw, title = m.group(1), m.group(2)
    number = re.sub(r"\s*\.\s*", ".", raw.strip())  # "1. 1" -> "1.1"
    return number, _collapse_spaces(title), len(number.split("."))

def build_sections(
    paragraphs: List[dict],
    heading_styles: Optional[Iterable[str]] = None,
    enforce_style_whitelist: bool = False,
) -> List[Section]:
    """
    숫자 제목을 바탕으로 섹션 트리를 만들고, 각 섹션의 문서 구간(span)을 채웁니다.

    입력
    - paragraphs: dict 리스트. 최소 `doc_index`, `text` 키 필요

    알고리즘(선형, O(N))
    1) 문단을 `doc_index` 오름차순으로 정렬합니다.
    2) 스택(stack)에 현재 열린 섹션들을 유지합니다(맨 위가 가장 최근/가장 깊은 섹션).
    3) 각 문단 p에 대해:
       - 제목이 아니면 skip
       - 제목이면 (번호, 제목, 레벨)을 구하고,
         · 현재 스택의 top.level >= 새 레벨인 섹션들을 "현재 p.doc_index"를 end로 닫습니다.
         · 새 섹션을 생성하여 span=[p.doc_index, 0]으로 push (end는 미정)
         · 루트면 roots에, 자식이면 부모의 subsections에 추가
    4) 순회 종료 후, 스택에 남은 섹션들은 문서 마지막 문단의 다음 인덱스를 end로 닫습니다.

    결과
    - 최상위 섹션 리스트(roots)를 반환합니다.
    - 각 섹션은 `span=[start, end)`로 반열린 구간을 가지며, `blocks`/`block_ids`는 비어 있습니다
      (실제 블록 할당은 layout_assembler.assign_blocks_to_sections에서 수행).
    """
    paras = sorted(paragraphs, key=lambda p: p.get("doc_index", 0))
    style_whitelist: Optional[Set[str]] = None
    style_levels = _style_levels_from(heading_styles)
    if heading_styles is not None and enforce_style_whitelist:
        style_whitelist = set(style_levels.keys())
        if not style_whitelist:
            style_whitelist = set()
    roots: List[Section] = []
    stack: List[Tuple[Section, int]] = []
    style_counters: List[int] = []

    def close_until(level: int, next_idx: int):
        # 새 제목이 등장했을 때, 같은 레벨 이상(>= level)의 열린 섹션을
        # 모두 next_idx로 닫는다. 이렇게 하면 형제/상위 섹션의 end가
        # "다음 동레벨↑ 헤딩의 doc_index"가 되어 span 규칙을 만족한다.
        while stack and stack[-1][1] >= level:
            section_to_close, _ = stack[-1]
            if section_to_close.span[1] == 0:
                section_to_close.span[1] = next_idx
            stack.pop()

    for p in paras:
        di = p.get("doc_index")
        style_key = (p.get("style") or "").strip().lower()
        det = _detect_heading(
            p.get("text", ""),
            style=p.get("style"),
            allowed_styles=style_whitelist if style_whitelist else None,
        )
        if not det and style_levels:
            # 숫자 패턴이 없지만 스타일 레벨로 섹션을 나눌 수 있는 경우에 대한 보조 로직
            level_from_style = style_levels.get(style_key)
            heading_text = _collapse_spaces(p.get("text") or "")
            if level_from_style and heading_text:
                number = _next_style_number(level_from_style, style_counters)
                det = (number, heading_text, level_from_style)
        if not det:
            continue
        number, title, level = det
        if style_levels:
            _sync_style_counters(number, level, style_counters)
        # 현재 제목의 레벨에 맞춰 기존 열린 섹션을 정리한 뒤 새 섹션을 연다
        close_until(level, di)
        sec = Section(
            id=f"sec_{number}",
            number=number,
            title=title,
            doc_index=di,
            span=[di, 0],
            # path는 루트부터의 누적 경로를 저장할 수 있도록 기본은 자기 자신만 설정
            # (필요 시 상위 정보를 조합해 확장 가능)
            path=[_collapse_spaces(f"{number} {title}")],
            block_ids=[],
            blocks=[],
            subsections=[],
            heading_block_id=f"h{di}",
        )
        if not stack:
            roots.append(sec)
        else:
            parent_section, _ = stack[-1]
            parent_section.subsections.append(sec)
        stack.append((sec, level))

    # 문서 끝(+1)을 기준으로 스택에 남은 섹션을 모두 닫는다
    last = paras[-1]["doc_index"] if paras else 0
    while stack:
        section_to_close, _ = stack[-1]
        if section_to_close.span[1] == 0:
            section_to_close.span[1] = last + 1
        stack.pop()
    return roots

def iter_sections(sections: List[Section]) -> Iterable[Section]:
    for s in sections:
        yield s
        for c in s.subsections:
            yield from iter_sections([c])
