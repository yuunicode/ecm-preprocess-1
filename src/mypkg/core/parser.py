from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Union, BinaryIO, List
from dataclasses import dataclass, field

# --- Result Report Dataclasses ---
@dataclass
class ParseResult:
    success: bool
    content: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    processing_time: float = 0.0
    payload_tag: str = ""

# --- Record Dataclasses ---
@dataclass
class RunRecord:
    """Run은 글자 서식의 단위입니다."""
    text: str  # Run의 텍스트 내용
    b: bool  # Bold 여부
    i: bool  # Italic 여부
    u: bool  # Underline 여부
    rStyle: Optional[str] = None  # Run 스타일 이름
    sz: Optional[float] = None  # 글자 크기 (pt)
    color: Optional[str] = None  # 글자 색상 (RGB)
    image_rids: List[str] = field(default_factory=list)  # Run에 포함된 inline image rId 목록

@dataclass
class ParagraphRecord:
    """문단(Paragraph) 정보를 담는 데이터 클래스 (docx와 xml 파서에서 공용으로 사용)"""
    text: str  # 문단 전체 텍스트
    
    # docx_parser에서 주로 사용
    doc_index: Optional[int] = None  # 문서 내 문단의 순서
    style: Optional[str] = None  # 문단 스타일 이름
    runs: List[RunRecord] = field(default_factory=list)  # 문단을 구성하는 Run 목록
    source_doc_indices: List[int] = field(default_factory=list)  # 병합 시 원본 문단 doc_index 추적

    # xml_parser에서 주로 사용
    emphasized: List[str] = field(default_factory=list)  # bold 된 텍스트 조각
    math_texts: List[str] = field(default_factory=list)  # 추출된 수식 문자열들
    image_included: bool = False  # 문단에 인라인 이미지가 포함되어 있는지 여부

@dataclass
class TableCellRecord:
    """표의 셀(Cell) 정보를 담는 데이터 클래스"""
    text: str  # 셀 텍스트
    gridSpan: int = 1  # 열 병합(gridSpan) 수
    vMerge: Optional[str] = None  # 행 병합(vMerge) 상태 ('restart' 또는 None)
    inline_images: List[str] = field(default_factory=list)  # 셀에 포함된 이미지 rId 목록
    has_style: bool = False  # 셀에 스타일(배경 등)이 적용되었는지 여부

@dataclass
class TableRecord:
    """표(Table) 정보를 담는 데이터 클래스"""
    tid: str  # 표 고유 ID
    rows: List[List[TableCellRecord]] = field(default_factory=list)  # 표의 행(row) 목록
    doc_index: Optional[int] = None  # 문서 내 순서
    has_borders: bool = True  # 표의 테두리 유무

@dataclass
class HeaderFooterRecord:
    """헤더/푸터 정보를 담는 데이터 클래스"""
    part: str  # 헤더/푸터 XML 파트 이름
    text: str  # 헤더/푸터 전체 텍스트

@dataclass
class RelationshipRecord:
    """문서 관계(Relationship) 정보를 담는 데이터 클래스"""
    rid: str  # 관계 ID
    type: str  # 관계 타입 URL
    target: str  # 대상 경로

@dataclass
class InlineImageRecord:
    """인라인 이미지 정보를 담는 데이터 클래스"""
    rId: str  # 관계 ID (Relationship ID)
    filename: Optional[str] = None  # 이미지 파일명
    doc_index: Optional[int] = None  # 이미지가 포함된 대표 단락 인덱스
    doc_indices: List[int] = field(default_factory=list)  # 이미지가 등장한 모든 단락 인덱스
    saved_path: Optional[str] = None  # data_store에 저장된 상대 경로

@dataclass
class DrawingRecord:
    """
    다이어그램 후처리를 위한 '필수 스키마' 버전의 드로잉 레코드.
    좌표가 불완전해도 군집화/정렬에 바로 사용할 수 있도록 설계됨.
    모든 좌표/크기 단위는 EMU 기준(불명확/없음은 0).
    """

    # --- ID & 분류 ---
    did: str                                           # 고유 ID (필수)
    kind: str                                          # "shape|connector|image|inline|group"
    preset: Optional[str] = None                       # 예: "rect", "rightArrow", ...

    # --- 위치/크기(원시 Anchor 기준) ---
    anchor_type: Optional[str] = None                  # "anchor|inline|vml"
    rel_from_h: Optional[str] = None                   # "page|margin|column|paragraph|character|null"
    rel_from_v: Optional[str] = None
    pos_offset: Dict[str, int] = field(default_factory=lambda: {"x": 0, "y": 0})  # wp:posOffset
    extent: Dict[str, int]     = field(default_factory=lambda: {"w": 0, "h": 0})  # wp:extent@cx,cy
    wrap: Optional[str] = None                         # "wrapNone|wrapSquare|wrapTight|wrapThrough|wrapTopAndBottom|null"
    z: int = 0                                         # wp:anchor@relativeHeight

    # --- 문맥(군집화 가중치용, 없으면 None/False) ---
    context: Dict[str, Any] = field(default_factory=lambda: {
        "p_idx": None, "doc_index": None, "tbl_idx": None, "tc_sig": None,
        "sect_idx": None, "in_header": False, "in_footer": False
    })

    # --- 텍스트 시그니처 ---
    text: Dict[str, Any] = field(default_factory=lambda: {
        "raw_runs": [],      # List[str]
        "norm": "",          # 정규화 텍스트
        "circled_num": False # ①~⑳, ⓐ 등 감지
    })

    # --- 커넥터 정보 ---
    connector: Dict[str, Any] = field(default_factory=lambda: {
        "is_connector": False,
        "st_cxn_id": None,
        "end_cxn_id": None,
        "has_arrow_head": None,
        "has_arrow_tail": None
    })

    # --- 내부 변환 / 단순좌표 / 신뢰도 / 페이지힌트 / 그룹 ---
    xfrm: Dict[str, Any] = field(default_factory=lambda: {
        "off": {"x": 0, "y": 0},
        "ext": {"w": 0, "h": 0},
        "rotation": 0
    })
    simple_pos: Dict[str, Any] = field(default_factory=lambda: {"enabled": False, "x": 0, "y": 0})
    position_confidence: float = 0.0                   # page/margin=1.0, column/paragraph/character=0.6, inline=0.0, vml=0.8
    page_hint: Optional[int] = None
    group_hierarchy: Dict[str, Any] = field(default_factory=lambda: {"parents": []})

    # --- 이미지/출처 ---
    image: Optional[Dict[str, Any]] = None             # {"rIds":[...], "filename": "..."} 등 (있으면)
    provenance: Dict[str, Any] = field(default_factory=lambda: {
        "xml_snippet": None,
        "node_path_hint": None
    })

# --- Parser Abstract Class ---
class BaseParser(ABC):
    def __init__(self) -> None:
        self.supported_formats: List[str] = []

    def can_handle(self, file_path: Union[str, Path]) -> bool:
        return Path(file_path).suffix.lower() in self.supported_formats

    @property
    @abstractmethod
    def provides(self) -> set:
        """
        이 파서가 책임지는 상위 키들을 반환.
        병합기/오케스트레이터가 충돌 없이 합치도록 사용.
        예: {'paragraphs', 'tables'} 또는 {'headers', 'footers', 'relationships'}
        """
        ...

    @abstractmethod
    async def parse(self, file_path: Union[str, Path, BinaryIO]) -> ParseResult:
        """문서 파싱 (추상 메서드)"""
        ...
