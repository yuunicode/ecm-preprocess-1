from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


# General FS helpers
def ensure_dir(p: Path) -> Path:
    """디렉터리가 존재하도록 보장한다.

    매개변수:
        p: 생성(보장)할 디렉터리 경로.

    반환값:
        입력과 동일한 경로 `p` (존재하도록 생성 후 반환).
    """
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json_output(data: Dict[str, Any], path: Path) -> None:
    """딕셔너리를 JSON 파일로 저장한다.

    상위 디렉터리를 먼저 생성한 뒤, UTF-8 인코딩과 들여쓰기를 적용해
    사람이 읽기 쉬운 JSON으로 기록한다.

    매개변수:
        data: JSON으로 직렬화 가능한 사전 객체.
        path: 저장할 대상 파일 경로.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def project_root() -> Path:
    """`mypkg` 모듈 위치를 기준으로 프로젝트 루트를 반환한다."""
    return _project_root_from_mypkg()


def datasets_root() -> Path:
    """프로젝트 루트의 형제 디렉터리 `_datasets` 경로를 반환한다."""
    return (project_root().parent / "_datasets").resolve()


def ecminer_docx_root() -> Path:
    """`_datasets/ecminer` DOCX 소스 디렉터리 경로를 반환한다."""
    return datasets_root() / "ecminer"


def default_output_root_path() -> Path:
    """프로젝트 루트 하위 기본 출력 루트(`output/`) 경로를 반환한다."""
    return project_root() / "output"


def default_processed_root_path() -> Path:
    """기본 processed 출력 루트(`output/processed`) 경로를 반환한다."""
    return default_output_root_path() / "processed"


def _project_root_from_mypkg() -> Path:
    """이 파일의 위치로부터 프로젝트 루트를 추정한다.

    디렉터리 구조 가정:
        <project>/src/mypkg/core/io.py
    위 구조에서 프로젝트 루트는 이 파일로부터 상위 3단계 지점이다.

    반환값:
        추정된 프로젝트 루트의 절대 경로.
    """
    here = Path(__file__).resolve()
    parents = here.parents
    idx = 3 if len(parents) > 3 else (len(parents) - 1)
    return parents[idx]


def _resolve_from_mypkg_root(path: str | Path) -> Path:
    """프로젝트 루트를 기준으로 상대 경로를 절대 경로로 변환한다.

    매개변수:
        path: 절대 또는 상대 경로.

    반환값:
        절대 경로. 상대 경로 입력은 프로젝트 루트를 기준으로 해석한다.
    """
    p = Path(path)
    if p.is_absolute():
        return p
    return (_project_root_from_mypkg() / p).resolve()


def resolve_sanitized_path(sanitized_path: str | Path) -> Path:
    """sanitized(JSON) 파일 경로를 절대 경로로 반환한다. (별칭)

    상대 경로가 주어진 경우, `mypkg`의 위치로부터 추정한 프로젝트 루트를
    기준으로 절대 경로를 계산한다.

    매개변수:
        sanitized_path: 절대 또는 프로젝트 루트 기준 상대 경로.

    반환값:
        절대 경로(`Path`).
    """
    return _resolve_from_mypkg_root(sanitized_path)


def version_dir_from_sanitized(sanitized_path: str | Path) -> Path:
    """sanitized 파일 경로로부터 버전 디렉터리를 반환한다.

    구조 가정:
        <...>/processed/{base_name}/{version}/_sanitized/<file>.json
    여기서 버전 디렉터리는 `<...>/processed/{base_name}/{version}`.

    매개변수:
        sanitized_path: sanitized JSON 경로(절대 또는 상대).

    반환값:
        버전 디렉터리의 절대 경로(`Path`).
    """
    p = resolve_sanitized_path(sanitized_path)
    return p.parent.parent


def base_dir_from_sanitized(sanitized_path: str | Path) -> Path:
    """sanitized 파일 경로로부터 베이스 디렉터리(`{base_name}`)를 반환한다.

    구조: `<...>/processed/{base_name}/{version}/_sanitized/<file>.json` 에서 `{base_name}` 디렉터리.
    """
    p = resolve_sanitized_path(sanitized_path)
    return p.parent.parent.parent


def components_dir_from_sanitized(sanitized_path: str | Path) -> Path:
    """컴포넌트 디렉터리를 반환(필요 시 생성)한다.

    디렉터리 이름은 버전 디렉터리(`{version}`) 하위의 `_comp` 이다.
    """
    return ensure_dir(version_dir_from_sanitized(sanitized_path) / "_comp")


def meta_path_from_sanitized(sanitized_path: str | Path, basename: str) -> Path:
    """문서 메타데이터 JSON 경로를 반환한다.

    버전 디렉터리 하위 `_meta/<basename>.json` 경로를 사용하며,
    디렉터리가 없으면 생성한다.

    매개변수:
        sanitized_path: sanitized JSON 경로(절대 또는 상대).
        basename: 원본 문서의 베이스 이름.

    반환값:
        메타데이터 JSON 파일의 절대 경로.
    """
    d = ensure_dir(version_dir_from_sanitized(sanitized_path) / "_meta")
    return d / f"{basename}.json"


def component_paths_from_sanitized(sanitized_path: str | Path, basename: str) -> Dict[str, Path]:
    """컴포넌트 JSON 파일 경로들을 구성한다.

    매개변수:
        sanitized_path: sanitized JSON 경로(절대 또는 상대).
        basename: 출력 파일명의 접두에 사용할 베이스 이름.

    반환값:
        `list`, `table`, `blocks`, `image`, `paragraph`, `ocr_cache` 키를 가지는 경로 매핑.
    """
    base = components_dir_from_sanitized(sanitized_path)
    return {
        "list":    base / "list_comp.json",
        "table":   base / "table_comp.json",
        "paragraph": base / "parag_comp.json",
        "blocks":  base / f"{basename}_blocks.json",
        "image": base / "image_comp.json",
        "ocr_cache": base / "ocr_cache.json",
    }


def docjson_output_path_from_sanitized(sanitized_path: str | Path, basename: str) -> Path:
    """최종 DocJSON 출력 경로를 계산한다.

    규칙: `<processed>/{base_name}/{basename}_{version}.docjson`
    즉, 문서별 디렉터리 바로 아래에 `{base_name}_{version}.docjson` 파일로 저장한다.
    """
    version_dir = version_dir_from_sanitized(sanitized_path)
    base_dir = version_dir.parent
    version = version_dir.name
    return base_dir / f"{basename}_{version}.docjson"


def save_json(obj: Any, path: Path) -> Path:
    """객체를 JSON으로 직렬화하여 `path`에 저장한다.

    상위 디렉터리를 생성한 뒤, UTF-8과 들여쓰기를 적용해 저장한다.

    매개변수:
        obj: JSON으로 직렬화 가능한 객체.
        path: 저장 대상 파일 경로.

    반환값:
        저장된 대상 경로(`Path`).
    """
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_json(path: Path) -> Any:
    """`path`로부터 JSON을 읽어 파싱한다.

    매개변수:
        path: 소스 파일 경로.

    반환값:
        파싱된 파이썬 객체.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def save_list_components_from_sanitized(obj: Dict[str, Any], sanitized_path: str | Path, basename: str) -> Path:
    """리스트 컴포넌트 페이로드를 `_comp`에 저장한다."""
    return save_json(obj, component_paths_from_sanitized(sanitized_path, basename)["list"])


def save_table_components_from_sanitized(obj: Dict[str, Any], sanitized_path: str | Path, basename: str) -> Path:
    """테이블 컴포넌트 페이로드를 `_comp`에 저장한다."""
    return save_json(obj, component_paths_from_sanitized(sanitized_path, basename)["table"])


def save_image_components_from_sanitized(obj: Dict[str, Any], sanitized_path: str | Path, basename: str) -> Path:
    """이미지 컴포넌트 페이로드를 `_comp`에 저장한다."""
    return save_json(obj, component_paths_from_sanitized(sanitized_path, basename)["image"])


def save_paragraph_components_from_sanitized(obj: Dict[str, Any], sanitized_path: str | Path, basename: str) -> Path:
    """문단 컴포넌트 페이로드를 `_comp`에 저장한다."""
    return save_json(obj, component_paths_from_sanitized(sanitized_path, basename)["paragraph"])


def save_blocks_components_from_sanitized(obj: Dict[str, Any], sanitized_path: str | Path, basename: str) -> Path:
    """문서 블록(assembled blocks) 컴포넌트를 `_comp`에 저장한다."""
    return save_json(obj, component_paths_from_sanitized(sanitized_path, basename)["blocks"])


def load_available_components_from_sanitized(sanitized_path: str | Path, basename: str) -> Dict[str, Any]:
    """존재하는 컴포넌트 JSON들을 로드해 합쳐서 반환한다."""
    paths = component_paths_from_sanitized(sanitized_path, basename)
    out: Dict[str, Any] = {}
    if paths["list"].exists():
        lo = load_json(paths["list"])
        out["lists"] = lo.get("lists", [])
        out["consumed"] = lo.get("consumed", [])
    if paths["table"].exists():
        to = load_json(paths["table"])
        out["tables"] = to.get("tables", [])
    if paths["paragraph"].exists():
        po = load_json(paths["paragraph"])
        out["paragraphs"] = po.get("paragraphs", [])
    if paths["blocks"].exists():
        bo = load_json(paths["blocks"])
        out["blocks"] = bo.get("blocks", [])
    if paths["image"].exists():
        io = load_json(paths["image"])
        out["images"] = io.get("images", io)
    return out
