"""
이미지 컴포넌트 정제기 (Image Sanitizer)

이 모듈은 파서(parser)로부터 추출된 인라인 이미지(`InlineImageRecord`) 목록을 받아,
추가 정보를 결합하고 OCR 및 이미지 요약을 수행하여 VDB에 적재하기 좋은 형태인
`ImageComponentData` 목록으로 변환하는 역할을 합니다.

주요 기능:
- **위치 분석**: 이미지가 테이블 또는 문단 내에 포함되어 있는지 여부를 확인합니다.
- **OCR 수행**: `.png` 확장자를 가진 이미지에 대해 `pytesseract`를 사용하여 텍스트를 추출합니다.
- **LLM 호출**: LLM(예: Ollama)을 호출하여 이미지에 대한 요약을 생성합니다. (젬마3:4B-양자화된 멀티모달 모델 사용.)
- **이미지 요약**: Ollama API를 호출하여 이미지에 대한 설명을 생성합니다.
- **캐싱(Caching)**: OCR 및 요약 결과를 `ocr_cache.json`에 캐싱하여 중복 API 호출을 방지합니다.
"""
from __future__ import annotations
import os
import base64
import mimetypes
import asyncio
import time # time.sleep 사용을 위해 import
from typing import List, Dict, Any
from pathlib import Path
import logging
import hashlib

import httpx

try:
    import pytesseract
    from PIL import Image
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False

from mypkg.core.parser import InlineImageRecord, TableRecord, ParagraphRecord
from mypkg.core.docjson_types import ImageComponentData
from mypkg.core.io import component_paths_from_sanitized, save_json, load_json

log = logging.getLogger(__name__)

def _get_file_hash(path: Path) -> str | None:
    """파일의 SHA256 해시를 계산합니다."""
    if not path.exists():
        return None
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        buf = f.read(65536)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(65536)
    return hasher.hexdigest()

class ImageSanitizer:
    """인라인 이미지에 대해 OCR, 요약 및 위치 분석을 수행합니다."""

    def __init__(self, lang: str = 'kor+eng', enable_llm: bool | None = None):
        self.lang = lang
        self._llm_call_count = 0
        if not PYTESSERACT_AVAILABLE:
            log.warning("pytesseract 또는 Pillow 라이브러리가 설치되지 않았습니다. OCR 기능이 비활성화됩니다.")
        env_enable = os.getenv("ECM_ENABLE_IMAGE_LLM")
        if enable_llm is None and env_enable is not None:
            enable_llm = env_enable.lower() in {"1", "true", "yes", "on"}
        self.enable_llm = bool(enable_llm)
        if not self.enable_llm:
            log.info("LLM 기반 이미지 요약이 비활성화되었습니다.")

    def _get_image_payload(self, image_path: Path, model_name: str) -> Dict[str, Any] | None:
        """Ollama API 요청을 위한 페이로드를 생성합니다."""
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        # Ollama는 mime_type 대신 base64 이미지 자체를 images 배열에 넣습니다.

        prompt_text = """
            Classify the image as one of: icon, software interface screenshot, table, or graph.
            
            Rules:
            - If the image appears very small, minimal, or symbolic (like a toolbar icon or button), classify as icon.
            - If it shows an entire application window or multiple interface panels, classify as software interface screenshot.
            - ...
            Output only one word.


        """
        
        return {
            "model": model_name,
            "prompt": prompt_text,
            "images": [base64_image],
            "stream": False # 스트리밍 비활성화
        }

    @staticmethod
    def _normalize_summary(text: str | None) -> str | None:
        if not text:
            return None
        summary = text.strip()
        lowered = summary.lower()
        for prefix in ("here is", "here's", "this is", "here are", "here're"):
            if lowered.startswith(prefix):
                cut_len = len(prefix)
                summary = summary[cut_len:].lstrip(":,-. \t")
                break
        return summary

    # 기존 동기 파이프라인을 위한 동기 메서드
    def _get_image_summary(self, image_path: Path, model_name: str = "gemma3:4b") -> str | None:
        if not image_path.exists(): return None

        payload = self._get_image_payload(image_path, model_name)
        if not payload: return None

        api_url = "http://localhost:11434/api/generate"

        result = None

        try:
            time.sleep(1) # Rate limit 회피를 위한 지연 시간 추가
            with httpx.Client() as client:
                response = client.post(api_url, json=payload, timeout=60.0)
                response.raise_for_status()
                result = response.json()
            
                return result.get("response", "").strip()
        except Exception as e:
            log.error(f"Sync image summary failed for {image_path}: {e} - Full response: {result}")
            return None
        
    def sanitize(self, inline_images: List[InlineImageRecord], tables: List[TableRecord], sanitized_paragraphs: List[ParagraphRecord], base_dir: Path, sanitized_path: Path, basename: str) -> List[ImageComponentData]:
        sanitized_images: List[ImageComponentData] = []
        
        rids_in_tables = {rId for table in tables or [] for row in table.rows for cell in row for rId in cell.inline_images}
        para_text_map = {p.doc_index: p.text for p in sanitized_paragraphs if p.doc_index is not None and p.text}

        cache_path = component_paths_from_sanitized(sanitized_path, basename)["ocr_cache"]
        ocr_cache = load_json(cache_path) if cache_path.exists() else {}

        for image_record in inline_images or []:
            doc_index = image_record.doc_index
            is_in_table = image_record.rId in rids_in_tables
            is_in_paragraph = doc_index in para_text_map and f"[image:{image_record.rId}]" in para_text_map[doc_index]

            preceding_text = None
            if doc_index is not None:
                if is_in_paragraph:
                    preceding_text = para_text_map[doc_index]
                elif (doc_index - 1) in para_text_map:
                    preceding_text = para_text_map[doc_index - 1]

            ocr_text = None
            llm_text = None
            full_image_path = None

            if image_record.saved_path:
                full_image_path = base_dir / image_record.saved_path
                if full_image_path.suffix.lower() == '.png':
                    image_hash = _get_file_hash(full_image_path)
                    cache_key = str(full_image_path)
                    cached_data = ocr_cache.get(cache_key, {}) if image_hash else {}

                    if cached_data.get('hash') == image_hash:
                        ocr_text = cached_data.get('ocr_text')
                        if self.enable_llm:
                            llm_text = cached_data.get('llm_text')
                    else:
                        ocr_text = self._perform_ocr(full_image_path)
                        if self.enable_llm:
                            summary_raw = self._get_image_summary(full_image_path)
                            summary = self._normalize_summary(summary_raw)
                            self._llm_call_count += 1
                            if self._llm_call_count % 50 == 0:
                                log.info(f"Image summary generation completed for {self._llm_call_count} images.")
                            if summary:
                                llm_text = f"image description: {summary}"

                        if image_hash:
                            cache_entry = {'hash': image_hash, 'ocr_text': ocr_text}
                            if self.enable_llm:
                                cache_entry['llm_text'] = llm_text
                            ocr_cache[cache_key] = cache_entry

            component = ImageComponentData(
                rId=image_record.rId,
                filename=image_record.filename,
                doc_index=doc_index,
                saved_path=str(full_image_path) if full_image_path else image_record.saved_path,
                ocr_text=ocr_text,
                is_in_table=is_in_table,
                is_in_paragraph=is_in_paragraph,
                preceding_text=preceding_text,
                llm_text=llm_text,
            )
            sanitized_images.append(component)

        save_json(ocr_cache, cache_path)
        return sanitized_images

    def build_components(self, sanitized_images: List[ImageComponentData]) -> Dict[str, List[Dict[str, Any]]]:
        return {"images": [img.to_dict() for img in sanitized_images]}

    def _perform_ocr(self, image_path: Path) -> str | None:
        if not PYTESSERACT_AVAILABLE or not image_path.exists() or image_path.suffix.lower() != '.png':
            return None
        try:
            with Image.open(image_path) as img:
                text = pytesseract.image_to_string(img, lang=self.lang)
                return text.strip()
        except Exception as e:
            log.error(f"OCR 실패: {image_path}: {e}")
            if "tesseract is not installed or it's not in your PATH" in str(e).lower():
                log.error("Tesseract OCR 엔진이 설치되지 않았거나 시스템 PATH에 등록되지 않은 것 같습니다.")
            return None
