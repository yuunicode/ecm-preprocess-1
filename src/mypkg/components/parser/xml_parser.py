import html, logging, time, zipfile, xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Union, BinaryIO, Optional, Tuple
import copy
from mypkg.core.parser import (
    BaseParser,
    ParseResult,
    TableRecord,
    TableCellRecord,
    RelationshipRecord,
    ParagraphRecord,
    RunRecord,
    InlineImageRecord,
)

log = logging.getLogger(__name__)

NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "wp": "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "pic":"http://schemas.openxmlformats.org/drawingml/2006/picture",
    "wps":"http://schemas.microsoft.com/office/word/2010/wordprocessingShape",
    "wpg":"http://schemas.microsoft.com/office/word/2010/wordprocessingGroup",
    "v": "urn:schemas-microsoft-com:vml",
    "o": "urn:schemas-microsoft-com:office:office",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "m": "http://schemas.openxmlformats.org/officeDocument/2006/math",
}

_MATH_TAGS = {
    f"{{{NS['m']}}}oMath",
    f"{{{NS['m']}}}oMathPara",
}

_PUNCT_FOLLOW_NO_SPACE = set(".,;:!?)]}%'")
_PUNCT_PREV_NO_SPACE = set("([{/'\"")


def _concat_tokens(left: str, right: str) -> str:
    if not left:
        return right
    if not right:
        return left

    left = left.rstrip()
    right = right.lstrip()
    if not left:
        return right
    if not right:
        return left

    last = left[-1]
    first = right[0]

    join_without_space = False
    if last in _PUNCT_PREV_NO_SPACE:
        join_without_space = True
    elif first in _PUNCT_FOLLOW_NO_SPACE:
        join_without_space = True
    elif last.isdigit() and first.isdigit():
        join_without_space = True
    elif last.isdigit() and first in "%)]":
        join_without_space = True
    elif last.isalpha() and first.isdigit():
        join_without_space = True
    elif last in ".-/" and first.isdigit():
        join_without_space = True
    elif last == '-' and first.isalpha():
        join_without_space = True

    if join_without_space:
        return f"{left}{right}"
    return f"{left} {right}"


def _join_tokens(tokens: List[str]) -> str:
    result = ""
    for token in tokens:
        if not token:
            continue
        result = token if not result else _concat_tokens(result, token)
    return result

class DocxXmlParser(BaseParser):
    def __init__(self):
        super().__init__();
        self.supported_formats=['.docx']

    @property
    def provides(self) -> set:
        return {"paragraphs", "tables", "inline_images"}

    async def parse(self, file_path: Union[str, Path, BinaryIO]) -> ParseResult:
        t0 = time.time()
        tables: List[TableRecord] = []
        relationships: Dict[str, RelationshipRecord] = {}
        paragraphs: List[ParagraphRecord] = []
        numbering_info: Dict[str, Dict[str, Any]] = {}

        try:
            with zipfile.ZipFile(file_path, "r") as zf:
                table_styles: Dict[str, Dict[str, str]] = {}

                # 관계 정보 선처리
                if "word/_rels/document.xml.rels" in zf.namelist():
                    rels_root = ET.fromstring(zf.read("word/_rels/document.xml.rels"))
                    for rel in rels_root.findall('.//{http://schemas.openxmlformats.org/package/2006/relationships}Relationship'):
                        rid = rel.attrib.get("Id")
                        if rid:
                            relationships[rid] = RelationshipRecord(
                                rid=rid,
                                type=rel.attrib.get("Type", ""),
                                target=rel.attrib.get("Target", ""),
                            )

                inline_images_map = self._build_inline_image_map(relationships)

                # 번호 포맷 정보 추출
                if "word/numbering.xml" in zf.namelist():
                    numbering_info = self._parse_numbering(zf.read("word/numbering.xml"))
                paragraph_styles: Dict[str, str] = {}
                if "word/styles.xml" in zf.namelist():
                    table_styles, paragraph_styles = self._parse_styles(zf.read("word/styles.xml"))

                paragraphs: List[ParagraphRecord] = []
                if "word/document.xml" in zf.namelist():
                    root = ET.fromstring(zf.read("word/document.xml"))
                    body = root.find(".//w:body", NS)
                    if body is not None:
                        doc_index = 0
                        for elem in body:
                            if elem.tag == f'{{{NS["w"]}}}p':
                                para_record = self._parse_paragraph_element(
                                    elem,
                                    doc_index,
                                    numbering_info,
                                    paragraph_styles,
                                    inline_images_map,
                                )
                                if para_record:
                                    paragraphs.append(para_record)
                                doc_index += 1
                            elif elem.tag == f'{{{NS["w"]}}}tbl':
                                style_id = None
                                tblPr = elem.find("./w:tblPr", NS)
                                if tblPr is not None:
                                    style_elem = tblPr.find("./w:tblStyle", NS)
                                    if style_elem is not None:
                                        style_id = style_elem.attrib.get(f'{{{NS["w"]}}}val')
                                table_record, nested_records = self._parse_table_element(
                                    elem,
                                    doc_index,
                                    table_styles,
                                    style_id,
                                    inline_images_map,
                                )
                                if table_record:
                                    tables.append(table_record)
                                if nested_records:
                                    tables.extend(nested_records)
                                doc_index += 1

                inline_images = list(inline_images_map.values())
                for record in inline_images:
                    if record.doc_indices:
                        record.doc_indices = sorted(set(record.doc_indices))
                        if record.doc_index is None:
                            record.doc_index = record.doc_indices[0]

            paragraphs_docx = paragraphs

            out = {
                "paragraphs": paragraphs_docx,
                "tables": tables,
                "inline_images": inline_images,
            }
            return ParseResult(True, out, processing_time=time.time()-t0, payload_tag="docx_xml")

        except Exception as e:
            log.exception("xml parser failed")
            return ParseResult(False, error=str(e), processing_time=time.time()-t0, payload_tag="docx_xml")

    # ---- helpers ----
    def _extract_paragraph_text_and_math(self, p: ET.Element) -> tuple[str, List[str]]:
        segments: List[str] = []
        math_texts: List[str] = []

        def traverse(node: ET.Element) -> None:
            for child in list(node):
                tag = child.tag
                if tag == f"{{{NS['w']}}}t":
                    val = child.text or ""
                    if val.strip():
                        segments.append(val)
                elif tag in _MATH_TAGS:
                    expr = self._serialize_math(child)
                    if expr:
                        math_texts.append(expr)
                        segments.append(expr)
                else:
                    traverse(child)

        traverse(p)
        text = _join_tokens(segments).strip()
        return text, math_texts

    def _serialize_math(self, math_elem: ET.Element) -> str:
        tokens: List[str] = []
        math_text_tag = f"{{{NS['m']}}}t"
        for node in math_elem.iter():
            if node.tag == math_text_tag:
                val = node.text or ""
                if val.strip():
                    tokens.append(val)
        return _join_tokens(tokens).strip()

    def _parse_numbering(self, numbering_xml_bytes: bytes) -> Dict[str, Dict[str, Any]]:
        numbering_info = {}
        root = ET.fromstring(numbering_xml_bytes)
        abstract_nums = {}
        for an in root.findall("w:abstractNum", NS):
            an_id = an.attrib.get(f'{{{NS["w"]}}}abstractNumId')
            if an_id is None:
                continue
            abstract_nums[an_id] = {}
            for lvl in an.findall("w:lvl", NS):
                lvl_idx = lvl.attrib.get(f'{{{NS["w"]}}}ilvl')
                numFmt_tag = lvl.find("w:numFmt", NS)
                if lvl_idx is not None and numFmt_tag is not None:
                    numFmt = numFmt_tag.attrib.get(f'{{{NS["w"]}}}val')
                    list_type = 'number' if numFmt not in ['bullet'] else 'bullet'
                    abstract_nums[an_id][lvl_idx] = {"numFmt": numFmt, "list_type": list_type}

        for num in root.findall("w:num", NS):
            num_id = num.attrib.get(f'{{{NS["w"]}}}numId')
            abstract_id_tag = num.find("w:abstractNumId", NS)
            if num_id is None or abstract_id_tag is None:
                continue
            abstract_id = abstract_id_tag.attrib.get(f'{{{NS["w"]}}}val')
            if abstract_id in abstract_nums:
                numbering_info[num_id] = abstract_nums[abstract_id]
        return numbering_info

    def _build_inline_image_map(
        self,
        relationships: Dict[str, RelationshipRecord],
    ) -> Dict[str, InlineImageRecord]:
        inline_images: Dict[str, InlineImageRecord] = {}
        for rid, rel in relationships.items():
            rel_type = (rel.type or "").lower()
            if "image" not in rel_type and "oleobject" not in rel_type:
                continue
            target = rel.target or ""
            if not target:
                continue
            if target.startswith("/"):
                part_path = target.lstrip("/")
            else:
                part_path = target if target.startswith("word/") else f"word/{target}"
            filename = Path(part_path).name
            inline_images[rid] = InlineImageRecord(
                rId=rid,
                filename=filename,
                saved_path=part_path,
            )
        return inline_images

    def _build_run_records(
        self,
        paragraph: ET.Element,
        inline_images_map: Dict[str, InlineImageRecord],
        doc_index: int,
    ) -> Tuple[List[RunRecord], List[str]]:
        runs: List[RunRecord] = []
        emphasized: List[str] = []

        for r in paragraph.findall(".//w:r", NS):
            text_segments: List[str] = []
            for t in r.findall(".//w:t", NS):
                if t.text:
                    text_segments.append(t.text)
            run_text = "".join(text_segments)
            image_rids: List[str] = []
            for blip in r.findall(".//a:blip", NS):
                rid = blip.get(f'{{{NS["r"]}}}embed')
                if rid:
                    image_rids.append(rid)
            for imagedata in r.findall(".//v:imagedata", NS):
                rid = imagedata.get(f'{{{NS["r"]}}}id')
                if rid:
                    image_rids.append(rid)
            for ole in r.findall(".//o:OLEObject", NS):
                rid = ole.get(f'{{{NS["r"]}}}id')
                if rid:
                    image_rids.append(rid)
            if image_rids:
                unique_rids: List[str] = []
                seen: set[str] = set()
                for rid in image_rids:
                    if rid in seen:
                        continue
                    seen.add(rid)
                    unique_rids.append(rid)
                image_rids = unique_rids
            for rid in image_rids:
                token = f"[image:{rid}]"
                if run_text:
                    run_text = _concat_tokens(run_text, token)
                else:
                    run_text = token
                record = inline_images_map.get(rid)
                if record:
                    if doc_index not in record.doc_indices:
                        record.doc_indices.append(doc_index)
                    if record.doc_index is None:
                        record.doc_index = doc_index

            rPr = r.find("w:rPr", NS)
            b = False
            i = False
            u = False
            r_style: Optional[str] = None
            sz: Optional[float] = None
            color_val: Optional[str] = None
            if rPr is not None:
                b = rPr.find("w:b", NS) is not None or rPr.find("w:bCs", NS) is not None
                i = rPr.find("w:i", NS) is not None or rPr.find("w:iCs", NS) is not None
                u = rPr.find("w:u", NS) is not None
                rStyle_elem = rPr.find("w:rStyle", NS)
                if rStyle_elem is not None:
                    r_style = rStyle_elem.attrib.get(f'{{{NS["w"]}}}val')
                sz_elem = rPr.find("w:sz", NS)
                if sz_elem is not None:
                    val = sz_elem.attrib.get(f'{{{NS["w"]}}}val')
                    if val:
                        try:
                            sz = float(val) / 2
                        except ValueError:
                            sz = None
                color_elem = rPr.find("w:color", NS)
                if color_elem is not None:
                    color_val = color_elem.attrib.get(f'{{{NS["w"]}}}val')
                    if color_val:
                        color_val = color_val.upper()

            # Append OMML math expressions, if any
            for math_elem in r.findall(".//m:oMath", NS) + r.findall(".//m:oMathPara", NS):
                expr = self._serialize_math(math_elem)
                if expr:
                    if run_text:
                        run_text = _concat_tokens(run_text, expr)
                    else:
                        run_text = expr

            run_record = RunRecord(
                text=run_text,
                b=b,
                i=i,
                u=u,
                rStyle=r_style,
                sz=sz,
                color=color_val,
                image_rids=image_rids,
            )
            runs.append(run_record)

            if b and run_text.strip():
                emphasized.append(run_text.strip())

        # Math elements that are direct children of the paragraph (not wrapped in w:r)
        direct_math_nodes = list(paragraph.findall("m:oMath", NS)) + list(paragraph.findall("m:oMathPara", NS))
        for math_elem in direct_math_nodes:
            expr = self._serialize_math(math_elem)
            if not expr:
                continue
            runs.append(
                RunRecord(
                    text=expr,
                    b=False,
                    i=False,
                    u=False,
                    rStyle=None,
                    sz=None,
                    color=None,
                    image_rids=[],
                )
            )

        return runs, emphasized

    def _parse_paragraph_element(
        self,
        p: ET.Element,
        doc_index: int,
        numbering_info: Dict[str, Dict[str, Any]],
        paragraph_styles: Dict[str, str],
        inline_images_map: Dict[str, InlineImageRecord],
    ):
        text_raw, math_texts = self._extract_paragraph_text_and_math(p)

        runs, emphasized = self._build_run_records(p, inline_images_map, doc_index)
        paragraph_text = "".join(run.text for run in runs).strip()
        if not paragraph_text:
            paragraph_text = text_raw

        para_record = None
        if paragraph_text or math_texts:
            style_name: Optional[str] = None
            pPr = p.find("w:pPr", NS)
            if pPr is not None:
                pStyle = pPr.find("w:pStyle", NS)
                if pStyle is not None:
                    style_id = pStyle.attrib.get(f'{{{NS["w"]}}}val')
                    if style_id:
                        style_name = paragraph_styles.get(style_id, style_id)

            para_record = ParagraphRecord(
                text=paragraph_text,
                doc_index=doc_index,
                style=style_name,
                runs=runs,
                emphasized=emphasized,
                math_texts=math_texts,
            )
            para_record.source_doc_indices = [doc_index]

        return para_record

    def _parse_table_element(
        self,
        tbl: ET.Element,
        doc_index: int,
        table_styles: Dict[str, Dict[str, str]],
        style_id: Optional[str],
        inline_images_map: Dict[str, InlineImageRecord],
        tid_seed: Optional[str] = None,
    ) -> Tuple[TableRecord, List[TableRecord]]:
        rows_data: List[List[TableCellRecord]] = []
        has_borders = False  # 기본값은 테두리 없음으로 가정
        nested_tables: List[TableRecord] = []
        nested_counter = 0
        tid = tid_seed or f"t{doc_index}"

        tblPr = tbl.find("./w:tblPr", NS)
        if tblPr is not None:
            tblBorders = tblPr.find("./w:tblBorders", NS)
            if tblBorders is not None:
                for border_tag in list(tblBorders):
                    val = border_tag.get(f'{{{NS["w"]}}}val')
                    if val and val.lower() not in ('none', 'nil'):
                        has_borders = True
                        break

        row_idx = 0
        for tr in tbl.findall("./w:tr", NS):
            row_data: List[TableCellRecord] = []
            col_idx = 0
            for tc in tr.findall("./w:tc", NS):
                tcPr = tc.find("w:tcPr", NS)
                gridSpan = 1
                vMerge = None
                cell_has_bold = False
                cell_has_style = False
                if tcPr is not None:
                    gs = tcPr.find("w:gridSpan", NS)
                    if gs is not None:
                        try:
                            gridSpan = int(gs.attrib.get('{%s}val' % NS["w"], "1"))
                        except Exception:
                            gridSpan = 1
                    vm = tcPr.find("w:vMerge", NS)
                    if vm is not None:
                        vMerge = vm.attrib.get('{%s}val' % NS["w"]) or "restart"
                    tcBorders = tcPr.find("w:tcBorders", NS)
                    if tcBorders is not None and not has_borders:
                        for border_tag in list(tcBorders):
                            val = border_tag.get(f'{{{NS["w"]}}}val')
                            if val and val.lower() not in ('none', 'nil'):
                                has_borders = True
                                break
                    shd = tcPr.find("w:shd", NS)
                    if shd is not None:
                        fill = shd.attrib.get(f'{{{NS["w"]}}}fill')
                        if fill:
                            fill = fill.strip()
                            if fill and fill.lower() != "auto":
                                cell_has_style = True

                texts: List[str] = []
                cell_inline_images: List[str] = []
                for child_elem in tc:
                    if child_elem.tag == f'{{{NS["w"]}}}p':
                        runs, _ = self._build_run_records(child_elem, inline_images_map, doc_index)
                        if runs:
                            texts.append("".join(run.text for run in runs).strip())
                            if any(run.b for run in runs):
                                cell_has_bold = True
                            for run in runs:
                                cell_inline_images.extend(run.image_rids)
                    elif child_elem.tag == f'{{{NS["w"]}}}tbl':
                        if not has_borders:
                            nested_counter += 1
                            nested_tid = f"{tid}_n{nested_counter}"
                            nested_record, deeper_nested = self._parse_table_element(
                                child_elem,
                                doc_index,
                                table_styles,
                                style_id,
                                inline_images_map,
                                tid_seed=nested_tid,
                            )
                            nested_record.doc_index = doc_index
                            nested_tables.append(nested_record)
                            nested_tables.extend(deeper_nested)
                            texts.append(f"[table:{nested_tid}]")
                        else:
                            nested_table_text = self._extract_text_from_table(child_elem, inline_images_map, doc_index)
                            if nested_table_text:
                                texts.append(nested_table_text)

                cell_text = " ".join(t for t in texts if t).strip()
                if not cell_has_style and style_id:
                    style_color = self._resolve_table_style_color(table_styles, style_id, row_idx, col_idx)
                    if style_color:
                        cell_has_style = True

                row_data.append(
                    TableCellRecord(
                        text=cell_text,
                        gridSpan=gridSpan,
                        vMerge=vMerge,
                        inline_images=sorted(set(cell_inline_images)),
                        has_style=cell_has_style,
                    )
                )
                col_idx += 1
            rows_data.append(row_data)
            row_idx += 1

        # 테이블 스타일에서 테두리 정보를 가져와 보정
        if not has_borders and style_id:
            style_entry = table_styles.get(style_id)
            if style_entry and style_entry.get("_hasBorders"):
                has_borders = True

        table_record = TableRecord(tid=tid, rows=rows_data, doc_index=doc_index, has_borders=has_borders)
        return table_record, nested_tables

    def _extract_text_from_table(self, tbl: ET.Element, inline_images_map: Dict[str, InlineImageRecord], doc_index: int) -> str:
        all_texts: List[str] = []
        for tr in tbl.findall("./w:tr", NS):
            for tc in tr.findall("./w:tc", NS):
                cell_texts: List[str] = []
                for child_elem in tc:
                    if child_elem.tag == f'{{{NS["w"]}}}p':
                        runs, _ = self._build_run_records(child_elem, inline_images_map, doc_index)
                        if runs:
                            cell_texts.append("".join(run.text for run in runs).strip())
                    elif child_elem.tag == f'{{{NS["w"]}}}tbl':
                        nested_text = self._extract_text_from_table(child_elem, inline_images_map, doc_index)
                        if nested_text:
                            cell_texts.append(nested_text)
                
                all_texts.append(" ".join(t for t in cell_texts if t).strip())
        return " ".join(t for t in all_texts if t).strip()

    def _parse_styles(
        self,
        styles_xml: bytes,
    ) -> Tuple[Dict[str, Dict[str, str]], Dict[str, str]]:
        table_styles: Dict[str, Dict[str, str]] = {}
        paragraph_styles: Dict[str, str] = {}
        root = ET.fromstring(styles_xml)
        for style in root.findall("w:style", NS):
            style_type = style.attrib.get(f'{{{NS["w"]}}}type')
            style_id = style.attrib.get(f'{{{NS["w"]}}}styleId')
            if not style_id:
                continue

            if style_type == "paragraph":
                name_elem = style.find("w:name", NS)
                if name_elem is not None:
                    value = name_elem.attrib.get(f'{{{NS["w"]}}}val')
                    if value:
                        paragraph_styles[style_id] = value
                        continue
                paragraph_styles.setdefault(style_id, style_id)

            if style_type == "table":
                entry: Dict[str, str] = {}
                has_style_borders = False
                tblPr = style.find("w:tblPr", NS)
                if tblPr is not None:
                    shd = tblPr.find("w:shd", NS)
                    fill = self._extract_shading_fill(shd)
                    if fill:
                        entry["wholeTable"] = fill
                    tblBorders = tblPr.find("w:tblBorders", NS)
                    if tblBorders is not None:
                        for border_tag in list(tblBorders):
                            val = border_tag.attrib.get(f'{{{NS["w"]}}}val')
                            if val and val.lower() not in ('none', 'nil'):
                                has_style_borders = True
                                break
                for tbl_style_pr in style.findall("w:tblStylePr", NS):
                    stype = tbl_style_pr.attrib.get(f'{{{NS["w"]}}}type')
                    if not stype:
                        continue
                    shd = None
                    tcPr = tbl_style_pr.find("w:tcPr", NS)
                    trPr = tbl_style_pr.find("w:trPr", NS)
                    if tcPr is not None:
                        shd = tcPr.find("w:shd", NS)
                    if shd is None and trPr is not None:
                        shd = trPr.find("w:shd", NS)
                    fill = self._extract_shading_fill(shd)
                    if fill:
                        entry[stype] = fill
                if has_style_borders:
                    entry["_hasBorders"] = True
                if entry:
                    table_styles[style_id] = entry

        return table_styles, paragraph_styles

    def _extract_shading_fill(self, shd: Optional[ET.Element]) -> Optional[str]:
        if shd is None:
            return None
        fill = shd.attrib.get(f'{{{NS["w"]}}}fill')
        if fill and fill.lower() != "auto":
            fill_norm = fill.strip().upper()
            if not fill_norm.startswith('#') and len(fill_norm) == 6:
                fill_norm = f"#{fill_norm}"
            return fill_norm
        theme_fill = shd.attrib.get(f'{{{NS["w"]}}}themeFill')
        if theme_fill:
            return theme_fill  # fallback to symbolic name
        theme_color = shd.attrib.get(f'{{{NS["w"]}}}themeColor')
        if theme_color:
            return theme_color
        return None

    def _resolve_table_style_color(
        self,
        table_styles: Dict[str, Dict[str, str]],
        style_id: str,
        row_idx: int,
        col_idx: int,
    ) -> Optional[str]:
        entry = table_styles.get(style_id)
        if not entry:
            return None
        if row_idx == 0 and "firstRow" in entry:
            return entry["firstRow"]
        if col_idx == 0 and "firstCol" in entry:
            return entry["firstCol"]
        return entry.get("wholeTable")

    def _parse_drawings_in_paragraph(
        self,
        p: ET.Element,
        doc_index: int,
        start_idx: int = 0,
        extra_context: Optional[Dict[str, Any]] = None,
        inline_images_map: Optional[Dict[str, InlineImageRecord]] = None,
    ) -> Tuple[List[Dict[str, Any]], int]:
        drawings: List[Dict[str, Any]] = []
        d_idx_in_p = start_idx
        
        def _texts(node):
            items = []
            for path in (".//a:t", ".//w:t", ".//wps:txBody//a:t", ".//a:txBody//a:t"):
                for t in node.findall(path, NS):
                    if t.text:
                        items.append({"text": t.text, "xpath": path})
            return items

        def _wrap(anchor):
            if anchor is None:
                return None
            for t in ("wrapNone","wrapSquare","wrapThrough","wrapTopAndBottom","wrapTight"):
                if anchor.find(f"./wp:{t}", NS) is not None:
                    return t
            return None

        def _image_rids(node: ET.Element) -> List[str]:
            rids: List[str] = []
            for blip in node.findall('.//a:blip', NS):
                rid = blip.get(f'{{{NS["r"]}}}embed')
                if rid:
                    rids.append(rid)
            return rids

        def _image_crop(node: ET.Element) -> Dict[str, int]:
            crop_elem = node.find('.//a:srcRect', NS)
            if crop_elem is None:
                return {}
            crop: Dict[str, int] = {}
            for key in ("l", "t", "r", "b"):
                val = crop_elem.get(key)
                if val is None:
                    continue
                try:
                    crop[key] = int(val)
                except (TypeError, ValueError):
                    continue
            return crop

        for drawing in p.findall(".//w:drawing", NS):
            entry = {
                "did": f"d{doc_index}_{d_idx_in_p}",
                "kind": None,
                "texts_raw": _texts(drawing),
                "image": None,
                "shape": None,
                "group": None,
                "page": None, # 페이지 정보는 별도 계산 필요
                "context": {
                    "p_idx": None, # p_idx는 전체 단락 기준이라 doc_index로 대체
                    "doc_index": doc_index,
                    "tbl_idx": None,
                    "tc_sig": None,
                    "sect_idx": 0, # 섹션 정보는 별도 계산 필요
                    "in_header": False,
                    "in_footer": False
                },
                "xml_snippet": ET.tostring(drawing, encoding="unicode", method="xml")[:4000]
            }
            if extra_context:
                entry["context"].update({k: v for k, v in extra_context.items() if v is not None})

            sp_root = (
                drawing.find(".//wps:wsp", NS) or
                drawing.find(".//wps:cxnSp", NS) or
                drawing.find(".//a:sp", NS) or
                drawing.find(".//a:cxnSp", NS)
            )
            if sp_root is not None:
                prst = None
                spPr = sp_root.find("./wps:spPr", NS) or sp_root.find("./a:spPr", NS) or sp_root.find(".//a:spPr", NS)
                if spPr is not None:
                    pg = spPr.find("./a:prstGeom", NS) or spPr.find(".//a:prstGeom", NS)
                    if pg is not None:
                        prst = pg.get("prst")
                entry["kind"] = "shape"
                entry["shape"] = {
                    "preset": prst,
                    "texts_raw": _texts(sp_root),
                    "tag": sp_root.tag
                }

            anchor = drawing.find(".//wp:anchor", NS)
            inline = drawing.find(".//wp:inline", NS)
            if anchor is not None:
                rel_h = anchor.find("./wp:positionH", NS)
                rel_v = anchor.find("./wp:positionV", NS)
                ph = anchor.find("./wp:positionH/wp:posOffset", NS)
                pv = anchor.find("./wp:positionV/wp:posOffset", NS)
                ext = anchor.find("./wp:extent", NS)
                entry["anchor"] = {
                    "type": "anchor",
                    "rel_from_h": rel_h.get("relativeFrom") if rel_h is not None else None,
                    "rel_from_v": rel_v.get("relativeFrom") if rel_v is not None else None,
                    "pos_offset": {"x": int(ph.text) if ph is not None else 0, "y": int(pv.text) if pv is not None else 0},
                    "extent": {"w": int(ext.get("cx")) if ext is not None else 0, "h": int(ext.get("cy")) if ext is not None else 0},
                    "wrap": _wrap(anchor),
                    "z": int(anchor.get("relativeHeight") or 0)
                }
            elif inline is not None:
                ext = inline.find("./wp:extent", NS)
                entry["anchor"] = {
                    "type": "inline",
                    "rel_from_h": None,
                    "rel_from_v": None,
                    "pos_offset": {"x": 0, "y": 0},
                    "extent": {"w": int(ext.get("cx")) if ext is not None else 0, "h": int(ext.get("cy")) if ext is not None else 0},
                    "wrap": None,
                    "z": 0
                }

            rids = _image_rids(drawing)
            if rids:
                entry["kind"] = entry.get("kind") or "image"
                image_payload = {"rIds": rids}
                crop = _image_crop(drawing)
                if crop:
                    image_payload["crop"] = crop
                entry["image"] = image_payload
                if inline_images_map:
                    for rid in rids:
                        record = inline_images_map.get(rid)
                        if record:
                            if doc_index not in record.doc_indices:
                                record.doc_indices.append(doc_index)
                            if record.doc_index is None:
                                record.doc_index = doc_index

            drawings.append(entry)
            d_idx_in_p += 1
        return drawings, d_idx_in_p


def _cell_text_from_matrix(matrix: List[List[Dict[str, Any]]], row: int, col: int) -> str:
    """
    Safely extracts stripped text from the sanitized table matrix.
    """
    if row < 0 or row >= len(matrix):
        return ""
    row_cells = matrix[row]
    if col < 0 or col >= len(row_cells):
        return ""
    cell = row_cells[col]
    text = ""
    if isinstance(cell, dict):
        text = cell.get("text") or ""
    else:
        text = str(cell or "")
    return text.strip()


def build_table_context_rows(
    matrix: List[List[Dict[str, Any]]],
    anchors: List[Dict[str, Any]],
    is_rowheader: bool,
    is_colheader: bool,
) -> List[List[Dict[str, Any]]]:
    """
    Builds table context rows by pairing headers with non-header cell values.

    Returns a list of lists, where each inner list represents a logical row of the table.
    """
    if not matrix:
        return []

    rows = len(matrix)
    cols = len(matrix[0]) if rows else 0
    if rows == 0 or cols == 0:
        return []

    data_row_start = 1 if is_rowheader else 0
    data_col_start = 1 if is_colheader else 0
    if data_row_start >= rows or data_col_start >= cols:
        return []

    row_headers: Dict[int, str] = {}
    if is_colheader:
        for r in range(data_row_start, rows):
            row_headers[r] = _cell_text_from_matrix(matrix, r, 0)

    col_headers: Dict[int, str] = {}
    if is_rowheader:
        for c in range(data_col_start, cols):
            col_headers[c] = _cell_text_from_matrix(matrix, 0, c)

    entries: List[Dict[str, Any]] = []
    for anchor in anchors or []:
        r = anchor.get("r")
        c = anchor.get("c")
        if r is None or c is None:
            continue
        if not isinstance(r, int) or not isinstance(c, int):
            continue
        if r >= rows or c >= cols:
            continue
        if anchor.get("vmerge") == "continue":
            continue
        if is_rowheader and r == 0:
            continue
        if is_colheader and c == 0:
            continue
        if r < data_row_start or c < data_col_start:
            continue

        cell_text = _cell_text_from_matrix(matrix, r, c)
        if not cell_text:
            continue

        header_parts: List[str] = []
        if is_colheader:
            row_header = row_headers.get(r, "")
            if row_header:
                header_parts.append(row_header)
        if is_rowheader:
            col_header = col_headers.get(c, "")
            if col_header:
                header_parts.append(col_header)

        header_text = " | ".join(part for part in header_parts if part).strip()
        entries.append(
            {
                "header": header_text,
                "headers": header_parts,
                "cell": cell_text,
                "row": r,
                "col": c,
            }
        )

    if not entries:
        return []

    # Group entries by their row index
    grouped_by_row: Dict[int, List[Dict[str, Any]]] = {}
    for entry in entries:
        row_idx = entry.get("row")
        if row_idx is not None:
            grouped_by_row.setdefault(row_idx, []).append(entry)

    # Sort by row index and return the list of lists
    sorted_rows: List[List[Dict[str, Any]]] = []
    for row_idx in sorted(grouped_by_row.keys()):
        sorted_rows.append(grouped_by_row[row_idx])

    return sorted_rows


def render_table_html(
    matrix: List[List[Dict[str, Any]]],
    anchors: List[Dict[str, Any]],
    is_rowheader: bool,
    is_colheader: bool,
) -> str:
    """
    Renders an HTML representation of the table, preserving basic header cues
    and merged cell spans inferred from the anchor metadata.
    """
    if not matrix:
        return "<table></table>"

    rows = len(matrix)
    cols = len(matrix[0]) if rows else 0
    anchor_map: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for anchor in anchors or []:
        r = anchor.get("r")
        c = anchor.get("c")
        if r is None or c is None:
            continue
        if not isinstance(r, int) or not isinstance(c, int):
            continue
        if anchor.get("vmerge") == "continue":
            continue
        anchor_map[(r, c)] = {
            "rowspan": max(1, int(anchor.get("rowspan", 1) or 1)),
            "colspan": max(1, int(anchor.get("colspan", 1) or 1)),
        }

    lines: List[str] = ["<table>"]
    for r in range(rows):
        lines.append("  <tr>")
        c = 0
        while c < cols:
            anchor_info = anchor_map.get((r, c))
            if anchor_info is None:
                c += 1
                continue

            cell = matrix[r][c]
            text = ""
            styled = False
            if isinstance(cell, dict):
                text = cell.get("text") or ""
                styled = bool(cell.get("styled"))
            else:
                text = str(cell or "")

            tag = "td"
            if (is_rowheader and r == 0) or (is_colheader and c == 0):
                tag = "th"

            attrs: List[str] = []
            rowspan = anchor_info.get("rowspan", 1)
            colspan = anchor_info.get("colspan", 1)
            if rowspan and rowspan > 1:
                attrs.append(f'rowspan="{rowspan}"')
            if colspan and colspan > 1:
                attrs.append(f'colspan="{colspan}"')
            if styled:
                attrs.append('style="background-color: #f2f2f2;"')

            attr_str = (" " + " ".join(attrs)) if attrs else ""
            lines.append(f'    <{tag}{attr_str}>{html.escape(text)}</{tag}>')
            c += max(1, colspan)
        lines.append("  </tr>")
    lines.append("</table>")
    return "\n".join(lines)
