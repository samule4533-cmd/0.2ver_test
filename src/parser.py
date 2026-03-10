import os
import re
import io
import json
import gc
import logging
import inspect
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import fitz  # PyMuPDF
from PIL import Image
import numpy as np
import easyocr

from google import genai
from google.genai import types
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from dotenv import load_dotenv

import cleaning
import field_extract

# =============================================================================
# Logging / Env
# =============================================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
load_dotenv()

# =============================================================================
# Paths
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = PROJECT_ROOT / "data" / "raw" / "sample_notices"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "parsing_result_notices"

DEFAULT_PDF_NAME = "공고문_샘플3.pdf"
SOURCE_PDF = INPUT_DIR / DEFAULT_PDF_NAME

# =============================================================================
# Config
# =============================================================================
TEXT_LEN_OCR_HINT = 120
MIN_GOOD_TEXT_LEN_AFTER_FAST = 140
MIN_GOOD_TEXT_LEN_AFTER_OCR = 140

GARBLED_MIN_LEN = 80
NEEDS_REVIEW_MIN_TEXT_LEN = 90
NEEDS_REVIEW_LOW_KO_RATIO = 0.03

GEMINI_MODEL_BASE = "gemini-2.5-flash"
GEMINI_MODEL_STRONG = os.getenv("GEMINI_MODEL_STRONG", "").strip()

GEMINI_MIN_TEXT_LEN = 140
GEMINI_FOR_TABLE_PAGES = True

RENDER_DPI = 280
ENABLE_OPENCV_PREPROCESS = True
IMPORTANT_FIELD_TRIGGERS = True

CLAUSE_KEYWORDS = [
    "입찰참가", "입찰 참가", "참가자격", "제출서류", "설명회", "현장설명",
    "입찰일시", "입찰장소", "제출기한", "제출일시", "접수", "문의",
    "담당자", "유의사항", "평가기준", "심사", "낙찰", "계약", "과업",
    "제안", "서류", "방문 제출",
]

SUPPLEMENTARY_KEYWORDS = [
    "청렴", "서약서", "확인서", "동의서", "서약", "서명", "날인",
]

DATE_PATTERNS = [
    r"\d{4}[./-]\d{1,2}[./-]\d{1,2}",
    r"\d{1,2}\s*월\s*\d{1,2}\s*일",
]

MONEY_PATTERNS = [
    r"(\d{1,3}(,\d{3})+|\d+)\s*원",
    r"(\d{1,3}(,\d{3})+|\d+)\s*만원",
]

DEADLINE_KEYWORDS = ["마감", "기한", "제출", "접수", "입찰", "개찰", "제안서"]

# =============================================================================
# Text Quality Helpers
# =============================================================================
def korean_ratio(s: str) -> float:
    if not s:
        return 0.0
    return len(re.findall(r"[가-힣]", s)) / max(len(s), 1)


def cjk_ratio(s: str) -> float:
    if not s:
        return 0.0
    return len(re.findall(r"[\u4e00-\u9fff]", s)) / max(len(s), 1)


def digit_ratio(s: str) -> float:
    if not s:
        return 0.0
    return len(re.findall(r"[0-9]", s)) / max(len(s), 1)


def symbol_ratio(s: str) -> float:
    if not s:
        return 0.0
    return len(re.findall(r"[|_=\-~`^•·…⋯◇◆■□○●※#@/\\]", s)) / max(len(s), 1)


def repeated_gibberish_score(s: str) -> float:
    if not s:
        return 0.0
    longest, run = 1, 1
    for i in range(1, len(s)):
        if s[i] == s[i - 1]:
            run += 1
            longest = max(longest, run)
        else:
            run = 1
    return longest / max(len(s), 1)


def looks_garbled_korean(s: str) -> bool:
    s2 = (s or "").strip()
    if len(s2) < GARBLED_MIN_LEN:
        return False

    ko = korean_ratio(s2)
    cjk = cjk_ratio(s2)
    sym = symbol_ratio(s2)
    rep = repeated_gibberish_score(s2)

    if s2.count("(") != s2.count(")") or s2.count("[") != s2.count("]"):
        return True

    if rep >= 0.10 and sym >= 0.06:
        return True

    if ko < 0.05 and (cjk > 0.10 or sym > 0.08):
        return True

    if ko < 0.10 and sym > 0.07 and cjk > 0.05:
        return True

    if len(re.findall(r"\(\s*\)", s2)) >= 2:
        return True

    if len(re.findall(r"\s{3,}", s2)) >= 4:
        return True

    return False


def evaluate_text_quality(text: str) -> Dict[str, Any]:
    t = (text or "").strip()
    return {
        "text_len": len(t),
        "korean_ratio": round(korean_ratio(t), 4),
        "cjk_ratio": round(cjk_ratio(t), 4),
        "digit_ratio": round(digit_ratio(t), 4),
        "symbol_ratio": round(symbol_ratio(t), 4),
        "repeat_score": round(repeated_gibberish_score(t), 4),
        "garbled": looks_garbled_korean(t),
    }


def important_fields_missing(text: str) -> bool:
    if not IMPORTANT_FIELD_TRIGGERS:
        return False

    t = (text or "").strip()
    if len(t) < 120:
        return False

    has_deadline_kw = any(k in t for k in DEADLINE_KEYWORDS)
    has_date = any(re.search(p, t) for p in DATE_PATTERNS)
    has_money = any(re.search(p, t) for p in MONEY_PATTERNS)

    if has_deadline_kw and not has_date:
        return True

    if ("예산" in t or "금액" in t or "원" in t) and not has_money:
        return True

    return False


# =============================================================================
# Page Type Heuristics
# =============================================================================
def count_clause_hints(text: str) -> int:
    t = (text or "").strip()
    if not t:
        return 0

    count = 0
    count += len(re.findall(r"(?m)^\s*[가-하]\s*[.)．]", t))
    count += len(re.findall(r"(?m)^\s*\d+\s*[.)．]", t))
    count += len(re.findall(r"(?m)^\s*[①-⑳]", t))
    count += sum(1 for kw in CLAUSE_KEYWORDS if kw in t)
    return count


def is_clause_like_page(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False

    hint_count = count_clause_hints(t)
    if hint_count >= 4:
        return True

    if sum(1 for kw in CLAUSE_KEYWORDS if kw in t) >= 2 and len(t) > 100:
        return True

    return False


def is_supplementary_like_page(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    return sum(1 for kw in SUPPLEMENTARY_KEYWORDS if kw in t) >= 2


# =============================================================================
# Preflight
# =============================================================================
def preflight_check(output_dir: Path) -> Dict[str, Any]:
    report = {"ok": True, "checks": []}

    def _add(name: str, ok: bool, msg: str = ""):
        report["checks"].append({"name": name, "ok": ok, "msg": msg})
        if not ok:
            report["ok"] = False

    _add("pdf_exists", SOURCE_PDF.exists(), str(SOURCE_PDF))

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        testfile = output_dir / "__write_test__.tmp"
        testfile.write_text("ok", encoding="utf-8")
        testfile.unlink(missing_ok=True)
        _add("output_writable", True, str(output_dir))
    except Exception as e:
        _add("output_writable", False, f"{output_dir} / {e}")

    try:
        _ = get_easyocr_reader()
        _add("easyocr_reader", True, "loaded")
    except Exception as e:
        _add("easyocr_reader", False, str(e))

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    _add("gemini_api_key_present", bool(api_key), "env: GEMINI_API_KEY")

    if ENABLE_OPENCV_PREPROCESS:
        try:
            import cv2  # noqa
            _add("opencv_available", True, "cv2 import ok")
        except Exception as e:
            _add("opencv_available", False, f"cv2 not available: {e}")

    return report


# =============================================================================
# EasyOCR / Render
# =============================================================================
_easyocr_reader = None


def get_easyocr_reader():
    global _easyocr_reader
    if _easyocr_reader is None:
        logging.info("🔧 EasyOCR Reader 로딩 중...")
        _easyocr_reader = easyocr.Reader(["ko", "en"], gpu=False)
    return _easyocr_reader


def preprocess_for_table_ocr(pil_img: Image.Image) -> Image.Image:
    if not ENABLE_OPENCV_PREPROCESS:
        return pil_img

    try:
        import cv2

        img = np.array(pil_img.convert("RGB"))
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        thr = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            25,
            15,
        )
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))

        h_lines = cv2.morphologyEx(thr, cv2.MORPH_OPEN, h_kernel, iterations=1)
        v_lines = cv2.morphologyEx(thr, cv2.MORPH_OPEN, v_kernel, iterations=1)
        lines = cv2.bitwise_or(h_lines, v_lines)

        cleaned_img = cv2.bitwise_and(thr, cv2.bitwise_not(lines))
        cleaned_img = cv2.bitwise_not(cleaned_img)
        cleaned_img = cv2.GaussianBlur(cleaned_img, (3, 3), 0)

        return Image.fromarray(cleaned_img).convert("RGB")

    except Exception as e:
        logging.warning(f"⚠️ OpenCV 전처리 실패: {e}")
        return pil_img


def ocr_image_text_easyocr(pil_img: Image.Image, table_mode: bool = False) -> Tuple[str, Dict[str, Any]]:
    meta = {"engine": "easyocr", "errors": [], "table_mode": table_mode}
    try:
        reader = get_easyocr_reader()
        img = preprocess_for_table_ocr(pil_img) if table_mode else pil_img
        lines = reader.readtext(np.array(img.convert("RGB")), detail=0, paragraph=True)
        text = "\n".join([str(x).strip() for x in lines if str(x).strip()]).strip()
        return text, meta
    except Exception as e:
        meta["errors"].append(f"easyocr_failed: {e}")
        return "", meta


def render_page_to_pil(fitz_doc, page_index0: int, dpi: int = RENDER_DPI) -> Image.Image:
    page = fitz_doc.load_page(page_index0)
    mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)


# =============================================================================
# Gemini
# =============================================================================
def gemini_page_vision(pil_img: Image.Image, model_name: str) -> Dict[str, Any]:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY 환경변수가 없습니다.")

    client = genai.Client(api_key=api_key)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")

    prompt = """
너는 한국어 공고문/제안서 문서 이미지 분석기다.
이미지에서 보이는 내용만 정확히 읽어라.
특히 아래를 잘 보존하라:
- '가.', '나.', '다.' 같은 조항형 목록
- '1)', '2)' 같은 번호 목록
- 날짜, 금액, 기관명, 장소, 문의처
- 표 안의 짧은 문구와 제출서류 목록

확신이 없으면 절대 추정하지 말고 '?' 또는 'UNKNOWN'으로 표시하라.

반드시 아래 JSON만 출력:
{
  "key_values": { "항목": "값" },
  "ocr_text": "",
  "summary": ""
}
"""

    resp = client.models.generate_content(
        model=model_name,
        contents=[prompt, types.Part.from_bytes(data=buf.getvalue(), mime_type="image/png")],
        config=types.GenerateContentConfig(
            temperature=0.0,
            response_mime_type="application/json",
        ),
    )

    try:
        return json.loads(resp.text)
    except json.JSONDecodeError as e:
        logging.warning(f"⚠️ Gemini JSON 파싱 실패, 복구 시도: {e}")
        match = re.search(r"\{.*\}", resp.text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
        return {"key_values": {}, "ocr_text": resp.text, "summary": "JSON 파싱 실패"}


def should_call_gemini(page_has_table_hint: bool, text: str, quality: Dict[str, Any], clause_like: bool) -> bool:
    t = (text or "").strip()

    if clause_like:
        return True

    if len(t) < GEMINI_MIN_TEXT_LEN:
        if page_has_table_hint or quality.get("garbled"):
            return True

    if quality.get("garbled"):
        return True

    if GEMINI_FOR_TABLE_PAGES and page_has_table_hint and (
        quality.get("korean_ratio", 0) < 0.10 or quality.get("symbol_ratio", 0) > 0.08
    ):
        return True

    if important_fields_missing(t):
        return True

    return False


def choose_gemini_model(page_has_table_hint: bool, text: str, quality: Dict[str, Any], clause_like: bool) -> str:
    if not GEMINI_MODEL_STRONG:
        return GEMINI_MODEL_BASE

    if clause_like:
        return GEMINI_MODEL_STRONG

    if quality.get("garbled") and quality.get("korean_ratio", 0) < 0.08:
        return GEMINI_MODEL_STRONG

    if important_fields_missing(text):
        return GEMINI_MODEL_STRONG

    if page_has_table_hint and (
        quality.get("symbol_ratio", 0) > 0.10 or quality.get("repeat_score", 0) > 0.10
    ):
        return GEMINI_MODEL_STRONG

    return GEMINI_MODEL_BASE


def render_ocr_page_text(
    fitz_doc,
    page_index0: int,
    page_has_table_hint: bool,
    clause_like: bool,
    supplementary_like: bool,
    dpi: int = RENDER_DPI,
    try_gemini: bool = True,
) -> Tuple[str, Dict[str, Any], Optional[Dict[str, Any]]]:
    pil_img = render_page_to_pil(fitz_doc, page_index0, dpi=dpi)

    text, meta = ocr_image_text_easyocr(pil_img, table_mode=page_has_table_hint)
    q = evaluate_text_quality(text)
    gem = None

    if try_gemini and os.getenv("GEMINI_API_KEY", "").strip():
        if should_call_gemini(page_has_table_hint, text, q, clause_like=clause_like):
            model = choose_gemini_model(page_has_table_hint, text, q, clause_like=clause_like)
            try:
                logging.info(
                    f"✨ Gemini Vision 보강 (page {page_index0 + 1}, model={model}, clause_like={clause_like})"
                )
                gem = gemini_page_vision(pil_img, model_name=model)
                gtext = (gem.get("ocr_text") or "").strip()

                should_replace = False

                if clause_like and gtext:
                    should_replace = True
                elif gtext and (
                    len(gtext) > len(text)
                    or q.get("garbled")
                    or important_fields_missing(text)
                ):
                    should_replace = True

                if supplementary_like and gtext:
                    gq = evaluate_text_quality(gtext)
                    if gq["garbled"] and gq["korean_ratio"] < 0.08:
                        should_replace = False

                if should_replace:
                    text = gtext
                    meta = {
                        "engine": model,
                        "errors": [],
                        "table_mode": page_has_table_hint,
                        "fallback_from": "easyocr",
                    }

            except Exception as e:
                meta["errors"].append(f"gemini_failed: {e}")

    return text.strip(), meta, gem


# =============================================================================
# Docling
# =============================================================================
def build_converters() -> Tuple[DocumentConverter, DocumentConverter]:
    f_opt = PdfPipelineOptions()
    f_opt.do_table_structure = True
    f_opt.do_ocr = False
    f_opt.generate_picture_images = True

    o_opt = PdfPipelineOptions()
    o_opt.do_table_structure = True
    o_opt.do_ocr = True
    o_opt.generate_picture_images = True

    fast = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=f_opt)})
    ocr = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=o_opt)})
    return fast, ocr


def export_markdown_compat(doc) -> str:
    fn = getattr(doc, "export_to_markdown", None)
    if not callable(fn):
        return ""
    try:
        sig = inspect.signature(fn)
        if "image_dir" in sig.parameters:
            return fn(image_dir=None) or ""
        return fn() or ""
    except Exception:
        return ""


def extract_texts_from_docdict(doc_dict: Any) -> List[str]:
    texts: List[str] = []

    def _walk(x: Any):
        if isinstance(x, dict):
            for k in ("text", "content", "value"):
                v = x.get(k)
                if isinstance(v, str) and v.strip():
                    texts.append(v.strip())
            if str(x.get("type", "")).lower() in ("paragraph", "heading", "title"):
                v = x.get("text")
                if isinstance(v, str) and v.strip():
                    texts.append(v.strip())
            for v in x.values():
                _walk(v)
        elif isinstance(x, list):
            for it in x:
                _walk(it)

    _walk(doc_dict)

    uniq, seen = [], set()
    for s in texts:
        key = s[:200]
        if key not in seen:
            uniq.append(s)
            seen.add(key)
    return uniq[:4000]


def write_temp_page_pdf(fitz_doc, page_index0: int, out_path: Path):
    temp = fitz.open()
    temp.insert_pdf(fitz_doc, from_page=page_index0, to_page=page_index0)
    temp.save(str(out_path), garbage=4, deflate=True)
    temp.close()

    if (not out_path.exists()) or out_path.stat().st_size == 0:
        raise RuntimeError("temp pdf 생성 실패(0 bytes)")


def run_docling_convert(
    converter: DocumentConverter, pdf_path: Path
) -> Tuple[Optional[Any], Optional[Dict[str, Any]], Optional[str]]:
    try:
        d = converter.convert(str(pdf_path)).document
        return d, d.export_to_dict(), None
    except Exception as e:
        return None, None, str(e)


# =============================================================================
# Table Helpers
# =============================================================================
def find_tables_in_docdict(obj: Any) -> List[Dict[str, Any]]:
    found: List[Dict[str, Any]] = []

    def _walk(x: Any):
        if isinstance(x, dict):
            if str(x.get("type", "")).lower() == "table":
                found.append(x)
            if "tables" in x and isinstance(x["tables"], list):
                for t in x["tables"]:
                    if isinstance(t, dict):
                        found.append(t)
            for v in x.values():
                _walk(v)
        elif isinstance(x, list):
            for it in x:
                _walk(it)

    _walk(obj)

    unique, seen = [], set()
    for t in found:
        if id(t) not in seen:
            unique.append(t)
            seen.add(id(t))
    return unique


def _to_matrix_from_table(tbl: Dict[str, Any]) -> Optional[List[List[str]]]:
    for key in ("rows", "data"):
        if isinstance(tbl.get(key), list) and tbl.get(key) and all(isinstance(r, list) for r in tbl[key]):
            return [[str(c).strip() if c is not None else "" for c in r] for r in tbl[key]]

    cells = tbl.get("cells")
    if isinstance(cells, list) and cells and all(isinstance(c, dict) for c in cells):
        max_r = max((int(c.get("row", 0) or 0) for c in cells), default=0)
        max_c = max((int(c.get("col", 0) or 0) for c in cells), default=0)
        m = [["" for _ in range(max_c + 1)] for _ in range(max_r + 1)]
        for c in cells:
            r = int(c.get("row", 0) or 0)
            col = int(c.get("col", 0) or 0)
            txt = c.get("text") or c.get("content") or ""
            m[r][col] = str(txt).replace("\n", " ").strip()
        return m

    return None


def normalize_table_to_kv_or_rows(matrix: List[List[str]]) -> Dict[str, Any]:
    m = [[c.strip() for c in r] for r in matrix if any((c or "").strip() for c in r)]
    if not m:
        return {"kind": "empty", "pairs": {}, "rows": [], "row_sentences": []}

    two_col_ratio = sum(1 for r in m if len(r) >= 2) / max(len(m), 1)
    if two_col_ratio >= 0.7:
        pairs: Dict[str, str] = {}
        for r in m:
            if len(r) < 2:
                continue
            k = r[0].strip()
            v = " ".join([x.strip() for x in r[1:] if x.strip()]).strip()
            if not k or not v:
                continue
            if len(k) > 60:
                continue
            pairs[k] = v

        if len(pairs) >= 2:
            return {
                "kind": "key_value",
                "pairs": pairs,
                "rows": m,
                "row_sentences": [f"{k}: {v}" for k, v in pairs.items()],
            }

    return {
        "kind": "rows",
        "pairs": {},
        "rows": m,
        "row_sentences": [" / ".join([c for c in r if c]) for r in m[:200]],
    }


def normalize_tables(doc_dict: Dict[str, Any], page_num: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for idx, tbl in enumerate(find_tables_in_docdict(doc_dict), start=1):
        matrix = _to_matrix_from_table(tbl)
        if not matrix:
            continue
        out.append({"page": page_num, "table_index": idx, **normalize_table_to_kv_or_rows(matrix)})
    return out


# =============================================================================
# Block Extraction
# =============================================================================
def classify_line_type(line: str, font_size: float, max_font_size: float) -> str:
    s = (line or "").strip()
    if not s:
        return "empty"

    if font_size >= max_font_size * 0.95 and len(s) <= 60:
        return "heading"

    if re.match(r"^\s*[가-하]\s*[.)．]", s):
        return "list"
    if re.match(r"^\s*\d+\s*[.)．]", s):
        return "list"
    if re.match(r"^\s*[①-⑳]", s):
        return "list"
    if re.match(r"^\s*[-•●]", s):
        return "list"

    return "paragraph"


def extract_blocks_from_fitz_page(fitz_page, page_num: int) -> List[Dict[str, Any]]:
    """
    PDF 내부 텍스트 레이아웃이 살아있을 때 가장 먼저 쓰는 블록 추출.
    """
    blocks: List[Dict[str, Any]] = []
    page_dict = fitz_page.get_text("dict")
    raw_blocks = page_dict.get("blocks", [])

    all_font_sizes = []
    for blk in raw_blocks:
        if blk.get("type") != 0:
            continue
        for line in blk.get("lines", []):
            for span in line.get("spans", []):
                size = float(span.get("size", 0) or 0)
                if size > 0:
                    all_font_sizes.append(size)

    max_font_size = max(all_font_sizes) if all_font_sizes else 12.0

    order = 0
    for blk in raw_blocks:
        blk_type = blk.get("type")

        # text block
        if blk_type == 0:
            lines_out = []
            font_sizes = []

            for line in blk.get("lines", []):
                spans = line.get("spans", [])
                parts = []
                local_sizes = []
                for sp in spans:
                    txt = str(sp.get("text", "") or "").strip()
                    if txt:
                        parts.append(txt)
                        local_sizes.append(float(sp.get("size", 0) or 0))

                line_text = " ".join(parts).strip()
                if line_text:
                    lines_out.append(line_text)
                if local_sizes:
                    font_sizes.append(max(local_sizes))

            text = "\n".join(lines_out).strip()
            if not text:
                continue

            avg_font = sum(font_sizes) / len(font_sizes) if font_sizes else 11.0
            block_kind = classify_line_type(text.split("\n")[0], avg_font, max_font_size)

            blocks.append(
                {
                    "block_id": f"page_{page_num}_block_{order}",
                    "page_number": page_num,
                    "type": block_kind,
                    "bbox": blk.get("bbox"),
                    "text": text,
                    "font_size": round(avg_font, 2),
                    "source": "fitz_layout",
                }
            )
            order += 1

        # image block
        elif blk_type == 1:
            blocks.append(
                {
                    "block_id": f"page_{page_num}_block_{order}",
                    "page_number": page_num,
                    "type": "image",
                    "bbox": blk.get("bbox"),
                    "text": "",
                    "source": "fitz_layout",
                }
            )
            order += 1

    return blocks


def fallback_blocks_from_text(text: str, page_num: int) -> List[Dict[str, Any]]:
    """
    레이아웃 정보가 없거나 깨졌을 때 텍스트를 문단/리스트 중심으로 다시 블록화.
    """
    blocks: List[Dict[str, Any]] = []
    lines = [x.rstrip() for x in (text or "").split("\n")]

    current: List[str] = []
    current_type = None
    order = 0

    def flush():
        nonlocal current, current_type, order
        if not current:
            return
        joined = "\n".join([x for x in current if x.strip()]).strip()
        if joined:
            blocks.append(
                {
                    "block_id": f"page_{page_num}_block_{order}",
                    "page_number": page_num,
                    "type": current_type or "paragraph",
                    "bbox": None,
                    "text": joined,
                    "source": "text_fallback",
                }
            )
            order += 1
        current = []
        current_type = None

    for line in lines:
        s = line.strip()

        if not s:
            flush()
            continue

        if re.match(r"^\s*[가-하]\s*[.)．]", s) or re.match(r"^\s*\d+\s*[.)．]", s) or re.match(r"^\s*[①-⑳]", s):
            if current_type not in (None, "list"):
                flush()
            current_type = "list"
            current.append(s)
            continue

        if len(s) <= 40 and not current:
            current_type = "heading"
            current.append(s)
            flush()
            continue

        if current_type not in (None, "paragraph"):
            flush()

        current_type = "paragraph"
        current.append(s)

    flush()
    return blocks


def attach_table_blocks(page_num: int, blocks: List[Dict[str, Any]], tables_normalized: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = list(blocks)
    base = len(out)
    for i, t in enumerate(tables_normalized, start=1):
        preview = "\n".join(t.get("row_sentences", [])[:20]).strip()
        out.append(
            {
                "block_id": f"page_{page_num}_block_{base + i}",
                "page_number": page_num,
                "type": "table",
                "bbox": None,
                "text": preview,
                "table_kind": t.get("kind"),
                "table_index": t.get("table_index"),
                "source": "normalized_table",
            }
        )
    return out


def build_page_blocks(fitz_page, page_num: int, final_text: str, tables_normalized: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    fitz_blocks = extract_blocks_from_fitz_page(fitz_page, page_num)

    if fitz_blocks:
        blocks = fitz_blocks
    else:
        blocks = fallback_blocks_from_text(final_text, page_num)

    blocks = attach_table_blocks(page_num, blocks, tables_normalized)
    return blocks


# =============================================================================
# Pipeline
# =============================================================================
def decide_needs_review(
    final_text: str,
    images_saved: int,
    tables_found: int,
    errors: List[str],
    warnings: List[str],
) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    t = (final_text or "").strip()

    if errors:
        reasons.append("has_errors")
    if len(t) < NEEDS_REVIEW_MIN_TEXT_LEN:
        reasons.append("too_short_text")
    if len(t) > 50 and korean_ratio(t) < NEEDS_REVIEW_LOW_KO_RATIO:
        reasons.append("low_korean_ratio")
    if tables_found == 0 and images_saved == 0 and "table_hint" in " ".join(warnings).lower():
        reasons.append("table_hint_but_no_tables_or_images")
    if looks_garbled_korean(t):
        reasons.append("garbled_text")
    if important_fields_missing(t):
        reasons.append("important_fields_missing")

    return (len(reasons) > 0), reasons


def convert_page(
    fast_converter: DocumentConverter,
    ocr_converter: DocumentConverter,
    fitz_doc,
    fitz_page,
    page_index0: int,
    temp_pdf_path: Path,
):
    warnings: List[str] = []

    fitz_text = (fitz_page.get_text("text") or "").strip()
    clause_like = is_clause_like_page(fitz_text)
    supplementary_like = is_supplementary_like_page(fitz_text)

    page_has_table_hint = (
        len(fitz_page.get_images(full=True)) > 0
        or len(fitz_text) <= TEXT_LEN_OCR_HINT
        or looks_garbled_korean(fitz_text)
        or clause_like
    )

    if page_has_table_hint:
        warnings.append("table_hint")
    if clause_like:
        warnings.append("clause_like_page")
    if supplementary_like:
        warnings.append("supplementary_like_page")

    def quality_from_dict(dd: Dict[str, Any]) -> Dict[str, Any]:
        return evaluate_text_quality("\n".join(extract_texts_from_docdict(dd)[:120]).strip())

    # 1) Docling fast
    doc, dd, err = run_docling_convert(fast_converter, temp_pdf_path)
    if err is None and dd is not None:
        q = quality_from_dict(dd)
        if clause_like:
            warnings.append("fast_ok_but_clause_like -> compare_render_ocr")
        elif q["text_len"] >= MIN_GOOD_TEXT_LEN_AFTER_FAST and not q["garbled"]:
            return doc, {"docling_dict": dd, "render_ocr_text": None, "gemini": None}, "fast", warnings, q
        warnings.append("fast_poor_or_garbled -> try_docling_ocr")
    else:
        warnings.append(f"fast_failed: {err}")

    # 2) Docling OCR
    doc2, dd2, err2 = run_docling_convert(ocr_converter, temp_pdf_path)
    if err2 is None and dd2 is not None:
        q2 = quality_from_dict(dd2)
        if clause_like:
            warnings.append("ocr_ok_but_clause_like -> compare_render_ocr")
        elif q2["text_len"] >= MIN_GOOD_TEXT_LEN_AFTER_OCR and not q2["garbled"]:
            return doc2, {"docling_dict": dd2, "render_ocr_text": None, "gemini": None}, "ocr", warnings, q2
        warnings.append("docling_ocr_poor_or_garbled -> render_ocr")
    else:
        warnings.append(f"docling_ocr_failed: {err2}")

    # 3) Render OCR + Gemini
    text, meta, gem = render_ocr_page_text(
        fitz_doc,
        page_index0,
        page_has_table_hint=page_has_table_hint,
        clause_like=clause_like,
        supplementary_like=supplementary_like,
        dpi=RENDER_DPI,
        try_gemini=True,
    )
    q3 = evaluate_text_quality(text)
    q3["render_engine"] = meta.get("engine")
    q3["clause_like"] = clause_like
    q3["supplementary_like"] = supplementary_like

    return None, {"docling_dict": {}, "render_ocr_text": text, "gemini": gem}, "render_ocr", warnings, q3


def build_markdown_from_text(page_num: int, title: str, text: str) -> str:
    text = (text or "").strip()
    if not text:
        return f"### [{title}] Page {page_num}\n\n> (텍스트 없음)\n"
    return f"### [{title}] Page {page_num}\n\n{text}\n"


# =============================================================================
# Main
# =============================================================================
def main():
    target_out_dir = OUTPUT_ROOT / SOURCE_PDF.stem
    parse_report_path = target_out_dir / "parse_report.json"
    final_md_path = target_out_dir / f"{SOURCE_PDF.stem}_최종_통합본.md"
    final_json_path = target_out_dir / f"{SOURCE_PDF.stem}_최종_통합본.json"
    preflight_path = target_out_dir / "preflight.json"
    fields_path = target_out_dir / "fields.json"

    pf = preflight_check(target_out_dir)
    target_out_dir.mkdir(parents=True, exist_ok=True)

    with open(preflight_path, "w", encoding="utf-8") as f:
        json.dump(pf, f, ensure_ascii=False, indent=2)

    if not pf["ok"]:
        logging.error("❌ Pre-flight check 실패. preflight.json 확인 후 수정하세요.")
        for c in pf["checks"]:
            if not c["ok"]:
                logging.error(f"- {c['name']}: {c['msg']}")
        return

    fast_converter, ocr_converter = build_converters()
    fitz_doc = fitz.open(str(SOURCE_PDF))
    total_pages = fitz_doc.page_count
    logging.info(f"🚀 총 {total_pages}페이지 파싱 시작 -> {target_out_dir}")

    parse_report: List[Dict[str, Any]] = []
    final_pages: List[Dict[str, Any]] = []

    for page_num in range(1, total_pages + 1):
        page_index0 = page_num - 1
        fitz_page = fitz_doc.load_page(page_index0)

        warnings: List[str] = []
        errors: List[str] = []

        temp_pdf_path = target_out_dir / f"temp_page_{page_num}.pdf"

        try:
            write_temp_page_pdf(fitz_doc, page_index0, temp_pdf_path)
        except Exception as e:
            errors.append(f"temp_pdf_create_failed: {e}")
            logging.error(f"[{page_num}] temp pdf 생성 실패: {e}")
            continue

        doc = None
        images_saved = 0
        tables_found = 0
        md_reason = "unknown"

        page_payload: Dict[str, Any] = {
            "page_number": page_num,
            "engine_used": None,
            "quality": {},
            "md_reason": None,
            "final_text": "",
            "cleaned_text": "",
            "cleaned_for_fields": "",
            "cleaned_render_ocr_text": "",
            "cleaned_gemini_key_values": {},
            "docling_dict": {},
            "render_ocr_text": None,
            "gemini_page_vision": None,
            "gemini_key_values": {},
            "tables_normalized": [],
            "blocks": [],
            "images": [],
            "image_ocr": [],
            "needs_review": False,
            "needs_review_reasons": [],
            "warnings": [],
            "errors": [],
        }

        try:
            doc, conv_payload, engine_used, w, quality = convert_page(
                fast_converter, ocr_converter, fitz_doc, fitz_page, page_index0, temp_pdf_path
            )
            warnings.extend(w)

            page_payload["engine_used"] = engine_used
            page_payload["quality"] = quality
            page_payload["docling_dict"] = conv_payload.get("docling_dict", {}) or {}
            page_payload["render_ocr_text"] = conv_payload.get("render_ocr_text")
            page_payload["gemini_page_vision"] = conv_payload.get("gemini")
            page_payload["gemini_key_values"] = (
                (conv_payload.get("gemini") or {}).get("key_values", {}) if conv_payload.get("gemini") else {}
            )

            if page_payload["docling_dict"]:
                nt = normalize_tables(page_payload["docling_dict"], page_num)
                page_payload["tables_normalized"] = nt
                tables_found = len(nt)

            final_text = ""
            clause_like = bool(quality.get("clause_like", False))

            if clause_like and page_payload["render_ocr_text"]:
                final_text = page_payload["render_ocr_text"]
                md_reason = f"clause_priority_render(engine={quality.get('render_engine')})"

            if not final_text and doc is not None:
                native_md = export_markdown_compat(doc) or ""
                if len(native_md.strip()) > 50 and not looks_garbled_korean(native_md):
                    final_text = native_md.strip()
                    md_reason = "docling_native_md"
                else:
                    warnings.append("docling_md_poor -> fallback")

            if not final_text:
                if page_payload["render_ocr_text"]:
                    final_text = page_payload["render_ocr_text"]
                    md_reason = f"render_ocr(engine={quality.get('render_engine')})"
                else:
                    final_text = (fitz_page.get_text("text") or "").strip()
                    md_reason = "fitz_text_fallback"

            page_payload["final_text"] = final_text
            page_payload["md_reason"] = md_reason

            page_payload = cleaning.clean_page_payload(page_payload)

            # 블록 구성
            page_payload["blocks"] = build_page_blocks(
                fitz_page=fitz_page,
                page_num=page_num,
                final_text=page_payload.get("cleaned_text") or final_text,
                tables_normalized=page_payload.get("tables_normalized", []),
            )

            # 이미지 추출
            if doc is not None:
                img_count = 1
                for item, _level in doc.iterate_items():
                    try:
                        if str(getattr(item, "label", "")).lower() == "picture":
                            img = item.get_image(doc)
                            if img is None:
                                continue
                            img_path = target_out_dir / f"page_{page_num}_img_{img_count}.png"
                            img.save(img_path)
                            images_saved += 1
                            page_payload["images"].append(img_path.name)

                            txt, meta = ocr_image_text_easyocr(Image.open(img_path).convert("RGB"), table_mode=False)
                            page_payload["image_ocr"].append(
                                {"page": page_num, "file": img_path.name, "ocr_text": txt, "ocr_meta": meta}
                            )
                            img_count += 1
                    except Exception as e:
                        warnings.append(f"image_extract_or_ocr_failed: {e}")

            needs_review, reasons = decide_needs_review(final_text, images_saved, tables_found, errors, warnings)
            page_payload["needs_review"] = needs_review
            page_payload["needs_review_reasons"] = reasons

            page_md = build_markdown_from_text(page_num, md_reason, final_text)

            if page_payload["blocks"]:
                page_md += "\n\n### 🧱 Blocks Preview\n"
                for b in page_payload["blocks"][:12]:
                    preview = (b.get("text") or "").replace("\n", " ").strip()
                    if len(preview) > 120:
                        preview = preview[:120] + "..."
                    page_md += f"- [{b.get('type')}] {preview}\n"

            if page_payload["gemini_page_vision"]:
                page_md += "\n\n### 🤖 Gemini 보강(원문 기반)\n```json\n"
                page_md += json.dumps(page_payload["gemini_page_vision"], ensure_ascii=False, indent=2)
                page_md += "\n```\n"

            if page_payload["tables_normalized"]:
                page_md += "\n\n### 📊 추출/정규화된 표(요약)\n"
                for t in page_payload["tables_normalized"][:8]:
                    page_md += f"- (table {t['table_index']}, kind={t['kind']}) rows={len(t.get('rows', []))}\n"

            if page_payload["needs_review"]:
                page_md += f"\n\n> ⚠ needs_review: {page_payload['needs_review_reasons']}\n"

            (target_out_dir / f"page_{page_num}.md").write_text(page_md, encoding="utf-8")
            with open(target_out_dir / f"page_{page_num}.json", "w", encoding="utf-8") as jf:
                json.dump(page_payload, jf, ensure_ascii=False, indent=2)

        except Exception as e:
            errors.append(str(e))
            page_payload["errors"] = errors
            logging.error(f"[{page_num}] 처리 실패: {e}")

        finally:
            page_payload["warnings"] = warnings
            page_payload["errors"] = errors

            if temp_pdf_path.exists():
                try:
                    os.remove(temp_pdf_path)
                except Exception:
                    pass

            try:
                del doc
            except Exception:
                pass
            gc.collect()

        parse_report.append(
            {
                "page": page_num,
                "engine_used": page_payload.get("engine_used"),
                "quality": page_payload.get("quality"),
                "images_saved": images_saved,
                "tables_found": tables_found,
                "md_reason": md_reason,
                "needs_review": page_payload.get("needs_review", False),
                "needs_review_reasons": page_payload.get("needs_review_reasons", []),
                "warnings": warnings,
                "errors": errors,
            }
        )
        final_pages.append(page_payload)

    fitz_doc.close()

    with open(parse_report_path, "w", encoding="utf-8") as f:
        json.dump(parse_report, f, ensure_ascii=False, indent=2)

    with open(final_md_path, "w", encoding="utf-8") as out_md:
        out_md.write(f"# {SOURCE_PDF.name} - 통합 파싱 결과\n\n")
        out_md.write(f"- 총 페이지: {total_pages}\n")
        out_md.write(f"- 출력 폴더: {target_out_dir}\n\n")

        for p in final_pages:
            pnum = p["page_number"]
            out_md.write(f"## --- Page {pnum} ---\n\n")
            out_md.write(build_markdown_from_text(pnum, p.get("md_reason", "unknown"), p.get("final_text", "")))

            if p.get("tables_normalized"):
                out_md.write("\n### 📌 Tables (row_sentences preview)\n")
                for t in p["tables_normalized"][:4]:
                    for line in t.get("row_sentences", [])[:10]:
                        out_md.write(f"- {line}\n")
                out_md.write("\n")

            if p.get("gemini_key_values"):
                out_md.write("\n### 🔑 Gemini key_values\n```json\n")
                out_md.write(json.dumps(p["gemini_key_values"], ensure_ascii=False, indent=2))
                out_md.write("\n```\n")

            if p.get("needs_review"):
                out_md.write(f"\n> ⚠ needs_review: {p.get('needs_review_reasons')}\n\n")

    with open(final_json_path, "w", encoding="utf-8") as jf:
        json.dump(final_pages, jf, ensure_ascii=False, indent=2)

    try:
        fields = {
            "bid_amount": field_extract.extract_bid_amount_from_final_pages(final_pages),
            # 나중에 여기 계속 추가:
            # "notice_no": field_extract.extract_notice_no_from_final_pages(final_pages),
            # "due_date": field_extract.extract_due_date_from_final_pages(final_pages),
            # "agency": field_extract.extract_agency_from_final_pages(final_pages),
        }

        with open(fields_path, "w", encoding="utf-8") as f:
            json.dump(fields, f, ensure_ascii=False, indent=2)

        logging.info(f"✅ fields.json 저장 완료: {fields_path}")

    except Exception as e:
        logging.warning(f"⚠️ fields.json 생성 실패: {e}")

    logging.info("🎉 완료! (Docling + EasyOCR + Gemini + Blocks + Cleaning + Field Extract)")


if __name__ == "__main__":
    main()