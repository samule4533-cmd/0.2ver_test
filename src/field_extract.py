from dataclasses import dataclass
import re
from typing import Any, Dict, List, Optional, Tuple

# 금액 관련 키워드
AMOUNT_KEYWORDS = [
    "기초금액", "예정가격", "추정가격", "추정금액", "사업비", "예산", "금액", "입찰금액",
    "계약금액", "총사업비", "총예산", "소요예산", "집행금액", "용역비", "공사비",
    "설계금액", "구매예정금액", "구매금액", "한도액", "배정액",
    "부가세", "VAT", "부가가치세",
]

# 제외/혼동 키워드
AMOUNT_NEGATIVE_KEYWORDS = [
    "입찰보증금", "계약보증금", "하자보증금", "보증금",
    "인지세", "수수료", "연체", "벌점", "지체상금",
    "납부", "보험", "보증보험",
]

MONEY_REGEXES = [
    r"(?P<num>\d{1,3}(?:,\d{3})+|\d+)\s*(?P<unit>원|만원|천원|백만원|억원|억|천만|백만)\b",
    r"(?P<eok>\d+)\s*억\s*(?P<cheon>\d+)?\s*천?\s*(?P<man>\d+)?\s*만?\s*원?",
    r"일금\s*(?P<kor_money>[일이삼사오육칠팔구십백천만억영공\s]+)원\s*정?",
]


def _clean_text(s: str) -> str:
    return (s or "").replace("\r\n", "\n").replace("\r", "\n").strip()


def _normalize_korean_money_phrase(kor_str: str) -> Optional[int]:
    """
    예:
    - 일억이천만원
    - 삼천오백만원
    - 일억이천삼백사십오만육천칠백원
    """
    if not kor_str:
        return None

    kor_str = kor_str.replace(" ", "")
    num_map = {
        "영": 0, "공": 0,
        "일": 1, "이": 2, "삼": 3, "사": 4, "오": 5,
        "육": 6, "칠": 7, "팔": 8, "구": 9,
    }
    small_unit_map = {"십": 10, "백": 100, "천": 1000}
    large_unit_map = {"만": 10_000, "억": 100_000_000}

    total = 0
    section = 0
    number = 0

    for ch in kor_str:
        if ch in num_map:
            number = num_map[ch]
        elif ch in small_unit_map:
            if number == 0:
                number = 1
            section += number * small_unit_map[ch]
            number = 0
        elif ch in large_unit_map:
            if number != 0:
                section += number
            if section == 0:
                section = 1
            total += section * large_unit_map[ch]
            section = 0
            number = 0
        else:
            return None

    total += section + number
    return total if total > 0 else None


def _normalize_money_to_won(raw: str) -> Optional[int]:
    """
    매우 보수적인 정규화:
    - 12,345,000원
    - 1200만원
    - 3억 2천만
    - 일금 일억이천만원정
    """
    if not raw:
        return None

    t = raw.replace(",", "").strip()

    # 1) "일금 ..." 한글 금액 우선
    m3 = re.search(r"일금\s*([일이삼사오육칠팔구십백천만억영공\s]+)원\s*정?", t)
    if m3:
        won = _normalize_korean_money_phrase(m3.group(1))
        if won is not None:
            return won

    # 2) 단순 단위
    m = re.search(r"(\d+)\s*(원|천원|만원|백만원|억원|억)\b", t)
    if m:
        n = int(m.group(1))
        unit = m.group(2)
        mul = {
            "원": 1,
            "천원": 1_000,
            "만원": 10_000,
            "백만원": 1_000_000,
            "억": 100_000_000,
            "억원": 100_000_000,
        }.get(unit, 1)
        return n * mul

    # 3) "3억 2천만", "3억2천5백만"류
    m2 = re.search(r"(?P<eok>\d+)\s*억", t)
    if m2:
        eok = int(m2.group("eok"))
        won = eok * 100_000_000

        m_cheonman = re.search(r"(\d+)\s*천만", t)
        if m_cheonman:
            won += int(m_cheonman.group(1)) * 10_000_000

        m_baekman = re.search(r"(\d+)\s*백만", t)
        if m_baekman:
            won += int(m_baekman.group(1)) * 1_000_000

        # "만"은 천만/백만과 중복 카운트 방지 위해 마지막 보정용
        m_man = re.search(r"(?<!천)(?<!백)(\d+)\s*만", t)
        if m_man:
            won += int(m_man.group(1)) * 10_000

        return won

    return None


@dataclass
class AmountCandidate:
    won: Optional[int]
    raw: str
    keyword: str
    page: int
    source: str
    context: str
    score: float


def _keyword_weight(k: str) -> float:
    k = k.strip()
    if k in ("입찰금액", "기초금액", "예정가격", "추정가격", "추정금액", "총사업비", "사업비"):
        return 3.0
    if k in ("예산", "총예산", "소요예산", "용역비", "공사비"):
        return 2.2
    if "부가세" in k or "VAT" in k:
        return 0.7
    return 1.5


def _has_negative_context(ctx: str) -> bool:
    c = ctx or ""
    return any(nk in c for nk in AMOUNT_NEGATIVE_KEYWORDS)


def _collect_text_sources_from_page(p: Dict[str, Any]) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []

    for key, source_name in [
        ("cleaned_for_fields", "cleaned_body"),
        ("cleaned_text", "cleaned_text"),
        ("cleaned_render_ocr_text", "cleaned_render_ocr"),
        ("final_text", "body"),
    ]:
        val = p.get(key, "") or ""
        if isinstance(val, str) and val.strip():
            out.append((val, source_name))

    for t in (p.get("tables_normalized") or []):
        for line in (t.get("row_sentences") or []):
            if isinstance(line, str) and line.strip():
                out.append((line, "table"))

    cgkv = p.get("cleaned_gemini_key_values") or {}
    if isinstance(cgkv, dict) and cgkv:
        for k, v in cgkv.items():
            out.append((f"{k}: {v}", "cleaned_gemini_kv"))

    gkv = p.get("gemini_key_values") or {}
    if isinstance(gkv, dict) and gkv:
        for k, v in gkv.items():
            out.append((f"{k}: {v}", "gemini_kv"))

    gt = p.get("gemini_tables_md") or ""
    if isinstance(gt, str) and gt.strip():
        out.append((gt, "gemini_table"))

    return out


def _extract_candidates_from_text(text: str, page: int, source: str) -> List[AmountCandidate]:
    candidates: List[AmountCandidate] = []
    t = _clean_text(text)
    if not t:
        return candidates

    lines = t.split("\n")
    for i, line in enumerate(lines):
        if not line.strip():
            continue

        kws = [k for k in AMOUNT_KEYWORDS if k in line]
        window = "\n".join(lines[max(0, i - 1): min(len(lines), i + 2)])

        if kws:
            for k in kws:
                for rgx in MONEY_REGEXES:
                    for m in re.finditer(rgx, window):
                        raw = m.group(0)
                        won = _normalize_money_to_won(raw)
                        ctx = window.strip()[:400]

                        if _has_negative_context(ctx):
                            continue

                        score = 0.0
                        score += _keyword_weight(k)

                        if won is not None:
                            score += 1.2

                        if source in ("table", "cleaned_gemini_kv", "gemini_kv", "gemini_table"):
                            score += 0.6

                        if source in ("cleaned_body", "cleaned_text", "cleaned_render_ocr"):
                            score += 0.4

                        if k in line and raw in line:
                            score += 0.6

                        candidates.append(
                            AmountCandidate(
                                won=won,
                                raw=raw,
                                keyword=k,
                                page=page,
                                source=source,
                                context=ctx,
                                score=score,
                            )
                        )

        if not kws and ("원" in line or "만원" in line or "억" in line) and ("|" in line or ":" in line):
            for rgx in MONEY_REGEXES:
                for m in re.finditer(rgx, line):
                    raw = m.group(0)
                    won = _normalize_money_to_won(raw)
                    ctx = line.strip()[:300]

                    if _has_negative_context(ctx):
                        continue

                    base = 1.0
                    if source in ("table", "gemini_table"):
                        base += 0.8
                    if source in ("cleaned_gemini_kv", "gemini_kv"):
                        base += 0.8
                    if source in ("cleaned_body", "cleaned_text", "cleaned_render_ocr"):
                        base += 0.3
                    if won is not None:
                        base += 0.8

                    candidates.append(
                        AmountCandidate(
                            won=won,
                            raw=raw,
                            keyword="(unknown)",
                            page=page,
                            source=source,
                            context=ctx,
                            score=base,
                        )
                    )

    return candidates


def _collect_all_candidates(final_pages: List[Dict[str, Any]]) -> List[AmountCandidate]:
    all_cands: List[AmountCandidate] = []

    for p in final_pages:
        page = int(p.get("page_number") or 0)
        for text, source in _collect_text_sources_from_page(p):
            all_cands.extend(_extract_candidates_from_text(text, page=page, source=source))

    return all_cands


def extract_bid_amount_from_final_pages(final_pages: List[Dict[str, Any]]) -> Dict[str, Any]:
    all_cands = _collect_all_candidates(final_pages)

    if not all_cands:
        return {
            "value_won": None,
            "value_raw": None,
            "confidence": 0.0,
            "candidates": [],
            "notes": "금액 후보를 찾지 못했습니다(파싱/스캔 품질 또는 문서에 금액 표기가 없을 수 있음).",
        }

    all_cands.sort(key=lambda c: c.score, reverse=True)
    top = all_cands[0]

    conf = 0.55
    if top.won is not None:
        conf += 0.20
    if top.keyword in ("입찰금액", "기초금액", "예정가격", "추정가격", "추정금액", "사업비", "총사업비"):
        conf += 0.15
    if top.source in ("table", "cleaned_gemini_kv", "gemini_kv", "gemini_table"):
        conf += 0.05
    if top.source in ("cleaned_body", "cleaned_text", "cleaned_render_ocr"):
        conf += 0.03
    if top.keyword in ("부가세", "VAT"):
        conf -= 0.15

    conf = max(0.0, min(1.0, conf))

    cand_out = []
    for c in all_cands[:8]:
        cand_out.append(
            {
                "page": c.page,
                "keyword": c.keyword,
                "raw": c.raw,
                "won": c.won,
                "source": c.source,
                "score": round(c.score, 3),
                "context": c.context,
            }
        )

    notes = ""
    top_won = top.won
    if top_won is not None:
        diffs = []
        for c in all_cands[1:8]:
            if c.won is not None:
                diffs.append(abs(c.won - top_won))
        if diffs and min(diffs) > max(100_000, int(top_won * 0.05)):
            notes = "상위 후보들 간 금액 차이가 큽니다. 원문 근거를 확인하세요."
            conf = min(conf, 0.65)

    return {
        "value_won": top.won,
        "value_raw": top.raw,
        "confidence": round(conf, 3),
        "candidates": cand_out,
        "notes": notes,
    }