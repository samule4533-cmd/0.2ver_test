import re
from typing import Any, Dict, List


def _safe_str(x: Any) -> str:
    if x is None:
        return ""
    return str(x)


def normalize_escapes(text: str) -> str:
    s = _safe_str(text)
    s = s.replace("\\r\\n", "\n")
    s = s.replace("\\n", "\n")
    s = s.replace("\\t", "\t")
    s = s.replace("\\r", "")
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    return s


def normalize_unicode_noise(text: str) -> str:
    s = _safe_str(text)
    replacements = {
        "\u00a0": " ",
        "\u200b": "",
        "\ufeff": "",
        "…": "...",
        "⋯": "...",
        "：": ":",
        "－": "-",
        "–": "-",
        "—": "-",
        "∼": "~",
    }
    for k, v in replacements.items():
        s = s.replace(k, v)
    return s


def normalize_checkbox_marks(text: str) -> str:
    s = _safe_str(text)

    s = re.sub(r"\[\s*[vV✓✔]\s*\]", "[선택]", s)
    s = s.replace("☑", "[선택]")
    s = s.replace("✅", "[선택]")
    s = s.replace("■", "[선택]")

    s = re.sub(r"\[\s*\]", "[미선택]", s)
    s = s.replace("☐", "[미선택]")
    s = s.replace("□", "[미선택]")

    return s


def remove_ocr_artifacts(text: str) -> str:
    s = _safe_str(text)

    s = re.sub(r"\|[Ww¥]+\b", " ", s)
    s = re.sub(r"\b[Ww¥]+\|", " ", s)
    s = re.sub(r"\|\s*\|\s*\|+", " | ", s)
    s = re.sub(r"[ \t]*\|[ \t]*\n", "\n", s)
    s = re.sub(r"\n[ \t]*\|[ \t]*", "\n", s)

    s = re.sub(r"\|\s*:?-{2,}:?\s*", "| ", s)
    s = re.sub(r"-{4,}", " ", s)
    s = re.sub(r"={4,}", " ", s)

    s = re.sub(r"([|/\\])\1{2,}", r"\1", s)
    s = re.sub(r"([.])\1{3,}", "...", s)

    return s


def normalize_spaces(text: str) -> str:
    s = _safe_str(text)
    s = s.replace("\t", " ")

    lines = []
    for line in s.split("\n"):
        line = re.sub(r"[ \u00a0]+", " ", line).strip()
        lines.append(line)

    s = "\n".join(lines)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def join_broken_sentences(text: str) -> str:
    lines = [_safe_str(x).rstrip() for x in text.split("\n")]
    out: List[str] = []

    def is_label_like(line: str) -> bool:
        return bool(re.match(r"^([0-9]+[.)]|[①-⑳]|[-•●]|[가-힣A-Za-z0-9_ ]{1,20}:)", line))

    for line in lines:
        if not line:
            out.append("")
            continue

        if not out:
            out.append(line)
            continue

        prev = out[-1]

        if not prev:
            out.append(line)
            continue

        prev_end_mid = bool(re.search(r"[가-힣A-Za-z0-9,)]$", prev))
        curr_starts_mid = bool(re.match(r"^[가-힣a-z0-9(]", line))

        if is_label_like(prev) or is_label_like(line):
            out.append(line)
            continue

        if prev_end_mid and curr_starts_mid and len(prev) < 80:
            out[-1] = f"{prev} {line}"
        else:
            out.append(line)

    return "\n".join(out)


def convert_pipe_tables_to_key_value_lines(text: str) -> str:
    lines = text.split("\n")
    out: List[str] = []

    for line in lines:
        raw = line.strip()
        if not raw:
            out.append("")
            continue

        if raw.count("|") >= 2:
            parts = [p.strip() for p in raw.split("|")]
            parts = [p for p in parts if p]

            if len(parts) == 2:
                k, v = parts
                if k and v:
                    out.append(f"{k}: {v}")
                    continue

            joined = " ".join(parts)
            if re.search(r"(항목|내용|구분|비고)", joined) and len(parts) <= 4:
                continue

        out.append(line)

    return "\n".join(out)


def normalize_colon_lines(text: str) -> str:
    lines = []
    for line in text.split("\n"):
        line = re.sub(r"\s*:\s*", ": ", line)
        line = re.sub(r"\s{2,}", " ", line).strip()
        lines.append(line)
    return "\n".join(lines).strip()


def flatten_short_multiline_values(text: str) -> str:
    lines = [x.strip() for x in text.split("\n")]
    out: List[str] = []

    label_pat = re.compile(
        r"^(발주기관|발주부서|발주날짜|발주내용|성명|소속|연락처|주소|사업명|공고번호|계약방법|기초금액|추정가격|예정가격)$"
    )

    i = 0
    while i < len(lines):
        cur = lines[i]

        if not cur:
            out.append("")
            i += 1
            continue

        if label_pat.match(cur):
            if i + 1 < len(lines):
                nxt = lines[i + 1].strip()
                if nxt and len(nxt) <= 80 and ":" not in nxt:
                    out.append(f"{cur}: {nxt}")
                    i += 2
                    continue

        out.append(cur)
        i += 1

    return "\n".join(out)


def clean_key(key: str) -> str:
    k = _safe_str(key).strip()
    k = normalize_escapes(k)
    k = normalize_unicode_noise(k)
    k = re.sub(r"\s+", " ", k)
    k = re.sub(r"^\d+\s*[.)]?\s*", "", k)
    k = re.sub(r"^\[[^\]]+\]\s*", "", k)
    k = k.replace(" - ", " > ")
    k = k.replace(" – ", " > ")
    k = k.replace(" — ", " > ")
    return k.strip(" :-")


def clean_value(value: Any) -> str:
    v = _safe_str(value)
    v = normalize_escapes(v)
    v = normalize_unicode_noise(v)
    v = normalize_checkbox_marks(v)
    v = remove_ocr_artifacts(v)
    v = normalize_spaces(v)
    return v


def clean_gemini_key_values(raw_kv: Dict[str, Any]) -> Dict[str, str]:
    if not isinstance(raw_kv, dict):
        return {}

    out: Dict[str, str] = {}
    for k, v in raw_kv.items():
        ck = clean_key(k)
        cv = clean_value(v)

        if not ck:
            continue

        if ck in out and cv and cv not in out[ck]:
            out[ck] = f"{out[ck]} / {cv}"
        else:
            out[ck] = cv

    return out


def build_cleaned_for_fields(text: str) -> str:
    s = _safe_str(text)
    s = normalize_escapes(s)
    s = normalize_unicode_noise(s)
    s = normalize_checkbox_marks(s)
    s = remove_ocr_artifacts(s)
    s = convert_pipe_tables_to_key_value_lines(s)
    s = flatten_short_multiline_values(s)
    s = normalize_colon_lines(s)
    s = join_broken_sentences(s)
    s = normalize_spaces(s)
    return s.strip()


def build_human_readable_cleaned_text(text: str) -> str:
    s = _safe_str(text)
    s = normalize_escapes(s)
    s = normalize_unicode_noise(s)
    s = normalize_checkbox_marks(s)
    s = remove_ocr_artifacts(s)
    s = convert_pipe_tables_to_key_value_lines(s)
    s = normalize_colon_lines(s)
    s = normalize_spaces(s)
    return s.strip()


def clean_page_payload(page_payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(page_payload, dict):
        return page_payload

    final_text = _safe_str(page_payload.get("final_text", ""))
    render_ocr_text = _safe_str(page_payload.get("render_ocr_text", ""))
    raw_gemini_kv = page_payload.get("gemini_key_values", {}) or {}

    cleaned_text = build_human_readable_cleaned_text(final_text)
    cleaned_for_fields = build_cleaned_for_fields(final_text)

    cleaned_render_ocr_text = ""
    if render_ocr_text:
        cleaned_render_ocr_text = build_human_readable_cleaned_text(render_ocr_text)

    cleaned_gemini_kv = clean_gemini_key_values(raw_gemini_kv)

    page_payload["cleaned_text"] = cleaned_text
    page_payload["cleaned_for_fields"] = cleaned_for_fields
    page_payload["cleaned_render_ocr_text"] = cleaned_render_ocr_text
    page_payload["cleaned_gemini_key_values"] = cleaned_gemini_kv

    return page_payload


def clean_final_pages(final_pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for p in final_pages:
        out.append(clean_page_payload(p))
    return out