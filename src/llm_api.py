import io
import json
import logging
import os
import re
from typing import Any, Dict, Optional

from PIL import Image
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

DEFAULT_GEMINI_MODEL = "gemini-3.0-flash"


class APIAuthenticationError(Exception):
    pass


class APIRateLimitError(Exception):
    pass


class APIServerError(Exception):
    pass


class APIResponseParseError(Exception):
    pass


def is_gemini_available() -> bool:
    return bool(os.getenv("GEMINI_API_KEY", "").strip())


def get_gemini_api_key() -> str:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise APIAuthenticationError("GEMINI_API_KEY 환경변수가 없습니다.")
    return api_key


def get_gemini_client() -> genai.Client:
    return genai.Client(api_key=get_gemini_api_key())


def safe_json_load(text: str) -> Dict[str, Any]:
    raw = (text or "").strip()
    if not raw:
        return {
            "key_values": {},
            "ocr_text": "",
            "summary": "빈 응답",
        }

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except Exception:
            pass

    brace = re.search(r"\{.*\}", raw, re.DOTALL)
    if brace:
        try:
            return json.loads(brace.group(0))
        except Exception:
            pass

    raise APIResponseParseError(f"JSON 파싱 실패. 원본 응답 일부: {raw[:500]}")


def pil_image_to_part(pil_img: Image.Image) -> types.Part:
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return types.Part.from_bytes(data=buf.getvalue(), mime_type="image/png")


def build_document_vision_prompt() -> str:
    return """
너는 한국어 공고문/제안서 문서 이미지 분석기다.
이미지에서 보이는 내용만 정확히 읽어라.
특히 아래를 잘 보존하라:
- '가.', '나.', '다.' 같은 조항형 목록
- '1)', '2)' 같은 번호 목록
- 날짜, 금액, 기관명, 장소, 문의처
- 표 안의 짧은 문구와 제출서류 목록
- 체크박스, 서명/날인, 표의 짧은 항목명

확신이 없으면 절대 추정하지 말고 '?' 또는 'UNKNOWN'으로 표시하라.

반드시 아래 JSON만 출력:
{
  "key_values": { "항목": "값" },
  "ocr_text": "",
  "summary": ""
}
""".strip()


def _wrap_genai_error(e: Exception) -> Exception:
    try:
        from google.genai import errors as genai_errors
    except Exception:
        return e

    msg = str(e)

    if isinstance(e, genai_errors.ClientError):
        code = getattr(e, "code", 0)
        lowered = msg.lower()
        if code in (401, 403) or "api_key" in lowered or "api key" in lowered:
            return APIAuthenticationError(f"Gemini 인증 실패: {msg}")
        if code == 429:
            return APIRateLimitError(f"Gemini 요청 한도 초과: {msg}")
        return APIServerError(f"Gemini 클라이언트 오류: {msg}")

    if isinstance(e, genai_errors.ServerError):
        return APIServerError(f"Gemini 서버 오류: {msg}")

    return e


def generate_image_json(
    pil_img: Image.Image,
    *,
    model_name: Optional[str] = None,
    prompt: Optional[str] = None,
    temperature: float = 0.0,
    max_output_tokens: int = 4096,
) -> Dict[str, Any]:
    client = get_gemini_client()

    try:
        response = client.models.generate_content(
            model=model_name or DEFAULT_GEMINI_MODEL,
            contents=[
                prompt or build_document_vision_prompt(),
                pil_image_to_part(pil_img),
            ],
            config=types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                response_mime_type="application/json",
            ),
        )
    except Exception as e:
        raise _wrap_genai_error(e) from e

    usage = getattr(response, "usage_metadata", None)
    if usage:
        logger.info(
            "Gemini token usage — prompt: %s, output: %s, total: %s",
            getattr(usage, "prompt_token_count", None),
            getattr(usage, "candidates_token_count", None),
            getattr(usage, "total_token_count", None),
        )

    raw_text = getattr(response, "text", "") or ""
    return safe_json_load(raw_text)