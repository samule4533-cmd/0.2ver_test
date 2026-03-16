# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

사내 챗봇을 위한 PDF 문서 파싱 → 청킹 → 벡터DB 적재 파이프라인. 한국어 기업 문서(특허, 인증서, 회사소개서 등) 처리에 특화된 RAG 인제스트 시스템.

## Commands

패키지 관리자는 `uv`를 사용한다.

```bash
# 의존성 설치
uv sync

# PDF 파싱 실행 (src/ 디렉토리에서 실행해야 상대 임포트가 동작함)
cd src && uv run python pdf_parser.py

# 벡터DB 적재 및 테스트 쿼리
cd src && uv run python vector_db.py
```

실행 전 `.env` 파일에 다음 환경변수가 필요하다:
- `GEMINI_API_KEY`: Gemini File API 키
- `DOC_SOURCE_TYPE`: `company` 또는 `notice`
- `DEFAULT_PDF_SUBDIR`: `data/raw/company/` 하위 서브디렉토리 (예: `certification_list_1`)
- `DEFAULT_PDF_NAME`: 처리할 PDF 파일명 (예: `sample_company.pdf`)

## Architecture

### 데이터 흐름

```
data/raw/{company|sample_notices}/{subdir}/{file}.pdf
    → pdf_parser.py (Gemini File API로 Markdown 변환)
    → chunker.py (헤더 기반 청크 분리)
    → output_writer.py (data/processed/에 .md, .json, vector_chunks.json 저장)
    → vector_db.py (ChromaDB에 임베딩 적재)
    → data/vector_store/chroma/ (영구 저장)
```

### 모듈 역할

- **`pdf_parser.py`**: 파이프라인 진입점. PDF를 Gemini File API에 업로드 → Markdown 생성 → `normalize_markdown_headings()`로 볼드 제목을 헤더로 보정. `async_main()`이 실제 실행 함수.
- **`chunker.py`**: Markdown 헤더(`#`, `##`, `###`) 기준 1차 분리, 1500자 초과 섹션은 빈 줄 기준 2차 분리(`paragraph_group`). 청크 간 overlap 없음.
- **`vector_db.py`**: `paraphrase-multilingual-MiniLM-L12-v2` 로컬 임베딩, ChromaDB PersistentClient. `upsert_chunks_to_chroma()`는 배치 50개 단위로 적재. `query_collection()`이 검색 진입점.
- **`llm_api.py`**: Gemini 클라이언트 싱글턴, `safe_json_load()`는 JSON 파싱 실패 시 3단계 폴백(직접 파싱 → 펜스 코드블록 추출 → `{}` 복구).
- **`output_writer.py`**: 처리 결과를 `.md`, `.json`, `vector_chunks.json`, `parse_report.json`, `fields.json`으로 저장.
- **`field_extract.py`**: 조달/입찰 공고 문서 전용 금액 필드 추출. 현재 company 모드에서는 비활성화.
- **`image_parser.py`**: PyMuPDF로 이미지 추출 + Gemini 캡션 생성. `ENABLE_IMAGE_CAPTIONS=False`로 현재 비활성화.
- **`company_ingest.py`**: 미구현 빈 파일.

### 한글 파일명 처리

`pdf_parser.py`의 `_upload_and_wait()`는 한글 파일명 업로드 오류를 피하기 위해 임시 디렉토리에 영문명(`upload_input.pdf`)으로 복사 후 업로드한다.

### 출력 디렉토리 구조

입력 경로의 상대 구조가 출력에도 유지된다:
- 입력: `data/raw/company/certification_list_1/sample_company.pdf`
- 출력: `data/processed/parsing_result_company/certification_list_1/sample_company/`

### ChromaDB 컬렉션 구성

- 컬렉션명: `ninewatt_company_local` (company 문서)
- 저장 위치: `data/vector_store/chroma/`
- 거리 메트릭: cosine similarity
- 메타데이터 필터: `doc_type`, `document_id`, `header` 등으로 where 절 필터링 가능

## 현재 미구현 영역

- RAG 응답 생성 (검색 결과 → LLM 프롬프트 조립 → 응답): 완전 미구현
- `company_ingest.py`: 여러 PDF 일괄 처리 로직 없음
- `main.py`: 실질적 오케스트레이션 없음

## 개발 방향 및 작업 원칙

### 핵심 목표
이 프로젝트는 단순 PDF 텍스트 추출기가 아니라, 회사 문서를 신뢰 가능한 지식 자원으로 변환하기 위한 파이프라인이다.  
최종 목표는 PDF 문서 파싱 → 청킹 → 벡터DB 적재를 거쳐, 사내 챗봇이 근거 기반 질의응답을 수행할 수 있도록 만드는 것이다.

### 중요 기준
- 정확도 우선
- 회사자료/정형 문서 특화
- 청킹 및 검색 품질 중심
- RAG 연결을 고려한 구조 유지
- 실사용 가능한 결과 지향

### 작업하면서 중점적으로 본 점
- 한글/숫자/표가 최대한 깨지지 않도록 파싱 정확도 개선
- OCR, Markdown 변환, 후처리 보강 방식 비교 및 적용
- 청킹이 문맥을 해치지 않도록 구조 유지
- 벡터DB 적재 이후 검색 가능한 형태의 metadata 유지
- 겉보기 결과보다 실제 업무 활용 가능성을 더 중요하게 판단
- 항상 구체적인 내용과 어떤 작업을 하면 그 이유를 명확하게 드러내도록

### 유의사항
- 공고문/회사자료는 숫자, 조건, 표 정보 손실에 민감하므로 단순 텍스트 추출 품질만으로 판단하면 안 된다.
- preview 기준 확인만으로는 부족할 수 있어, 필요 시 full chunk 기준 검증이 필요하다.
- 파싱, 청킹, 검색은 분리된 단계가 아니라 최종 답변 품질에 함께 영향을 주는 하나의 흐름으로 본다.