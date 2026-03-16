"""
company_vectordb.py — 회사 문서 ChromaDB 적재 및 쿼리 테스트

역할:
  - data/processed/parsing_result_company/ 하위 모든 vector_chunks.json 수집
  - 전체 청크를 ChromaDB에 upsert (chunk_id 기준 중복 없음)
  - 적재 후 테스트 쿼리 실행

이전 단계:
  company_ingest.py 실행 → PDF 파싱 완료 후 실행

사용법:
  cd src && uv run python company_vectordb.py
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

from vector_db import get_chroma_dir, query_collection, print_query_summary, reset_collection, upsert_chunks_to_chroma

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# =============================================================================
# Paths
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

COMPANY_OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "parsing_result_company"
CHROMA_DIR = str(get_chroma_dir())

# =============================================================================
# Config (.env에서 읽음)
# =============================================================================
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "ninewatt_company")
# CHROMA_RESET=true 시 기존 컬렉션 삭제 후 재생성 (임베딩 모델 전환 시 사용)
CHROMA_RESET = os.getenv("CHROMA_RESET", "false").lower() == "true"


# =============================================================================
# Helpers
# =============================================================================
def collect_all_chunks() -> List[Dict[str, Any]]:
    """COMPANY_OUTPUT_ROOT 하위의 모든 vector_chunks.json을 수집하여 반환"""
    all_chunks: List[Dict[str, Any]] = []
    chunk_files = sorted(COMPANY_OUTPUT_ROOT.rglob("vector_chunks.json"))

    if not chunk_files:
        logger.warning("vector_chunks.json 없음: %s", COMPANY_OUTPUT_ROOT)
        return all_chunks

    for chunks_path in chunk_files:
        with open(chunks_path, encoding="utf-8") as f:
            chunks = json.load(f)
        all_chunks.extend(chunks)
        logger.info("[LOAD] %s (%d 청크)", chunks_path.relative_to(COMPANY_OUTPUT_ROOT), len(chunks))

    return all_chunks


# =============================================================================
# 메인 적재 로직
# =============================================================================
def upsert_all() -> None:
    # -------------------------------------------------------------------------
    # 1. 모든 파싱 결과 수집
    # -------------------------------------------------------------------------
    all_chunks = collect_all_chunks()

    if not all_chunks:
        print("적재할 청크가 없습니다. 먼저 company_ingest.py를 실행하세요.")
        return

    # -------------------------------------------------------------------------
    # 2. ChromaDB upsert
    #    내부적으로 get_or_create_collection()을 호출하며,
    #    컬렉션 생성 시 hnsw:space="cosine" 으로 설정됨 → 코사인 유사도 사용.
    #    (벡터 크기가 아닌 방향(각도) 기준으로 유사도 측정)
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # 2-0. 임베딩 모델 전환 시 기존 컬렉션 리셋 (CHROMA_RESET=true)
    #      local(384차원) → openai(1536차원) 등 차원이 달라지면 반드시 실행해야 함.
    #      이 플래그 없이 다른 차원 벡터를 upsert하면 ChromaDB가 오류를 냄.
    # -------------------------------------------------------------------------
    if CHROMA_RESET:
        logger.warning("CHROMA_RESET=true: 컬렉션 '%s' 삭제 후 재생성", COLLECTION_NAME)
        reset_collection(
            collection_name=COLLECTION_NAME,
            persist_dir=CHROMA_DIR,
            embedding_provider=EMBEDDING_PROVIDER,
        )

    logger.info("ChromaDB 적재 시작: 총 %d 청크 → 컬렉션 '%s'", len(all_chunks), COLLECTION_NAME)
    upsert_chunks_to_chroma(
        chunks=all_chunks,
        collection_name=COLLECTION_NAME,
        persist_dir=CHROMA_DIR,
        embedding_provider=EMBEDDING_PROVIDER,
        default_doc_type="company",
    )

    # -------------------------------------------------------------------------
    # 3. 결과 요약 출력
    # -------------------------------------------------------------------------
    print("\n" + "=" * 55)
    print(f"[적재 완료] 총 {len(all_chunks)}개 청크")
    print(f"  컬렉션: {COLLECTION_NAME}")
    print(f"  임베딩: {EMBEDDING_PROVIDER}")
    print(f"  저장 경로: {CHROMA_DIR}")
    print("=" * 55)

    # -------------------------------------------------------------------------
    # 4. 적재 sanity check — 컬렉션에 데이터가 실제로 들어갔는지 확인
    # -------------------------------------------------------------------------
    import chromadb
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    col = client.get_collection(COLLECTION_NAME)
    print(f"\n[sanity check] 컬렉션 '{COLLECTION_NAME}' 총 청크 수: {col.count()}")


if __name__ == "__main__":
    upsert_all()
