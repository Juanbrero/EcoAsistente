"""
scripts/compare_baseline.py
---------------------------
Compara el sistema RAG contra un baseline sin RAG.

Flujo por caso:
1. Analiza la imagen con Gemini Vision.
2. Genera una respuesta baseline sin documentos.
3. Ejecuta el pipeline completo con RAG.
4. Guarda ambas respuestas para comparacion cualitativa.

Uso:
    python scripts/compare_baseline.py --cases tests/casos_evaluacion.csv
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import validate_settings
from src.image_analyzer import ImageAnalyzer
from src.baseline_generator import BaselineGenerator
from src.rag_engine import EcoAsistenteRAG
from src.evaluation_utils import read_cases_csv, write_rows_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Compara Gemini sin RAG vs EcoAsistente RAG.")
    parser.add_argument("--cases", default="tests/casos_evaluacion.csv")
    parser.add_argument("--output", default="outputs/baseline_comparison.csv")
    args = parser.parse_args()

    validate_settings()
    cases = read_cases_csv(args.cases)
    image_analyzer = ImageAnalyzer()
    baseline = BaselineGenerator()
    rag = EcoAsistenteRAG()
    rows = []

    for case in cases:
        case_id = case.get("id", "")
        image_path = case.get("image_path", "").strip()
        user_question = case.get("user_question", "").strip() or None
        expected_result = case.get("expected_result", "").strip()

        print(f"Comparando caso {case_id}: {image_path}")
        if not image_path or not os.path.exists(image_path):
            rows.append(
                {
                    "id": case_id,
                    "status": "SKIPPED_IMAGE_NOT_FOUND",
                    "image_path": image_path,
                    "expected_result": expected_result,
                    "baseline_answer": "",
                    "rag_answer": "",
                    "rag_sources_count": "",
                    "rag_confidence": "",
                    "total_seconds": "",
                    "error": "Imagen no encontrada.",
                }
            )
            continue

        start = time.perf_counter()
        try:
            visual_analysis = image_analyzer.analyze(image_path)
            baseline_answer = baseline.generate(visual_analysis, user_question)
            rag_result = rag.answer(
                image_path=image_path,
                image_filename=os.path.basename(image_path),
                user_question=user_question,
            )
            rows.append(
                {
                    "id": case_id,
                    "status": "OK",
                    "image_path": image_path,
                    "expected_result": expected_result,
                    "baseline_answer": baseline_answer,
                    "rag_answer": rag_result.get("final_answer", ""),
                    "rag_sources_count": len(rag_result.get("retrieved_chunks", [])),
                    "rag_confidence": rag_result.get("confidence", {}).get("label", ""),
                    "total_seconds": round(time.perf_counter() - start, 3),
                    "error": "",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "id": case_id,
                    "status": "ERROR",
                    "image_path": image_path,
                    "expected_result": expected_result,
                    "baseline_answer": "",
                    "rag_answer": "",
                    "rag_sources_count": "",
                    "rag_confidence": "",
                    "total_seconds": round(time.perf_counter() - start, 3),
                    "error": str(exc),
                }
            )

    write_rows_csv(args.output, rows)
    print(f"Comparacion guardada en: {args.output}")


if __name__ == "__main__":
    main()
