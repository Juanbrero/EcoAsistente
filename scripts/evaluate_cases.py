"""
scripts/evaluate_cases.py
-------------------------
Ejecuta una evaluacion offline sobre casos de prueba con imagenes.

Uso recomendado desde la raiz del proyecto:
    python scripts/evaluate_cases.py --cases tests/casos_evaluacion.csv

Opcionalmente reconstruir indice antes de evaluar:
    python scripts/evaluate_cases.py --cases tests/casos_evaluacion.csv --reindex

El CSV de casos debe tener estas columnas:
    id,image_path,user_question,expected_category,expected_result,notes

Las imagenes deben existir localmente. Si una imagen falta, el caso se marca
como SKIPPED para no cortar toda la evaluacion.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Permite importar src cuando el script se ejecuta desde /scripts.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings, validate_settings
from src.rag_engine import EcoAsistenteRAG
from src.vector_store import VectorStore
from src.evaluation_utils import read_cases_csv, write_rows_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Evalua EcoAsistente sobre casos con imagenes.")
    parser.add_argument("--cases", default="tests/casos_evaluacion.csv", help="Ruta del CSV de casos.")
    parser.add_argument("--output", default="outputs/evaluation_results.csv", help="Ruta del CSV de resultados.")
    parser.add_argument("--reindex", action="store_true", help="Reconstruye el indice vectorial antes de evaluar.")
    args = parser.parse_args()

    validate_settings()
    os.makedirs(settings.outputs_dir, exist_ok=True)

    if args.reindex:
        print("Reconstruyendo indice vectorial...")
        count = VectorStore().index_documents(reset=True)
        print(f"Chunks indexados: {count}")

    cases = read_cases_csv(args.cases)
    pipeline = EcoAsistenteRAG()
    rows = []

    for case in cases:
        case_id = case.get("id", "")
        image_path = case.get("image_path", "").strip()
        user_question = case.get("user_question", "").strip() or None
        expected_category = case.get("expected_category", "").strip()
        expected_result = case.get("expected_result", "").strip()

        print(f"Evaluando caso {case_id}: {image_path}")

        if not image_path or not os.path.exists(image_path):
            rows.append(
                {
                    "id": case_id,
                    "image_path": image_path,
                    "status": "SKIPPED_IMAGE_NOT_FOUND",
                    "expected_category": expected_category,
                    "predicted_object": "",
                    "predicted_category": "",
                    "confidence_label": "",
                    "risk_hallucination": "",
                    "requires_human_review": "",
                    "total_seconds": "",
                    "log_path": "",
                    "expected_result": expected_result,
                    "final_answer": "",
                    "error": "Imagen no encontrada.",
                }
            )
            continue

        start = time.perf_counter()
        try:
            result = pipeline.answer(
                image_path=image_path,
                image_filename=os.path.basename(image_path),
                user_question=user_question,
            )
            visual = result.get("visual_analysis", {})
            evaluation = result.get("answer_evaluation", {})
            confidence = result.get("confidence", {})
            rows.append(
                {
                    "id": case_id,
                    "image_path": image_path,
                    "status": "OK",
                    "expected_category": expected_category,
                    "predicted_object": visual.get("objeto", ""),
                    "predicted_category": visual.get("categoria_tentativa", ""),
                    "confidence_label": confidence.get("label", ""),
                    "risk_hallucination": evaluation.get("riesgo_alucinacion", ""),
                    "requires_human_review": evaluation.get("requiere_revision_humana", ""),
                    "total_seconds": result.get("timings", {}).get("total_seconds", round(time.perf_counter() - start, 3)),
                    "log_path": result.get("log_path", ""),
                    "expected_result": expected_result,
                    "final_answer": result.get("final_answer", ""),
                    "error": "",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "id": case_id,
                    "image_path": image_path,
                    "status": "ERROR",
                    "expected_category": expected_category,
                    "predicted_object": "",
                    "predicted_category": "",
                    "confidence_label": "",
                    "risk_hallucination": "",
                    "requires_human_review": "",
                    "total_seconds": round(time.perf_counter() - start, 3),
                    "log_path": "",
                    "expected_result": expected_result,
                    "final_answer": "",
                    "error": str(exc),
                }
            )

    write_rows_csv(args.output, rows)
    print(f"Resultados guardados en: {args.output}")


if __name__ == "__main__":
    main()
