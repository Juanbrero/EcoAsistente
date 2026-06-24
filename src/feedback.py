"""
feedback.py
-----------
Registro enriquecido de retroalimentacion humana.

El feedback no reentrena automaticamente el modelo. Se usa como evidencia para:
- analisis de errores;
- evaluacion humana del prototipo;
- mejora del corpus documental;
- ajuste de prompts;
- construccion de casos de prueba.
"""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime
from typing import Any, Dict

from .config import settings


FEEDBACK_FILE = "feedback.csv"


def _load_log_payload(log_path: str) -> Dict[str, Any]:
    if not log_path or not os.path.exists(log_path):
        return {}

    try:
        with open(log_path, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception:
        return {}


def save_feedback(
    *,
    log_path: str,
    rating: str,
    comment: str | None = None,
    error_category: str | None = None,
) -> str:
    """Agrega una fila enriquecida de feedback y devuelve la ruta del CSV."""
    os.makedirs(settings.outputs_dir, exist_ok=True)

    path = os.path.join(settings.outputs_dir, FEEDBACK_FILE)
    file_exists = os.path.exists(path)

    log_payload = _load_log_payload(log_path)

    confidence = log_payload.get("confidence", {})
    answer_evaluation = log_payload.get("answer_evaluation", {})

    fieldnames = [
        "timestamp",
        "log_path",
        "rating",
        "error_category",
        "comment",
        "image_filename",
        "user_question",
        "confidence_label",
        "confidence_score",
        "risk_hallucination",
        "requires_human_review",
        "retrieved_chunks_count",
        "final_answer_excerpt",
    ]

    final_answer = log_payload.get("final_answer", "") or ""

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "log_path": log_path,
        "rating": rating,
        "error_category": error_category or "",
        "comment": comment or "",
        "image_filename": log_payload.get("image_filename", ""),
        "user_question": log_payload.get("user_question", ""),
        "confidence_label": confidence.get("label", ""),
        "confidence_score": confidence.get("score", ""),
        "risk_hallucination": answer_evaluation.get("riesgo_alucinacion", ""),
        "requires_human_review": answer_evaluation.get("requiere_revision_humana", ""),
        "retrieved_chunks_count": len(log_payload.get("retrieved_chunks", [])),
        "final_answer_excerpt": final_answer[:500],
    }

    with open(path, "a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)

    return path