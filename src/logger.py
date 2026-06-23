"""
logger.py
---------
Registra cada ejecucion del prototipo.

Los logs son utiles para la evaluacion del trabajo final: permiten revisar que
imagen se analizo, que atributos se extrajeron, que documentos se recuperaron,
que respuesta produjo el sistema, que confianza operacional obtuvo y que dijo el
evaluador simple.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Dict, Any, List
from .config import settings


def save_interaction_log(
    image_filename: str,
    user_question: str | None,
    visual_analysis: Dict[str, Any],
    rag_query: str,
    retrieved_chunks: List[Dict[str, Any]],
    final_answer: str,
    confidence: Dict[str, Any] | None = None,
    answer_evaluation: Dict[str, Any] | None = None,
    timings: Dict[str, float] | None = None,
) -> str:
    """Guarda un log JSON y devuelve la ruta del archivo creado."""
    os.makedirs(settings.log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(settings.log_dir, f"interaction_{timestamp}.json")

    payload = {
        "timestamp": timestamp,
        "image_filename": image_filename,
        "user_question": user_question,
        "visual_analysis": visual_analysis,
        "rag_query": rag_query,
        "retrieved_chunks": retrieved_chunks,
        "final_answer": final_answer,
        "confidence": confidence or {},
        "answer_evaluation": answer_evaluation or {},
        "timings": timings or {},
    }

    with open(log_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)

    return log_path
