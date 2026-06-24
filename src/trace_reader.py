"""
trace_reader.py
---------------
Lectura de logs de trazabilidad generados por el EcoAsistente.

Este modulo no ejecuta IA ni modifica datos. Solo transforma los logs JSON
guardados en una estructura apta para mostrarse en la interfaz /trace.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

from .config import settings


def list_trace_logs(limit: int = 30) -> List[Dict[str, Any]]:
    """Lista los ultimos logs disponibles, ordenados del mas reciente al mas antiguo."""
    log_dir = Path(settings.log_dir)

    if not log_dir.exists():
        return []

    files = sorted(
        log_dir.glob("interaction_*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    traces: List[Dict[str, Any]] = []

    for path in files[:limit]:
        try:
            with open(path, "r", encoding="utf-8") as file:
                payload = json.load(file)

            traces.append(
                {
                    "id": path.stem,
                    "path": str(path),
                    "timestamp": payload.get("timestamp", ""),
                    "image_filename": payload.get("image_filename", ""),
                    "user_question": payload.get("user_question", ""),
                    "final_answer": payload.get("final_answer", ""),
                    "confidence_label": payload.get("confidence", {}).get("label", ""),
                    "confidence_score": payload.get("confidence", {}).get("score", ""),
                    "total_seconds": payload.get("timings", {}).get("total_seconds", ""),
                }
            )
        except Exception:
            continue

    return traces


def load_trace(log_id: str) -> Dict[str, Any]:
    """Carga un log especifico por id, por ejemplo interaction_20260624_153000."""
    safe_id = Path(log_id).stem

    if not safe_id.startswith("interaction_"):
        raise ValueError("Identificador de trazabilidad invalido.")

    path = Path(settings.log_dir) / f"{safe_id}.json"

    if not path.exists():
        raise FileNotFoundError(f"No existe el log solicitado: {safe_id}")

    with open(path, "r", encoding="utf-8") as file:
        payload = json.load(file)

    return {
        "id": safe_id,
        "path": str(path),
        "payload": payload,
        "steps": build_trace_steps(payload),
    }


def build_trace_steps(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convierte el JSON del log en pasos explicables para mostrar en /trace."""
    timings = payload.get("timings", {})
    visual_analysis = payload.get("visual_analysis", {})
    retrieved_chunks = payload.get("retrieved_chunks", [])
    confidence = payload.get("confidence", {})
    answer_evaluation = payload.get("answer_evaluation", {})

    return [
        {
            "number": 1,
            "title": "Recepción de la imagen",
            "description": "El usuario sube una fotografía del residuo. La aplicación valida el formato y guarda una copia temporal para analizarla.",
            "data": {
                "image_filename": payload.get("image_filename"),
                "user_question": payload.get("user_question"),
            },
            "time": None,
        },
        {
            "number": 2,
            "title": "Análisis visual multimodal",
            "description": "Gemini Vision analiza la imagen y extrae atributos observables del residuo: tipo de objeto, materiales probables, estado, contaminantes visibles y nivel de confianza.",
            "data": visual_analysis,
            "time": timings.get("image_analysis_seconds"),
        },
        {
            "number": 3,
            "title": "Construcción de consulta RAG",
            "description": "El sistema transforma el análisis visual y la consulta opcional del usuario en una búsqueda textual optimizada para recuperar normas o guías relevantes.",
            "data": {
                "rag_query": payload.get("rag_query"),
            },
            "time": None,
        },
        {
            "number": 4,
            "title": "Recuperación documental",
            "description": "La consulta se convierte en embedding y se compara contra la base vectorial Chroma para recuperar los fragmentos documentales más cercanos.",
            "data": {
                "retrieved_chunks_count": len(retrieved_chunks),
                "retrieved_chunks": retrieved_chunks,
            },
            "time": timings.get("retrieval_seconds"),
        },
        {
            "number": 5,
            "title": "Generación de respuesta final",
            "description": "El modelo de texto genera una recomendación usando como contexto el análisis visual y los fragmentos recuperados. La respuesta debe quedar fundada en la base documental.",
            "data": {
                "final_answer": payload.get("final_answer"),
            },
            "time": timings.get("generation_seconds"),
        },
        {
            "number": 6,
            "title": "Cálculo de confianza operacional",
            "description": "El sistema calcula una confianza transparente usando señales como confianza visual, cantidad de fragmentos recuperados y similitud aproximada.",
            "data": confidence,
            "time": None,
        },
        {
            "number": 7,
            "title": "Auditoría automática de respuesta",
            "description": "Un evaluador simple revisa si la respuesta parece usar fuentes, si puede tener riesgo de alucinación y si requiere revisión humana.",
            "data": answer_evaluation,
            "time": timings.get("answer_evaluation_seconds"),
        },
        {
            "number": 8,
            "title": "Registro de trazabilidad",
            "description": "La interacción completa se guarda como JSON para permitir análisis posterior, defensa técnica y evaluación del prototipo.",
            "data": {
                "log_path": payload.get("log_path"),
                "stored_path": payload.get("path"),
                "total_seconds": timings.get("total_seconds"),
            },
            "time": timings.get("total_seconds"),
        },
    ]