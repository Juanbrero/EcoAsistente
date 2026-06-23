"""
scoring.py
----------
Calcula una confianza operacional simple para la respuesta del sistema.

Este modulo no intenta medir la "verdad" del residuo. Su objetivo es dar una
senial transparente y explicable sobre la solidez de la respuesta generada:
- que tan clara fue la salida del analisis visual;
- cuantos documentos recupero el RAG;
- que tan cercanos fueron los fragmentos recuperados segun la distancia vectorial;
- si hay advertencias o incertidumbres explicitas.

La idea es sumar trazabilidad y criterio de ingenieria sin convertir el sistema
en una caja negra adicional.
"""

from __future__ import annotations

from typing import Any, Dict, List


def _normalize_visual_confidence(value: Any) -> str:
    """Normaliza variantes de confianza devueltas por el modelo visual."""
    text = str(value or "").strip().lower()
    if "alta" in text or "high" in text:
        return "alta"
    if "media" in text or "medium" in text:
        return "media"
    if "baja" in text or "low" in text:
        return "baja"
    return "desconocida"


def _distance_to_similarity(distance: float | int | None) -> float:
    """
    Convierte distancia vectorial a una similitud aproximada entre 0 y 1.

    Chroma puede usar distintas metricas segun la coleccion. Para no asumir una
    escala absoluta, usamos una transformacion monotona simple: menor distancia
    implica mayor similitud. Esto sirve para ordenar y explicar, no para publicar
    como metrica estadistica rigurosa.
    """
    if distance is None:
        return 0.0
    try:
        d = float(distance)
    except (TypeError, ValueError):
        return 0.0
    if d < 0:
        d = 0.0
    return round(1.0 / (1.0 + d), 4)


def compute_confidence_score(
    visual_analysis: Dict[str, Any],
    retrieved_chunks: List[Dict[str, Any]],
    final_answer: str,
) -> Dict[str, Any]:
    """
    Calcula una confianza operacional Alta/Media/Baja.

    Devuelve un diccionario para mostrar en la interfaz y guardar en logs.
    El score se calcula con reglas simples y documentadas, faciles de defender.
    """
    reasons: List[str] = []
    score = 0

    visual_confidence = _normalize_visual_confidence(visual_analysis.get("confianza"))
    if visual_confidence == "alta":
        score += 2
        reasons.append("El analisis visual informo confianza alta.")
    elif visual_confidence == "media":
        score += 1
        reasons.append("El analisis visual informo confianza media.")
    elif visual_confidence == "baja":
        score -= 1
        reasons.append("El analisis visual informo confianza baja.")
    else:
        reasons.append("El analisis visual no informo una confianza clara.")

    if len(retrieved_chunks) >= 3:
        score += 2
        reasons.append("El RAG recupero al menos tres fragmentos documentales.")
    elif len(retrieved_chunks) >= 1:
        score += 1
        reasons.append("El RAG recupero fragmentos documentales, aunque pocos.")
    else:
        score -= 2
        reasons.append("No se recuperaron fragmentos documentales relevantes.")

    similarities = [_distance_to_similarity(chunk.get("distance")) for chunk in retrieved_chunks]
    best_similarity = max(similarities) if similarities else 0.0
    avg_similarity = round(sum(similarities) / len(similarities), 4) if similarities else 0.0

    if best_similarity >= 0.65:
        score += 2
        reasons.append("El fragmento mas cercano tiene una similitud aproximada alta.")
    elif best_similarity >= 0.45:
        score += 1
        reasons.append("El fragmento mas cercano tiene una similitud aproximada media.")
    else:
        reasons.append("La similitud aproximada de los fragmentos no es alta.")

    answer_lower = final_answer.lower()
    uncertainty_terms = ["incertid", "no se puede", "no es posible", "podria", "consultar"]
    if any(term in answer_lower for term in uncertainty_terms):
        reasons.append("La respuesta explicita incertidumbre o derivacion cuando corresponde.")

    if score >= 5:
        label = "alta"
    elif score >= 2:
        label = "media"
    else:
        label = "baja"

    return {
        "label": label,
        "score": score,
        "visual_confidence": visual_confidence,
        "retrieved_chunks_count": len(retrieved_chunks),
        "best_similarity_approx": best_similarity,
        "avg_similarity_approx": avg_similarity,
        "reasons": reasons,
    }
