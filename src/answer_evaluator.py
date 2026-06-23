"""
answer_evaluator.py
-------------------
Agente simple de evaluacion/auditoria de respuestas.

Este componente NO orquesta el sistema ni reemplaza al pipeline RAG. Se ejecuta
al final, como una capa de control de calidad. Revisa si la respuesta generada:
- esta respaldada por los fragmentos recuperados;
- contiene afirmaciones potencialmente no respaldadas;
- expresa incertidumbre cuando la evidencia es insuficiente;
- incluye advertencias ante residuos especiales o peligrosos.

En la defensa se puede presentar como "agente simple evaluador" o "verificador
LLM posterior a la generacion", no como arquitectura multiagente.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from .config import settings
from .gemini_client import generate_content


EVALUATOR_SYSTEM_PROMPT = """
Sos un evaluador tecnico de respuestas generadas por un sistema RAG sobre gestion de residuos.
Tu tarea es auditar la respuesta, no corregirla creativamente.
Debes responder exclusivamente JSON valido.
"""


def _safe_json_loads(text: str) -> Dict[str, Any]:
    """Parsea JSON aunque el modelo devuelva fences de markdown por accidente."""
    cleaned = text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned.removeprefix("```json").strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```").strip()
    if cleaned.endswith("```"):
        cleaned = cleaned.removesuffix("```").strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return {
            "respuesta_aprobada": False,
            "riesgo_alucinacion": "desconocido",
            "usa_fuentes": False,
            "expresa_incertidumbre": False,
            "advertencias": ["No se pudo parsear la auditoria del evaluador."],
            "comentario_evaluador": cleaned[:1000],
        }
    if not isinstance(parsed, dict):
        return {
            "respuesta_aprobada": False,
            "riesgo_alucinacion": "desconocido",
            "usa_fuentes": False,
            "expresa_incertidumbre": False,
            "advertencias": ["La auditoria no devolvio un objeto JSON."],
            "comentario_evaluador": str(parsed)[:1000],
        }
    return parsed


def _format_chunks(retrieved_chunks: List[Dict[str, Any]]) -> str:
    """Resume chunks recuperados para que el evaluador pueda compararlos."""
    if not retrieved_chunks:
        return "No se recuperaron fragmentos."

    formatted = []
    for index, chunk in enumerate(retrieved_chunks, start=1):
        formatted.append(
            f"Fragmento {index}\n"
            f"Fuente: {chunk.get('source', 'desconocido')}\n"
            f"Texto: {chunk.get('text', '')[:1200]}"
        )
    return "\n\n".join(formatted)


class AnswerEvaluator:
    """Evaluador LLM posterior a la respuesta final."""

    def evaluate(
        self,
        *,
        visual_analysis: Dict[str, Any],
        retrieved_chunks: List[Dict[str, Any]],
        final_answer: str,
    ) -> Dict[str, Any]:
        """
        Evalua la respuesta final y devuelve un JSON de auditoria.

        La salida se disena para poder mostrarse en interfaz y guardarse en logs.
        """
        prompt = f"""
Audita la siguiente respuesta de un sistema RAG de residuos.

Analisis visual estructurado:
{json.dumps(visual_analysis, ensure_ascii=False, indent=2)}

Fragmentos recuperados:
{_format_chunks(retrieved_chunks)}

Respuesta final generada:
{final_answer}

Devolve exclusivamente este JSON:
{{
  "respuesta_aprobada": true,
  "riesgo_alucinacion": "bajo|medio|alto",
  "usa_fuentes": true,
  "expresa_incertidumbre": true,
  "requiere_revision_humana": false,
  "advertencias": ["..."],
  "comentario_evaluador": "..."
}}

Criterios:
- riesgo_alucinacion es alto si la respuesta afirma reglas no presentes en los fragmentos.
- requiere_revision_humana debe ser true si el residuo parece peligroso, especial o ambiguo.
- respuesta_aprobada debe ser false si la respuesta no esta respaldada por evidencia suficiente.
"""
        raw = generate_content(
            model=settings.gemini_text_model,
            system_instruction=EVALUATOR_SYSTEM_PROMPT,
            prompt=prompt,
            temperature=0.0,
            max_output_tokens=600,
            timeout=settings.text_timeout_seconds,
            response_mime_type="application/json",
        )
        return _safe_json_loads(raw)
