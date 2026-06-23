"""
baseline_generator.py
---------------------
Genera una respuesta base sin RAG para comparar contra el sistema propuesto.

El baseline usa la misma informacion visual, pero NO recibe fragmentos
recuperados desde documentos propios. Sirve para demostrar si el RAG mejora:
- adecuacion a normativa/localidad;
- trazabilidad;
- reduccion de respuestas genericas;
- manejo de excepciones.
"""

from __future__ import annotations

import json
from typing import Any, Dict

from .config import settings
from .gemini_client import generate_content


BASELINE_SYSTEM_PROMPT = """
Sos un asistente general sobre separacion de residuos.
Respondé en español, de forma clara y prudente.
No cites normativa local específica porque no se te proporcionan documentos.
Si falta informacion, explicita la incertidumbre.
"""


class BaselineGenerator:
    """Generador de respuestas sin RAG para evaluacion comparativa."""

    def generate(self, visual_analysis: Dict[str, Any], user_question: str | None = None) -> str:
        prompt = f"""
Analisis visual estructurado:
{json.dumps(visual_analysis, ensure_ascii=False, indent=2)}

Pregunta o contexto adicional del usuario:
{user_question or 'No informado'}

Genera una recomendacion general de clasificacion y disposicion del residuo.
No uses fuentes documentales ni inventes normativa local.
"""
        return generate_content(
            model=settings.gemini_text_model,
            system_instruction=BASELINE_SYSTEM_PROMPT,
            prompt=prompt,
            temperature=0.2,
            max_output_tokens=700,
            timeout=settings.text_timeout_seconds,
        )
