"""
response_generator.py
---------------------
Genera la respuesta final para el usuario usando Gemini API.

Este modulo recibe un prompt ya construido con:
- analisis visual del residuo;
- fragmentos recuperados por RAG;
- consulta opcional del usuario.

La generacion se mantiene separada del analisis de imagen y del vector store
para que el flujo sea facil de explicar, probar y reemplazar.
"""

from __future__ import annotations

from .config import settings
from .prompt_builder import SYSTEM_PROMPT
from .gemini_client import generate_content


class ResponseGenerator:
    """Generador de respuestas con Gemini API."""

    def generate(self, prompt: str) -> str:
        """
        Envia el prompt final al modelo de texto configurado.

        La respuesta final debe estar fundada en los fragmentos recuperados por
        el RAG y debe explicitar dudas cuando la imagen o la evidencia documental
        sean insuficientes.
        """
        return generate_content(
            model=settings.gemini_text_model,
            system_instruction=SYSTEM_PROMPT,
            prompt=prompt,
            temperature=0.2,
            max_output_tokens=900,
            timeout=settings.text_timeout_seconds,
        )
