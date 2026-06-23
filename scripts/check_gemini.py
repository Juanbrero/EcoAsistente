"""
check_gemini.py
---------------
Script de diagnostico para verificar que la configuracion de Gemini API funcione.

Uso:
    python scripts/check_gemini.py

Este script no modifica archivos. Solo realiza:
1. Una llamada de generacion textual minima.
2. Una llamada de embedding minima.
"""

from pathlib import Path
import sys

# Permite importar src cuando el script se ejecuta desde la raiz del proyecto.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings, validate_settings  # noqa: E402
from src.gemini_client import generate_content, embed_text  # noqa: E402


def main() -> None:
    print("Verificando configuracion de Gemini API...")
    validate_settings()

    print(f"Modelo de texto: {settings.gemini_text_model}")
    text = generate_content(
        model=settings.gemini_text_model,
        prompt="Responde solo con la palabra OK.",
        temperature=0,
        max_output_tokens=64,
        timeout=settings.text_timeout_seconds,
    )
    print(f"Respuesta de texto: {text}")

    print(f"Modelo de embeddings: {settings.gemini_embedding_model}")
    vector = embed_text(
        "botella plastica reciclable",
        task_type="RETRIEVAL_QUERY",
        timeout=settings.embedding_timeout_seconds,
    )
    print(f"Embedding generado correctamente. Dimension: {len(vector)}")
    print("OK: Gemini API responde correctamente.")


if __name__ == "__main__":
    main()
