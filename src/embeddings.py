"""
embeddings.py
-------------
Genera embeddings remotos usando Gemini API.

Los embeddings convierten texto en vectores numericos. Chroma usa esos vectores
para encontrar fragmentos semanticamente parecidos a una consulta. Todas las representaciones vectoriales
se solicitan a Gemini API.
"""

from __future__ import annotations

from typing import List

from .config import settings
from .gemini_client import embed_text


class GeminiEmbeddingProvider:
    """Genera embeddings con Gemini API."""

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Genera embeddings para una lista de textos documentales.

        Se procesan de a uno para mantener el codigo simple y explicable. Para
        corpus pequenos, como el de este prototipo, esto es suficiente. Si el
        corpus creciera mucho, podria reemplazarse por un batch endpoint.
        """
        embeddings: List[List[float]] = []
        for text in texts:
            embeddings.append(
                embed_text(
                    text,
                    task_type="RETRIEVAL_DOCUMENT",
                    timeout=settings.embedding_timeout_seconds,
                )
            )
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Genera el embedding de una consulta individual."""
        return embed_text(
            text,
            task_type="RETRIEVAL_QUERY",
            timeout=settings.embedding_timeout_seconds,
        )


def get_embedding_provider() -> GeminiEmbeddingProvider:
    """Factory simple"""
    return GeminiEmbeddingProvider()
