"""
vector_store.py
---------------
Gestiona la base vectorial Chroma.

Chroma almacena:
- el texto de cada fragmento documental;
- los metadatos del fragmento;
- el vector numerico asociado al texto.

En una consulta, se calcula el embedding de la pregunta y se recuperan los chunks
mas cercanos. Esos chunks se pasan luego al LLM como contexto.
"""

from __future__ import annotations

from typing import Dict, List, Any
import chromadb
from chromadb.config import Settings as ChromaSettings
from .config import settings
from .document_loader import DocumentChunk, load_documents
from .embeddings import get_embedding_provider


COLLECTION_NAME = "ecoasistente_residuos"


class VectorStore:
    """Capa simple para indexar y consultar documentos en Chroma."""

    def __init__(self) -> None:
        self.client = chromadb.PersistentClient(
            path=settings.vectorstore_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(name=COLLECTION_NAME)
        self.embedding_provider = get_embedding_provider()

    def reset(self) -> None:
        """Borra la coleccion y la vuelve a crear. Util para reconstruir el indice."""
        try:
            self.client.delete_collection(COLLECTION_NAME)
        except Exception:
            # Si no existia, no es un error relevante para el flujo.
            pass
        self.collection = self.client.get_or_create_collection(name=COLLECTION_NAME)

    def index_documents(self, reset: bool = True) -> int:
        """
        Carga documentos desde data/docs, genera embeddings y los guarda en Chroma.

        Devuelve la cantidad de chunks indexados.
        """
        if reset:
            self.reset()

        chunks: List[DocumentChunk] = load_documents(
            docs_dir=settings.docs_dir,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

        if not chunks:
            return 0

        texts = [chunk.text for chunk in chunks]
        embeddings = self.embedding_provider.embed_texts(texts)
        ids = [chunk.chunk_id for chunk in chunks]
        metadatas = [
            {
                "source": chunk.source,
                "page": chunk.page if chunk.page is not None else "",
            }
            for chunk in chunks
        ]

        self.collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        return len(chunks)

    def search(self, query: str, top_k: int | None = None) -> List[Dict[str, Any]]:
        """Recupera los fragmentos mas relevantes para una consulta."""
        top_k = top_k or settings.top_k
        query_embedding = self.embedding_provider.embed_query(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        retrieved: List[Dict[str, Any]] = []
        for text, metadata, distance in zip(documents, metadatas, distances):
            retrieved.append(
                {
                    "text": text,
                    "source": metadata.get("source", "desconocido"),
                    "page": metadata.get("page") or None,
                    "distance": distance,
                }
            )
        return retrieved

    def count(self) -> int:
        """Devuelve la cantidad de chunks indexados."""
        return self.collection.count()
