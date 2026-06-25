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
import os
import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
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

        La indexación se realiza por lotes para evitar timeouts, cortes por cuota
        y para mostrar progreso durante la construcción offline del índice.
        """
        if reset:
            self.reset()

        chunks: List[DocumentChunk] = load_documents(
            docs_dir=settings.docs_dir,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

        if not chunks:
            print("No se encontraron chunks para indexar.")
            return 0

        batch_size = int(os.getenv("INDEX_BATCH_SIZE", "10"))
        indexed_total = 0

        print(f"Chunks detectados: {len(chunks)}")
        print(f"Tamaño de lote: {batch_size}")

        for start in range(0, len(chunks), batch_size):
            batch = chunks[start:start + batch_size]

            print(f"Generando embeddings para chunks {start + 1}-{start + len(batch)} de {len(chunks)}...")

            texts = [chunk.text for chunk in batch]
            embeddings = self.embedding_provider.embed_texts(texts)

            ids = [chunk.chunk_id for chunk in batch]
            metadatas = [
                {
                    "source": chunk.source,
                    "page": chunk.page if chunk.page is not None else "",
                }
                for chunk in batch
            ]

            self.collection.add(
                ids=ids,
                documents=texts,
                metadatas=metadatas,
                embeddings=embeddings,
            )

            indexed_total += len(batch)
            print(f"Indexados {indexed_total}/{len(chunks)} chunks")

        return indexed_total

    def _lexical_search(self, query: str, top_k: int):
        chunks_path = Path("data/document_chunks.json")

        if not chunks_path.exists():
            return []

        with chunks_path.open("r", encoding="utf-8") as f:
            chunks = json.load(f)

        query_terms = {
            term.lower()
            for term in re.findall(r"\w+", query)
            if len(term) > 2
        }

        scored = []

        for chunk in chunks:
            text = chunk.get("text", "")
            text_lower = text.lower()

            score = sum(1 for term in query_terms if term in text_lower)

            if score > 0:
                scored.append(
                    {
                        "text": text,
                        "source": chunk.get("source", "document_chunks.json"),
                        "page": chunk.get("page", ""),
                        "score": float(score),
                    }
                )

        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[:top_k]

    def search(self, query: str, top_k: Optional[int] = None):
        top_k = top_k or settings.top_k

        try:
            if self.collection.count() == 0:
                return self._lexical_search(query, top_k)

            query_embedding = self.embedding_provider.embed_query(query)

            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )

            documents = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]

            retrieved = []

            for text, metadata, distance in zip(documents, metadatas, distances):
                retrieved.append(
                    {
                        "text": text,
                        "source": metadata.get("source", ""),
                        "page": metadata.get("page", ""),
                        "score": float(distance),
                    }
                )

            return retrieved

        except Exception as exc:
            print(f"Advertencia: falló Chroma/embeddings. Usando fallback lexical. Detalle: {exc}")
            return self._lexical_search(query, top_k)

    def count(self) -> int:
        """Devuelve la cantidad de chunks indexados."""
        return self.collection.count()
