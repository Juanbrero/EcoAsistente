"""
document_loader.py
------------------
Carga y segmenta documentos para alimentar la base vectorial.

El sistema RAG no consulta directamente archivos completos: primero divide los
textos en fragmentos relativamente pequenos llamados chunks. Cada chunk conserva
metadatos basicos, como el nombre del archivo fuente y un identificador.

Formatos soportados en este prototipo:
- .txt
- .md
- .pdf

"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List
from pypdf import PdfReader
from pypdf.errors import PdfReadError


@dataclass
class DocumentChunk:
    """Representa un fragmento indexable de un documento."""

    chunk_id: str
    text: str
    source: str
    page: int | None = None


def read_text_file(path: str) -> str:
    """Lee archivos de texto plano usando UTF-8."""
    with open(path, "r", encoding="utf-8") as file:
        return file.read()


def read_pdf_file(path: str) -> List[tuple[int, str]]:
    """
    Extrae texto de un PDF.

    Devuelve una lista de pares (numero_de_pagina, texto). Se conserva la pagina
    para poder mostrar trazabilidad en la respuesta final.
    Si el PDF no puede leerse, devuelve una lista vacia y no interrumpe el flujo.
    """
    try:
        reader = PdfReader(path, strict=False)
    except PdfReadError as exc:
        print(f"Advertencia: no se pudo leer el PDF '{path}': {exc}")
        return []
    except Exception as exc:
        print(f"Advertencia: error inesperado al leer '{path}': {exc}")
        return []

    pages: List[tuple[int, str]] = []

    for index, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:
            print(f"Advertencia: error extrayendo texto de '{path}' pagina {index}: {exc}")
            continue

        if text.strip():
            pages.append((index, text))

    return pages


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """
    Divide texto en chunks con solapamiento.

    El solapamiento evita que una idea quede cortada entre dos fragmentos. Por
    ejemplo, si una norma explica una excepcion justo al final de un chunk, el
    siguiente chunk conserva parte del contexto anterior.
    """
    clean_text = " ".join(text.split())
    if not clean_text:
        return []

    chunks: List[str] = []
    start = 0

    while start < len(clean_text):
        end = start + chunk_size
        chunks.append(clean_text[start:end])

        # Si llegamos al final, cortamos el bucle.
        if end >= len(clean_text):
            break

        # Retrocedemos chunk_overlap caracteres para preservar contexto.
        start = max(end - chunk_overlap, 0)

    return chunks


def load_documents(docs_dir: str, chunk_size: int, chunk_overlap: int) -> List[DocumentChunk]:
    """
    Recorre la carpeta de documentos y devuelve todos los chunks indexables.
    """
    chunks: List[DocumentChunk] = []

    if not os.path.isdir(docs_dir):
        raise FileNotFoundError(f"No existe la carpeta de documentos: {docs_dir}")

    for filename in sorted(os.listdir(docs_dir)):
        path = os.path.join(docs_dir, filename)
        if not os.path.isfile(path):
            continue

        extension = os.path.splitext(filename)[1].lower()

        if extension in {".txt", ".md"}:
            text = read_text_file(path)
            for i, chunk_text in enumerate(split_text(text, chunk_size, chunk_overlap), start=1):
                chunks.append(
                    DocumentChunk(
                        chunk_id=f"{filename}::chunk-{i}",
                        text=chunk_text,
                        source=filename,
                        page=None,
                    )
                )

        elif extension == ".pdf":
            pdf_pages = read_pdf_file(path)
            if not pdf_pages:
                print(f"Advertencia: PDF ignorado por fallo de lectura: {filename}")
                continue
            for page_number, page_text in pdf_pages:
                for i, chunk_text in enumerate(split_text(page_text, chunk_size, chunk_overlap), start=1):
                    chunks.append(
                        DocumentChunk(
                            chunk_id=f"{filename}::page-{page_number}::chunk-{i}",
                            text=chunk_text,
                            source=filename,
                            page=page_number,
                        )
                    )

    return chunks
