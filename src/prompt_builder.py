"""
prompt_builder.py
-----------------
Construye prompts para el LLM generador.

Separar los prompts en un modulo propio facilita explicar y auditar como se le
pide al modelo que use la evidencia recuperada por RAG.
"""

from __future__ import annotations

from typing import Dict, List, Any


SYSTEM_PROMPT = """
Sos EcoAsistente, un sistema de apoyo para clasificar residuos urbanos.
Debes responder de forma clara, prudente y fundada en el contexto documental.

Reglas:
1. Usa la informacion visual y los fragmentos recuperados.
2. No inventes normativa local si no aparece en el contexto.
3. Si hay incertidumbre, explicala.
4. Si el residuo podria ser peligroso o especial, recomienda consultar canales oficiales.
5. La respuesta debe ser practica para una persona usuaria no tecnica.
"""


def build_rag_query(visual_analysis: Dict[str, Any], user_question: str | None = None) -> str:
    """
    Convierte el analisis visual en una consulta textual para el retriever.

    Esta funcion es el puente entre la etapa multimodal y la etapa RAG.
    """
    parts = [
        f"Objeto: {visual_analysis.get('objeto', 'desconocido')}",
        f"Materiales probables: {', '.join(visual_analysis.get('materiales_probables', []))}",
        f"Estado: {visual_analysis.get('estado', 'desconocido')}",
        f"Categoria tentativa: {visual_analysis.get('categoria_tentativa', 'desconocida')}",
        "Buscar tratamiento previo, reciclabilidad, disposicion recomendada y excepciones.",
    ]

    if user_question:
        parts.append(f"Pregunta adicional del usuario: {user_question}")

    return "\n".join(parts)


def format_context(retrieved_chunks: List[Dict[str, Any]]) -> str:
    """Formatea los fragmentos recuperados para pasarlos al LLM."""
    if not retrieved_chunks:
        return "No se recuperaron fragmentos documentales relevantes."

    formatted = []
    for i, chunk in enumerate(retrieved_chunks, start=1):
        source = chunk.get("source", "desconocido")
        page = chunk.get("page")
        location = f"{source}, pagina {page}" if page else source
        text = chunk.get("text", "")
        formatted.append(f"[Fragmento {i} | Fuente: {location}]\n{text}")

    return "\n\n".join(formatted)


def build_answer_prompt(
    visual_analysis: Dict[str, Any],
    retrieved_chunks: List[Dict[str, Any]],
    user_question: str | None = None,
) -> str:
    """Arma el prompt final para generar la respuesta al usuario."""
    context = format_context(retrieved_chunks)

    return f"""
Analisis visual estructurado:
{visual_analysis}

Pregunta o contexto adicional del usuario:
{user_question or 'No informado'}

Fragmentos documentales recuperados:
{context}

Genera una respuesta en espanol con esta estructura:
1. Clasificacion probable del residuo.
2. Tratamiento previo recomendado.
3. Disposicion sugerida.
4. Justificacion basada en los fragmentos recuperados.
5. Advertencias o incertidumbres.
6. Fuentes consultadas en formato breve.
"""
