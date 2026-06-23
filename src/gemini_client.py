"""
gemini_client.py
----------------
Cliente REST minimo para Gemini API.

Se usa requests en lugar de un SDK para que el proyecto sea facil de auditar:
- generate_content: texto o texto + imagen.
- embed_text: embeddings para documentos y consultas RAG.

Este modulo encapsula los detalles HTTP para que el resto del proyecto no dependa
ni de Ollama ni de un proveedor local.
"""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from .config import settings


class GeminiAPIError(RuntimeError):
    """Error controlado para fallos de Gemini API."""


def _api_url(model: str, method: str) -> str:
    """
    Construye una URL REST de Gemini.

    Ejemplo resultante:
    https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=...
    """
    base_url = settings.gemini_api_base_url.rstrip("/")
    return f"{base_url}/models/{model}:{method}?key={settings.gemini_api_key}"


def _raise_for_gemini_error(response: requests.Response, action: str) -> None:
    """Convierte errores HTTP en mensajes utiles para la interfaz Flask."""
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        detail = response.text[:1500]
        raise GeminiAPIError(
            f"Gemini API fallo durante {action}. "
            f"Status HTTP: {response.status_code}. Detalle: {detail}"
        ) from exc


def _extract_text(data: Dict[str, Any]) -> str:
    """
    Extrae texto desde la respuesta generateContent.

    Gemini puede devolver varias candidates y varias parts; para este prototipo se
    concatena todo el texto disponible de la primera candidate.
    """
    candidates = data.get("candidates", [])
    if not candidates:
        raise GeminiAPIError(f"Gemini no devolvio candidates validos: {data}")

    parts = candidates[0].get("content", {}).get("parts", [])
    texts = [part.get("text", "") for part in parts if part.get("text")]
    content = "\n".join(texts).strip()
    if not content:
        raise GeminiAPIError(f"Gemini no devolvio texto util: {data}")
    return content


def encode_image_inline_data(image_path: str) -> Dict[str, Any]:
    """
    Convierte una imagen local al formato inline_data de Gemini.

    Gemini acepta imagenes como datos inline para archivos pequenos. En este
    proyecto antes se optimiza la imagen, por lo que suele quedar muy por debajo
    del limite practico de request.
    """
    path = Path(image_path)
    mime_type, _ = mimetypes.guess_type(path.name)
    if mime_type is None:
        mime_type = "image/jpeg"

    with open(path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode("utf-8")

    return {
        "inline_data": {
            "mime_type": mime_type,
            "data": encoded,
        }
    }


def generate_content(
    *,
    model: str,
    prompt: str,
    system_instruction: Optional[str] = None,
    image_path: Optional[str] = None,
    temperature: float = 0.0,
    max_output_tokens: int = 1024,
    timeout: int = 60,
    response_mime_type: Optional[str] = None,
) -> str:
    """
    Ejecuta generateContent de Gemini para texto o texto + imagen.

    Parametros importantes:
    - image_path: si se informa, la request es multimodal.
    - response_mime_type='application/json': fuerza salida JSON cuando corresponde.
    - max_output_tokens: acota costo, latencia y longitud de respuesta.
    """
    parts: List[Dict[str, Any]] = [{"text": prompt}]
    if image_path:
        parts.append(encode_image_inline_data(image_path))

    payload: Dict[str, Any] = {
        "contents": [
            {
                "role": "user",
                "parts": parts,
            }
        ],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_output_tokens,

            # Gemini 2.5 Flash puede consumir presupuesto de salida en
            # tokens internos de thinking y no devolver texto visible si el
            # limite es bajo. Para este prototipo necesitamos respuestas
            # directas y auditables, por eso se desactiva thinking.
            "thinkingConfig": {"thinkingBudget": 0},
        },
    }

    if system_instruction:
        payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}

    if response_mime_type:
        payload["generationConfig"]["responseMimeType"] = response_mime_type

    response = requests.post(
        _api_url(model, "generateContent"),
        json=payload,
        timeout=timeout,
    )
    _raise_for_gemini_error(response, "generateContent")
    return _extract_text(response.json())


def embed_text(text: str, task_type: str, timeout: int) -> List[float]:
    """
    Genera un embedding para un texto individual.

    task_type se usa para distinguir documentos de consultas:
    - RETRIEVAL_DOCUMENT durante indexacion.
    - RETRIEVAL_QUERY durante busqueda.
    """
    payload = {
        "content": {"parts": [{"text": text}]},
        "taskType": task_type,
        "outputDimensionality": settings.gemini_embedding_dim,
    }

    response = requests.post(
        _api_url(settings.gemini_embedding_model, "embedContent"),
        json=payload,
        timeout=timeout,
    )
    _raise_for_gemini_error(response, "embedContent")

    data = response.json()
    values = data.get("embedding", {}).get("values")
    if not values:
        raise GeminiAPIError(f"Gemini no devolvio embedding valido: {data}")
    return values
