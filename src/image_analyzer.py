"""
image_analyzer.py
-----------------
Analiza la imagen subida por el usuario usando Gemini API.

Este modulo NO consulta normativa ni decide la disposicion final. Su funcion es
extraer atributos observables del residuo: objeto, materiales probables, estado,
componentes y dudas.

Version cloud:
- No ejecuta modelos locales.
- Optimiza la imagen antes de enviarla para reducir latencia y consumo de cuota.
- Solicita una salida JSON para que las siguientes etapas del pipeline puedan
  trabajar con datos estructurados.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

from PIL import Image

from .config import settings
from .gemini_client import generate_content


SYSTEM_PROMPT = """
Sos un asistente tecnico para analizar imagenes de residuos urbanos.
Tu tarea es describir atributos observables, no decidir la normativa final.
Devolve exclusivamente JSON valido, sin markdown, sin explicaciones externas.
"""

USER_PROMPT = """
Analiza la imagen del residuo y devolve un JSON con esta estructura exacta:
{
  "objeto": "descripcion breve del objeto",
  "materiales_probables": ["material 1", "material 2"],
  "estado": "limpio, sucio, mojado, roto, con restos, desconocido, etc.",
  "componentes": ["partes distinguibles, por ejemplo tapa, etiqueta, envase"],
  "categoria_tentativa": "reciclable seco / organico / no reciclable / peligroso / desconocido",
  "confianza": "alta / media / baja",
  "observaciones": "dudas o advertencias relevantes"
}
No inventes informacion que no se vea en la imagen.
Si la imagen es ambigua, indicalo en confianza y observaciones.
Usa respuestas breves.
"""

def optimize_image_for_api(image_path: str, max_size: int = 768, quality: int = 75) -> str:
    """
    Reduce y comprime la imagen antes de enviarla a Gemini API.

    Donde impacta:
    - Esta funcion se ejecuta justo antes de enviar la imagen al modelo.
    - No modifica la imagen original subida por el usuario.
    - Crea una copia con sufijo _optimized.jpg en la misma carpeta.

    Motivo:
    - Las fotos de celular pueden pesar varios MB.
    - Reducir resolucion baja latencia y consumo de cuota.
    - Para clasificar un residuo, una imagen de 768 px de lado maximo suele ser
      suficiente para una demo academica.
    """
    original_path = Path(image_path)
    optimized_path = original_path.with_name(f"{original_path.stem}_optimized.jpg")

    with Image.open(original_path) as img:
        img = img.convert("RGB")
        img.thumbnail((max_size, max_size))
        img.save(optimized_path, format="JPEG", quality=quality, optimize=True)

    return str(optimized_path)


def safe_json_loads(text: str) -> Dict[str, Any]:
    """
    Intenta convertir la salida del modelo a JSON.

    Aunque se solicita responseMimeType application/json, se mantiene este parser
    defensivo para tolerar respuestas con caracteres extra en entornos reales.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise ValueError(f"El modelo no devolvio JSON valido. Respuesta: {text[:500]}")


class ImageAnalyzer:
    """Analizador multimodal basado en Gemini API."""

    def analyze(self, image_path: str) -> Dict[str, Any]:
        """
        Envia la imagen optimizada a Gemini.

        Requisitos:
        - GEMINI_API_KEY debe estar configurada en .env.
        - El modelo indicado en GEMINI_VISION_MODEL debe soportar entrada multimodal.
        """
        optimized_path = optimize_image_for_api(image_path)

        content = generate_content(
            model=settings.gemini_vision_model,
            system_instruction=SYSTEM_PROMPT,
            prompt=USER_PROMPT,
            image_path=optimized_path,
            temperature=0.0,
            max_output_tokens=256,
            timeout=settings.vision_timeout_seconds,
            response_mime_type="application/json",
        )
        return safe_json_loads(content)
