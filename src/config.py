"""
config.py
---------
Centraliza la configuracion de la aplicacion.

Version Gemini API:
- No ejecuta modelos localmente.
- Usa una API key de Gemini para vision, generacion de texto y embeddings.
- Mantiene la misma arquitectura del proyecto: Flask + analisis de imagen + RAG + respuesta final.
- Las variables se leen desde .env para poder cambiar modelos, rutas y parametros sin modificar codigo.
"""

import os
from dataclasses import dataclass
from dotenv import load_dotenv


# Carga variables desde un archivo .env si existe.
load_dotenv()


@dataclass
class Settings:
    """Agrupa toda la configuracion necesaria del prototipo."""

    # API key de Gemini. Se obtiene desde Google AI Studio.
    # No debe escribirse dentro del codigo fuente ni subirse a repositorios publicos.
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

    # Endpoint REST de Gemini API.
    # Se deja configurable por si Google cambia version o si se usa un entorno proxy.
    gemini_api_base_url: str = os.getenv(
        "GEMINI_API_BASE_URL",
        "https://generativelanguage.googleapis.com/v1beta",
    )

    # Modelo multimodal usado para analizar la imagen y para generar la respuesta final.
    # Gemini acepta imagen + texto como entrada en generateContent.
    gemini_vision_model: str = os.getenv("GEMINI_VISION_MODEL", "gemini-2.5-flash")
    gemini_text_model: str = os.getenv("GEMINI_TEXT_MODEL", "gemini-2.5-flash")

    # Modelo de embeddings para RAG.
    # gemini-embedding-001 es un modelo estable para busqueda semantica/document retrieval.
    gemini_embedding_model: str = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")

    # Dimension de salida de embeddings. Usar una dimension fija evita inconsistencias en Chroma.
    gemini_embedding_dim: int = int(os.getenv("GEMINI_EMBEDDING_DIM", "768"))

    # Rutas del proyecto.
    docs_dir: str = os.getenv("DOCS_DIR", "data/docs")
    vectorstore_dir: str = os.getenv("VECTORSTORE_DIR", "data/vectorstore")
    upload_dir: str = os.getenv("UPLOAD_DIR", "static/uploads")
    log_dir: str = os.getenv("LOG_DIR", "logs")

    # Parametros RAG.
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "900"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))
    top_k: int = int(os.getenv("TOP_K", "4"))

    # Flask utiliza esta clave para firmar sesiones y mensajes internos.
    # No es una API key externa. Debe ser una cadena aleatoria propia.
    flask_secret_key: str = os.environ.get("FLASK_SECRET_KEY")

    # Timeouts para llamadas remotas a Gemini API.
    vision_timeout_seconds: int = int(os.getenv("VISION_TIMEOUT_SECONDS", "60"))
    text_timeout_seconds: int = int(os.getenv("TEXT_TIMEOUT_SECONDS", "90"))
    embedding_timeout_seconds: int = int(os.getenv("EMBEDDING_TIMEOUT_SECONDS", "60"))

    # Componentes de evaluacion/auditoria.
    # ENABLE_ANSWER_EVALUATOR agrega una llamada extra al modelo para auditar la respuesta.
    enable_answer_evaluator: bool = os.getenv("ENABLE_ANSWER_EVALUATOR", "true").lower() == "true"

    # Carpeta donde se guardan resultados de evaluaciones offline y feedback.
    outputs_dir: str = os.getenv("OUTPUTS_DIR", "outputs")


def validate_settings() -> None:
    """
    Valida configuraciones obligatorias al iniciar la aplicacion.

    Se falla temprano con mensajes claros para evitar errores confusos durante la demo.
    """
    missing = []
    if not settings.gemini_api_key:
        missing.append("GEMINI_API_KEY")
    if not settings.flask_secret_key:
        missing.append("FLASK_SECRET_KEY")

    if missing:
        raise RuntimeError(
            "Faltan variables obligatorias en .env: "
            + ", ".join(missing)
            + ". Revisar .env.example."
        )


settings = Settings()
