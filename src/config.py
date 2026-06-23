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

# 🔍 BLOQUE DE DIAGNÓSTICO SEGURO (Solo lista los nombres, nunca los valores)
print("\n=== [DIAGNÓSTICO] VARIABLES DETECTADAS POR DOCKER ===", flush=True)
for key in sorted(os.environ.keys()):
    if "GEMINI" in key or "SECRET" in key or "KEY" in key:
        print(f"-> Clave detectada en el OS: '{key}' (Largo del valor: {len(os.environ[key])})", flush=True)
print("====================================================\n", flush=True)

# 1. 🔥 CAPTURA PREVIA DEL ENTORNO REAL (Hugging Face Secrets)
# Guardamos los tokens reales inyectados por la nube antes de que ocurra cualquier lectura de archivos.
hf_gemini_key = os.environ.get("GEMINI_API_KEY")
print("prueba")
print(hf_gemini_key[:3])
hf_flask_key = os.environ.get("FLASK_SECRET_KEY")
print(hf_flask_key[:3])

# 2. CARGA DEL ARCHIVO .env
# Esto carga de forma segura todos los modelos, rutas, tamaños de chunks y solapamientos.
load_dotenv()


@dataclass
class Settings:
    """Agrupa toda la configuracion necesaria del prototipo."""

    # 3. PRIORIDAD ABSOLUTA DE CREDENCIALES
    # Si Hugging Face inyectó el token en el sistema operativo, usamos ese (hf_gemini_key).
    # Si no (entorno local), usamos lo que haya leído load_dotenv() del archivo .env.
    gemini_api_key: str = (hf_gemini_key or os.environ.get("GEMINI_API_KEY") or "").strip()
    flask_secret_key: str = (hf_flask_key or os.environ.get("FLASK_SECRET_KEY") or "").strip()

    # 4. CONFIGURACIONES GENERALES (Se leen del .env o usan el fallback de la derecha)
    gemini_api_base_url: str = os.getenv(
        "GEMINI_API_BASE_URL",
        "https://generativelanguage.googleapis.com/v1beta",
    )
    gemini_vision_model: str = os.getenv("GEMINI_VISION_MODEL", "gemini-2.5-flash")
    gemini_text_model: str = os.getenv("GEMINI_TEXT_MODEL", "gemini-2.5-flash")
    gemini_embedding_model: str = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
    gemini_embedding_dim: int = int(os.getenv("GEMINI_EMBEDDING_DIM", "768"))

    # Rutas del proyecto
    docs_dir: str = os.getenv("DOCS_DIR", "data/docs")
    vectorstore_dir: str = os.getenv("VECTORSTORE_DIR", "data/vectorstore")
    upload_dir: str = os.getenv("UPLOAD_DIR", "static/uploads")
    log_dir: str = os.getenv("LOG_DIR", "logs")

    # Parametros RAG
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "900"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))
    top_k: int = int(os.getenv("TOP_K", "4"))

    # Timeouts y componentes de auditoría
    vision_timeout_seconds: int = int(os.getenv("VISION_TIMEOUT_SECONDS", "60"))
    text_timeout_seconds: int = int(os.getenv("TEXT_TIMEOUT_SECONDS", "90"))
    embedding_timeout_seconds: int = int(os.getenv("EMBEDDING_TIMEOUT_SECONDS", "60"))
    enable_answer_evaluator: bool = os.getenv("ENABLE_ANSWER_EVALUATOR", "true").lower() == "true"
    outputs_dir: str = os.getenv("OUTPUTS_DIR", "outputs")


def validate_settings() -> None:
    """Valida configuraciones obligatorias al iniciar la aplicacion."""
    missing = []
    if not settings.gemini_api_key:
        missing.append("GEMINI_API_KEY")
    if not settings.flask_secret_key:
        missing.append("FLASK_SECRET_KEY")

    if missing:
        raise RuntimeError(
            "Faltan variables obligatorias o están vacías: "
            + ", ".join(missing)
            + ". Verificar los Secrets en la configuración de Hugging Face o el archivo .env."
        )


settings = Settings()