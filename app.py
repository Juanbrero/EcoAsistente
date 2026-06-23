"""
app.py
------
Aplicacion Flask del EcoAsistente Multimodal.

Esta es la entrada principal del prototipo. Permite:
- subir una imagen de un residuo;
- ejecutar el pipeline RAG multimodal automatico;
- ver el analisis visual, los fragmentos recuperados y la respuesta final;
- ver confianza operacional y auditoria del evaluador simple;
- registrar feedback del usuario;
- reconstruir el indice vectorial desde la interfaz.

Ejecucion:
    python app.py
Luego abrir:
    http://127.0.0.1:5000
"""

from __future__ import annotations

import os
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from src.config import settings, validate_settings
from src.rag_engine import EcoAsistenteRAG
from src.vector_store import VectorStore
from src.feedback import save_feedback


ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}


validate_settings()

app = Flask(__name__)
app.secret_key = settings.flask_secret_key

# Aseguramos que existan las carpetas necesarias.
Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
Path(settings.docs_dir).mkdir(parents=True, exist_ok=True)
Path(settings.vectorstore_dir).mkdir(parents=True, exist_ok=True)
Path(settings.log_dir).mkdir(parents=True, exist_ok=True)
Path(settings.outputs_dir).mkdir(parents=True, exist_ok=True)


def allowed_file(filename: str) -> bool:
    """Valida que el archivo subido sea una imagen soportada."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET", "POST"])
def index():
    """Pantalla principal: formulario de carga y resultados."""
    result = None
    image_url = None
    vector_count = VectorStore().count()

    if request.method == "POST":
        uploaded_file = request.files.get("image")
        user_question = request.form.get("question", "").strip() or None

        if uploaded_file is None or uploaded_file.filename == "":
            flash("Debe subirse una imagen del residuo.")
            return redirect(url_for("index"))

        if not allowed_file(uploaded_file.filename):
            flash("Formato no soportado. Usar PNG, JPG, JPEG o WEBP.")
            return redirect(url_for("index"))

        filename = secure_filename(uploaded_file.filename)
        image_path = os.path.join(settings.upload_dir, filename)
        uploaded_file.save(image_path)
        image_url = url_for("static", filename=f"uploads/{filename}")

        try:
            # El pipeline ejecuta siempre el modo automatico:
            # imagen -> modelo de vision remoto -> RAG -> respuesta final -> auditoria.
            pipeline = EcoAsistenteRAG()
            result = pipeline.answer(
                image_path=image_path,
                image_filename=filename,
                user_question=user_question,
            )
        except Exception as exc:
            # En una demo academica conviene mostrar un error claro en vez de fallar silenciosamente.
            flash(f"Error durante el analisis: {exc}")

    return render_template(
        "index.html",
        result=result,
        image_url=image_url,
        vector_count=vector_count,
    )


@app.route("/reindex", methods=["POST"])
def reindex():
    """
    Reconstruye el indice vectorial desde los documentos en data/docs.

    Esta accion es util durante la demo: si se agregan o modifican documentos,
    se puede reindexar sin ejecutar scripts externos.
    """
    try:
        vector_store = VectorStore()
        count = vector_store.index_documents(reset=True)
        flash(f"Indice reconstruido correctamente. Chunks indexados: {count}")
    except Exception as exc:
        flash(f"No se pudo reconstruir el indice: {exc}")

    return redirect(url_for("index"))


@app.route("/feedback", methods=["POST"])
def feedback():
    """Registra feedback del usuario sobre la recomendacion generada."""
    rating = request.form.get("rating", "").strip()
    comment = request.form.get("comment", "").strip()
    log_path = request.form.get("log_path", "").strip()

    if rating not in {"correcta", "dudosa", "incorrecta"}:
        flash("Feedback invalido.")
        return redirect(url_for("index"))

    try:
        feedback_path = save_feedback(log_path=log_path, rating=rating, comment=comment)
        flash(f"Feedback registrado correctamente en {feedback_path}")
    except Exception as exc:
        flash(f"No se pudo registrar el feedback: {exc}")

    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True, port=7860)
