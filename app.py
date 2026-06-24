"""
app.py
------
Interfaz Flask del EcoAsistente Multimodal.

Esta version publica esta pensada para despliegue en Hugging Face:
- permite subir una imagen de un residuo;
- ejecuta el pipeline RAG multimodal automatico;
- muestra una recomendacion clara para el usuario;
- expone detalles tecnicos de trazabilidad para evaluacion academica;
- registra feedback del usuario;
- NO permite reconstruir el indice desde la interfaz.

El indice vectorial debe generarse previamente en entorno local y subirse junto
con el proyecto en data/vectorstore.
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
from src.trace_reader import list_trace_logs, load_trace


ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}


validate_settings()

app = Flask(__name__)
app.secret_key = settings.flask_secret_key

Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
Path(settings.log_dir).mkdir(parents=True, exist_ok=True)
Path(settings.outputs_dir).mkdir(parents=True, exist_ok=True)


def allowed_file(filename: str) -> bool:
    """Valida que el archivo subido sea una imagen soportada."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET", "POST"])
def index():
    """Pantalla principal: carga de imagen, analisis y resultados."""
    result = None
    image_url = None

    try:
        vector_count = VectorStore().count()
    except Exception:
        vector_count = 0

    if request.method == "POST":
        uploaded_file = request.files.get("image")
        user_question = request.form.get("question", "").strip() or None

        if uploaded_file is None or uploaded_file.filename == "":
            flash("Subí una imagen del residuo para poder analizarlo.")
            return redirect(url_for("index"))

        if not allowed_file(uploaded_file.filename):
            flash("Formato no soportado. Usá una imagen PNG, JPG, JPEG o WEBP.")
            return redirect(url_for("index"))

        filename = secure_filename(uploaded_file.filename)
        image_path = os.path.join(settings.upload_dir, filename)
        uploaded_file.save(image_path)

        image_url = url_for("static", filename=f"uploads/{filename}")

        try:
            pipeline = EcoAsistenteRAG()
            result = pipeline.answer(
                image_path=image_path,
                image_filename=filename,
                user_question=user_question,
            )
        except Exception as exc:
            flash(
                "No se pudo completar el análisis. "
                f"Detalle técnico: {exc}"
            )

    return render_template(
        "index.html",
        result=result,
        image_url=image_url,
        vector_count=vector_count,
    )


@app.route("/feedback", methods=["POST"])
def feedback():
    """Registra feedback del usuario sobre la recomendacion generada."""
    rating = request.form.get("rating", "").strip()
    comment = request.form.get("comment", "").strip()
    log_path = request.form.get("log_path", "").strip()
    error_category = request.form.get("error_category", "").strip()

    if rating not in {"correcta", "dudosa", "incorrecta"}:
        flash("La valoración enviada no es válida.")
        return redirect(url_for("index"))

    try:
        save_feedback(
            log_path=log_path,
            rating=rating,
            comment=comment,
            error_category=error_category,
        )
        flash("Gracias. Tu feedback fue registrado para evaluación y mejora del sistema.")
    except Exception as exc:
        flash(f"No se pudo registrar el feedback. Detalle técnico: {exc}")

    return redirect(url_for("index"))


@app.route("/trace", methods=["GET"])
def trace_index():
    """Panel de trazabilidad: lista las ultimas interacciones registradas."""
    traces = list_trace_logs(limit=30)
    return render_template("trace.html", traces=traces)


@app.route("/trace/<log_id>", methods=["GET"])
def trace_detail(log_id: str):
    """Detalle paso a paso de una interaccion registrada."""
    try:
        trace = load_trace(log_id)
    except Exception as exc:
        flash(f"No se pudo cargar la trazabilidad solicitada: {exc}")
        return redirect(url_for("trace_index"))

    return render_template("trace_detail.html", trace=trace)


if __name__ == "__main__":
    app.run(debug=True, port=7860)