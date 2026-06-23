"""
feedback.py
-----------
Registro simple de retroalimentacion del usuario.

Este modulo permite guardar si la persona usuaria considero util, incorrecta o
dudosa la recomendacion. No se usa para entrenar modelos; sirve como evidencia
para analisis de errores y mejoras futuras.
"""

from __future__ import annotations

import csv
import os
from datetime import datetime
from .config import settings


FEEDBACK_FILE = "feedback.csv"


def save_feedback(*, log_path: str, rating: str, comment: str | None = None) -> str:
    """Agrega una fila de feedback y devuelve la ruta del CSV."""
    os.makedirs(settings.outputs_dir, exist_ok=True)
    path = os.path.join(settings.outputs_dir, FEEDBACK_FILE)
    file_exists = os.path.exists(path)

    with open(path, "a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["timestamp", "log_path", "rating", "comment"],
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "log_path": log_path,
                "rating": rating,
                "comment": comment or "",
            }
        )
    return path
