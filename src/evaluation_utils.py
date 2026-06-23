"""
evaluation_utils.py
-------------------
Funciones auxiliares para evaluaciones offline.

Se mantienen separadas de Flask para que el trabajo pueda ejecutarse como
prototipo interactivo y tambien como experimento reproducible con casos de prueba.
"""

from __future__ import annotations

import csv
import os
from typing import Dict, Iterable, List


def read_cases_csv(path: str) -> List[Dict[str, str]]:
    """Lee casos de evaluacion desde CSV y devuelve una lista de diccionarios."""
    with open(path, "r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_rows_csv(path: str, rows: Iterable[Dict[str, object]]) -> None:
    """Escribe resultados en CSV creando la carpeta si hace falta."""
    rows = list(rows)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
