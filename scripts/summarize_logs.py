"""
scripts/summarize_logs.py
-------------------------
Genera un resumen simple de los logs de interacciones.

Uso:
    python scripts/summarize_logs.py

Salida:
    outputs/log_summary.csv
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings


def main() -> None:
    os.makedirs(settings.outputs_dir, exist_ok=True)
    rows = []

    for filename in sorted(os.listdir(settings.log_dir)):
        if not filename.endswith(".json"):
            continue
        path = os.path.join(settings.log_dir, filename)
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)

        rows.append(
            {
                "log_file": filename,
                "timestamp": data.get("timestamp", ""),
                "image_filename": data.get("image_filename", ""),
                "visual_object": data.get("visual_analysis", {}).get("objeto", ""),
                "visual_category": data.get("visual_analysis", {}).get("categoria_tentativa", ""),
                "confidence_label": data.get("confidence", {}).get("label", ""),
                "risk_hallucination": data.get("answer_evaluation", {}).get("riesgo_alucinacion", ""),
                "requires_human_review": data.get("answer_evaluation", {}).get("requiere_revision_humana", ""),
                "retrieved_chunks_count": len(data.get("retrieved_chunks", [])),
                "total_seconds": data.get("timings", {}).get("total_seconds", ""),
            }
        )

    output_path = os.path.join(settings.outputs_dir, "log_summary.csv")
    if not rows:
        print("No hay logs para resumir.")
        return

    with open(output_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Resumen guardado en: {output_path}")


if __name__ == "__main__":
    main()
