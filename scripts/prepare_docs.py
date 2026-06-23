import json
import os
from pathlib import Path
import sys

# Permite importar desde la carpeta raiz del proyecto.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.document_loader import load_documents

if __name__ == "__main__":
    docs_dir = PROJECT_ROOT / "data" / "docs"
    chunks = load_documents(docs_dir, chunk_size=900, chunk_overlap=150)

    out = []
    for c in chunks:
        out.append({
            "chunk_id": c.chunk_id,
            "text": c.text,
            "source": c.source,
            "page": c.page,
        })

    output_dir = PROJECT_ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "document_chunks.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"Chunks generados: {len(out)} -> {output_file}")