---
title: EcoAsistente Multimodal
emoji: ♻️
colorFrom: green
colorTo: green
sdk: docker
pinned: false
---

# EcoAsistente Multimodal - RAG con Gemini API - HuggingFace

Prototipo Flask para clasificacion contextualizada de residuos mediante una arquitectura RAG multimodal secuencial.

Esta version mantiene el flujo automatico por imagen, pero **no ejecuta modelos localmente**. El analisis de imagen, la generacion textual y los embeddings se realizan mediante Gemini API.

## 1. Arquitectura

```text
Imagen del residuo
  -> optimizacion de imagen
  -> analisis multimodal remoto con Gemini API
  -> extraccion estructurada de atributos
  -> consulta RAG sobre documentos propios
  -> recuperacion vectorial con Chroma
  -> respuesta final fundada con Gemini API
  -> log de trazabilidad
```

No se usan agentes ni multiagentes. El sistema es un pipeline secuencial, modular y auditable.

## 2. Componentes principales

```text
app.py                         interfaz Flask
src/config.py                  configuracion por variables de entorno
src/gemini_client.py           cliente REST para Gemini API
src/image_analyzer.py          analisis automatico de imagen
src/embeddings.py              embeddings remotos para RAG
src/vector_store.py            Chroma persistente
src/rag_engine.py              orquestacion secuencial del flujo
src/response_generator.py      generacion final de respuesta
src/logger.py                  logs de interacciones
templates/index.html           interfaz web
scripts/check_gemini.py        diagnostico de API
```

## 3. Requisitos

- Python 3.10 o superior.
- Conexion a internet.
- API key de Gemini.
- Documentos propios en `data/docs`.

No hace falta instalar Ollama ni descargar modelos locales.

## 4. Obtener API key de Gemini

1. Entrar a Google AI Studio.
2. Crear una API key.
3. Copiarla.
4. Pegarla en el archivo `.env` como `GEMINI_API_KEY`.

No subir la API key a GitHub ni escribirla directamente en el codigo fuente.

## 5. Instalacion

Crear entorno virtual:

```bash
python -m venv .venv
```

Activar en Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Activar en Linux/macOS:

```bash
source .venv/bin/activate
```

Instalar dependencias:

```bash
pip install -r requirements.txt
```

Copiar configuracion:

```powershell
Copy-Item .env.example .env
```

En Linux/macOS:

```bash
cp .env.example .env
```

## 6. Configurar `.env`

Ejemplo:

```env
GEMINI_API_KEY=tu_api_key
GEMINI_API_BASE_URL=https://generativelanguage.googleapis.com/v1beta

GEMINI_VISION_MODEL=gemini-2.5-flash
GEMINI_TEXT_MODEL=gemini-2.5-flash
GEMINI_EMBEDDING_MODEL=gemini-embedding-001
GEMINI_EMBEDDING_DIM=768

DOCS_DIR=data/docs
VECTORSTORE_DIR=data/vectorstore
UPLOAD_DIR=static/uploads
LOG_DIR=logs

CHUNK_SIZE=900
CHUNK_OVERLAP=150
TOP_K=4

FLASK_SECRET_KEY=clave_generada_por_vos

VISION_TIMEOUT_SECONDS=60
TEXT_TIMEOUT_SECONDS=90
EMBEDDING_TIMEOUT_SECONDS=60
```

Generar `FLASK_SECRET_KEY`:

```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

## 7. Verificar Gemini API

Desde la raiz del proyecto:

```bash
python scripts/check_gemini.py
```

El script verifica:

- que `GEMINI_API_KEY` este configurada;
- que el modelo de texto responda;
- que el modelo de embeddings devuelva un vector valido.

## 8. Ejecutar la aplicacion

```bash
python app.py
```

Abrir en navegador:

```text
http://127.0.0.1:5000
```

## 9. Uso de la interfaz desplegada

La aplicación desplegada en Hugging Face utiliza un índice vectorial previamente construido
en `data/vectorstore`. Por ese motivo, la interfaz pública no permite cargar documentos ni
reconstruir el índice dinámicamente.

Flujo de uso:

1. Subir una imagen clara del residuo.
2. Agregar una consulta opcional si el caso requiere contexto adicional.
3. Presionar **Analizar residuo**.
4. Revisar la recomendación final.
5. Opcionalmente, abrir las secciones de trazabilidad para ver análisis visual, consulta RAG,
   fragmentos recuperados, confianza operacional y auditoría automática.

Para modificar la base documental, se debe reconstruir el índice en entorno local y luego
subir nuevamente `data/vectorstore` al despliegue.

El sistema mostrara:

- analisis visual estructurado;
- consulta RAG generada;
- fragmentos documentales recuperados;
- respuesta final;
- ruta del log generado.

## 10. Documentos para RAG

El proyecto incluye un corpus minimo en:

```text
data/docs/corpus_materiales.md
```

Para una entrega final, agregar guias oficiales, documentos GIRSU, normativas locales o instrucciones institucionales en PDF, TXT o Markdown.

Luego reconstruir el indice desde la interfaz.

## 11. Evaluacion sugerida

Usar `tests/casos_prueba.csv` como base y completar pruebas con imagenes reales.

Comparar:

```text
Baseline: LLM general sin documentos propios.
Sistema propuesto: imagen + RAG documental.
```

Criterios:

- clasificacion correcta del residuo;
- adecuacion a la normativa/documentos cargados;
- utilidad practica de la instruccion;
- trazabilidad de fuentes;
- deteccion de ambiguedad;
- tiempo de respuesta.

## 12. Justificacion tecnica

El proyecto no entrena un clasificador visual propio. Utiliza un modelo multimodal preentrenado para extraer atributos desde la imagen. La adaptacion al dominio se realiza mediante RAG sobre documentos propios: guias, normativas y criterios de disposicion.

Esta decision reduce complejidad y evita una arquitectura multiagente innecesaria. La complejidad se concentra en integrar entrada visual, recuperacion documental y generacion fundada.

## 13. Limitaciones

- Requiere conexion a internet.
- Requiere API key de Gemini y queda sujeto a cuota, latencia y politicas del proveedor.
- La calidad del resultado depende de la imagen subida y de la calidad de los documentos indexados.
- Si el documento local no contiene una regla clara, el sistema debe advertir incertidumbre en lugar de inventar normativa.

## 14. Archivos generados durante el uso

```text
static/uploads/       imagenes subidas y versiones optimizadas
data/vectorstore/     base vectorial Chroma
logs/                 logs JSON de interacciones
```

Estos archivos no son necesarios para versionar el codigo fuente.

## Extensiones de evaluación, trazabilidad y auditoría

Esta versión incorpora componentes adicionales sin modificar la arquitectura base del proyecto. El flujo principal sigue siendo secuencial:

```text
Imagen -> Gemini Vision -> RAG documental -> Gemini Text -> Respuesta final
```

Sobre ese flujo se agregan capas transversales:

1. **Confianza operacional** (`src/scoring.py`): calcula una confianza alta, media o baja usando reglas transparentes sobre confianza visual, cantidad de fragmentos recuperados y distancia vectorial.
2. **Evaluador simple de respuesta** (`src/answer_evaluator.py`): realiza una auditoría posterior de la respuesta para detectar riesgo de alucinación, falta de fuentes o necesidad de revisión humana.
3. **Feedback de usuario** (`src/feedback.py`): registra si la respuesta fue correcta, dudosa o incorrecta en `outputs/feedback.csv`.
4. **Evaluación offline** (`scripts/evaluate_cases.py`): ejecuta casos de prueba con imágenes y guarda resultados en `outputs/evaluation_results.csv`.
5. **Comparación contra baseline sin RAG** (`scripts/compare_baseline.py`): compara una respuesta general sin documentos contra el sistema RAG.
6. **Resumen de logs** (`scripts/summarize_logs.py`): consolida métricas de interacciones en `outputs/log_summary.csv`.

### Variables nuevas en `.env`

```env
ENABLE_ANSWER_EVALUATOR=true
OUTPUTS_DIR=outputs
```

Si se quiere ahorrar cuota o reducir latencia, se puede desactivar el evaluador posterior:

```env
ENABLE_ANSWER_EVALUATOR=false
```

### Evaluar casos de prueba

Crear imágenes propias en:

```text
tests/images/
```

Editar `tests/casos_evaluacion.csv` con las rutas reales y luego ejecutar:

```bash
python scripts/evaluate_cases.py --cases tests/casos_evaluacion.csv
```

Para reconstruir el índice antes de evaluar:

```bash
python scripts/evaluate_cases.py --cases tests/casos_evaluacion.csv --reindex
```

### Comparar contra baseline sin RAG

```bash
python scripts/compare_baseline.py --cases tests/casos_evaluacion.csv
```

El resultado queda en:

```text
outputs/baseline_comparison.csv
```

### Resumir logs

```bash
python scripts/summarize_logs.py
```

El resumen queda en:

```text
outputs/log_summary.csv
```

### Cómo defender estas mejoras

El sistema no se convirtió en una arquitectura multiagente. Se mantiene como RAG multimodal secuencial. El evaluador simple es una capa posterior de auditoría que ayuda a justificar la respuesta, detectar posibles alucinaciones y registrar evidencia para el análisis crítico del prototipo.
