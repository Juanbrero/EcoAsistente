"""
rag_engine.py
-------------
Orquesta el flujo secuencial del sistema.

Flujo automatico:
1. Analizar imagen con modelo multimodal remoto.
2. Convertir el analisis visual en una consulta RAG.
3. Recuperar fragmentos documentales relevantes.
4. Generar una respuesta final fundada en documentos.
5. Calcular confianza operacional.
6. Auditar la respuesta con un evaluador simple opcional.
7. Registrar la interaccion para evaluacion y trazabilidad.

Este modulo sigue siendo un pipeline secuencial, modular y auditable. El
evaluador simple se ejecuta al final como control de calidad, no como orquestador.
"""

from __future__ import annotations

import time
from typing import Dict, Any

from .config import settings
from .image_analyzer import ImageAnalyzer
from .vector_store import VectorStore
from .prompt_builder import build_rag_query, build_answer_prompt
from .response_generator import ResponseGenerator
from .logger import save_interaction_log
from .scoring import compute_confidence_score
from .answer_evaluator import AnswerEvaluator


class EcoAsistenteRAG:
    """Pipeline principal del EcoAsistente."""

    def __init__(self) -> None:
        self.image_analyzer = ImageAnalyzer()
        self.vector_store = VectorStore()
        self.response_generator = ResponseGenerator()
        self.answer_evaluator = AnswerEvaluator()

    def answer(self, image_path: str, image_filename: str, user_question: str | None = None) -> Dict[str, Any]:
        """
        Ejecuta el flujo completo usando analisis visual automatico.

        Entrada:
        - image_path: ruta local de la imagen subida.
        - image_filename: nombre de archivo usado para registrar logs.
        - user_question: consulta o contexto adicional opcional del usuario.

        Salida:
        - diccionario con analisis visual, consulta RAG, fragmentos recuperados,
          respuesta final, confianza, evaluacion y ruta del log.
        """
        timings: Dict[str, float] = {}
        total_start = time.perf_counter()

        # 1. Analisis multimodal: extrae atributos observables del residuo.
        start = time.perf_counter()
        visual_analysis = self.image_analyzer.analyze(image_path)
        timings["image_analysis_seconds"] = round(time.perf_counter() - start, 3)

        # 2. Query enriquecida para recuperar normativa o guias relevantes.
        rag_query = build_rag_query(visual_analysis, user_question)

        # 3. Busqueda vectorial sobre la base documental del proyecto.
        start = time.perf_counter()
        retrieved_chunks = self.vector_store.search(rag_query)
        timings["retrieval_seconds"] = round(time.perf_counter() - start, 3)

        # 4. Prompt final: combina analisis visual + documentos recuperados.
        answer_prompt = build_answer_prompt(visual_analysis, retrieved_chunks, user_question)

        # 5. Generacion textual final con modelo remoto de Gemini.
        start = time.perf_counter()
        final_answer = self.response_generator.generate(answer_prompt)
        timings["generation_seconds"] = round(time.perf_counter() - start, 3)

        # 6. Confianza operacional basada en reglas transparentes.
        confidence = compute_confidence_score(
            visual_analysis=visual_analysis,
            retrieved_chunks=retrieved_chunks,
            final_answer=final_answer,
        )

        # 7. Evaluador simple opcional. Suma una llamada a Gemini, por eso se
        # deja configurable en .env.
        answer_evaluation: Dict[str, Any] = {}
        if settings.enable_answer_evaluator:
            start = time.perf_counter()
            answer_evaluation = self.answer_evaluator.evaluate(
                visual_analysis=visual_analysis,
                retrieved_chunks=retrieved_chunks,
                final_answer=final_answer,
            )
            timings["answer_evaluation_seconds"] = round(time.perf_counter() - start, 3)

        timings["total_seconds"] = round(time.perf_counter() - total_start, 3)

        # 8. Registro para trazabilidad y evaluacion posterior.
        log_path = save_interaction_log(
            image_filename=image_filename,
            user_question=user_question,
            visual_analysis=visual_analysis,
            rag_query=rag_query,
            retrieved_chunks=retrieved_chunks,
            final_answer=final_answer,
            confidence=confidence,
            answer_evaluation=answer_evaluation,
            timings=timings,
        )

        return {
            "visual_analysis": visual_analysis,
            "rag_query": rag_query,
            "retrieved_chunks": retrieved_chunks,
            "final_answer": final_answer,
            "confidence": confidence,
            "answer_evaluation": answer_evaluation,
            "timings": timings,
            "log_path": log_path,
        }
