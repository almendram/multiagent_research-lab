# ============================================================
# Multi-Agent Research Lab (HF Inference / DuckDuckGo / Orquestación)
# ============================================================

import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper


# ==========================
# Cargar token HF desde .env
# ==========================

load_dotenv()

def leer_token():
    token = os.getenv("HF_TOKEN")
    if token is None:
        raise ValueError(
            "❌ ERROR: No se encontró HF_TOKEN en .env.\n"
            "Crea un archivo .env con:\nHF_TOKEN=tu_token_aqui"
        )
    return token


# ======================================================
# 1. AGENTE INVESTIGADOR
# ======================================================

class Investigador:
    """
    Realiza búsquedas web usando DuckDuckGo.
    """
    def __init__(self, top_k=5):
        self.search = DuckDuckGoSearchAPIWrapper()  # API funcional
        self.top_k = top_k

    def buscar(self, query):
        try:
            resultados = self.search.results(query, max_results=self.top_k)
            textos = []

            for r in resultados:
                titulo = r.get("title", "")
                snippet = r.get("body", "")
                textos.append(f"📌 {titulo}\n{snippet}\n")

            return "\n".join(textos)

        except Exception as e:
            return f"Error en búsqueda: {e}"


# ======================================================
# 2. AGENTE REDACTOR — Modelo HF via text_generation
# ======================================================

class Redactor:
    """
    Genera texto usando InferenceClient (HF).
    """
    def __init__(self, modelo="meta-llama/Llama-3.1-8B-Instruct"):
        self.modelo = modelo
        self.client = InferenceClient(token=leer_token())

    def generar_resumen(self, texto):
        """
        Genera un resumen o narrativa usando text_generation() actualizado.
        """
        try:
            respuesta = self.client.text_generation(
                model=self.modelo,
                prompt=(
                    "Eres un investigador académico. "
                    "Resume los siguientes hallazgos en un estilo claro, conciso y profesional:\n\n"
                    f"{texto}\n\nResumen:"
                ),
                max_new_tokens=450,
                temperature=0.5,
            )

            return respuesta

        except Exception as e:
            return f"Error en generación: {e}"


# ======================================================
# 3. AGENTE REVISOR — Simula revisión académica
# ======================================================

class Revisor:
    """
    Genera retroalimentación del texto.
    """
    def __init__(self):
        pass

    def evaluar_texto(self, texto):
        evaluacion = (
            "• El resumen presenta coherencia general y sigue una estructura clara.\n"
            "• Se sugiere fortalecer el tono académico usando transiciones formales.\n"
            "• Podrías incluir ejemplos concretos para ilustrar puntos clave.\n"
            "• Incluye limitaciones o vacíos en la literatura para mayor solidez.\n"
        )
        return evaluacion


# ======================================================
# 4. COORDINADOR — Orquesta el flujo
# ======================================================

class Coordinator:
    def __init__(self, investigador, redactor, revisor):
        self.investigador = investigador
        self.redactor = redactor
        self.revisor = revisor

    def run(self, tema, top_k=5):
        # 1) BÚSQUEDA
        fuentes = self.investigador.buscar(tema)

        # 2) BORRADOR
        draft = self.redactor.generar_resumen(fuentes)

        # 3) REVISIÓN
        review = self.revisor.evaluar_texto(draft)

        # 4) FINAL
        final = (
            f"{draft}\n\n"
            "### Ajustes propuestos por el revisor:\n"
            f"{review}\n"
        )

        return {
            "sources": fuentes,
            "draft": draft,
            "review": review,
            "final": final
        }
