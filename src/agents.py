# ============================================================
# Multi-Agent Research Lab — Version estable para Google Colab
# Compatible con: HF Inference API (sin inputs/prompt), DuckDuckGo
# ============================================================

import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper


# ==========================
# Cargar token HF
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
# 1. AGENTE INVESTIGADOR (Search)
# ======================================================

class Investigador:
    def __init__(self, top_k=5):
        self.top_k = top_k
        self.search = DuckDuckGoSearchAPIWrapper()   # Wrapper correcto

    def buscar(self, query):
        """
        Realiza búsqueda web y devuelve títulos + snippets.
        """
        try:
            resultados = self.search.results(query, max_results=self.top_k)
            textos = []

            for r in resultados:
                titulo = r.get("title", "")
                cuerpo = r.get("body", "")
                textos.append(f"{titulo}\n{cuerpo}\n")

            return "\n".join(textos)

        except Exception as e:
            return f"Error en búsqueda: {e}"



# ======================================================
# 2. AGENTE REDACTOR — RESUMEN (HF API)
# ======================================================

class Redactor:
    def __init__(self, modelo="facebook/bart-large-cnn"):
        self.modelo = modelo
        self.client = InferenceClient(token=leer_token())

    def generar_resumen(self, texto):
        """
        Resumen usando HuggingFace Inference API.
        Importante: NO usamos 'inputs', 'prompt', etc.
        """
        try:
            result = self.client.summarization(
                model=self.modelo,
                text=texto
            )

            # El API devuelve un dict
            if isinstance(result, dict) and "summary_text" in result:
                return result["summary_text"]

            return str(result)

        except Exception as e:
            return f"Error en generación de resumen: {e}"



# ======================================================
# 3. AGENTE REVISOR
# ======================================================

class Revisor:
    def __init__(self):
        pass

    def evaluar_texto(self, texto):
        """
        Evaluación simple simulada.
        """
        evaluacion = (
            "• El texto presenta una estructura clara y mantiene coherencia general.\n"
            "• Se recomienda reforzar el tono académico con transiciones más formales.\n"
            "• Añadir ejemplos concretos que conecten aplicaciones con desafíos éticos.\n"
            "• Sugerencia: incluir citas/links de respaldo y aclarar limitaciones metodológicas."
        )
        return evaluacion



# ======================================================
# 4. COORDINADOR — ORQUESTA TODO EL FLUJO
# ======================================================

class Coordinator:
    def __init__(self, investigador, redactor, revisor):
        self.investigador = investigador
        self.redactor = redactor
        self.revisor = revisor

    def run(self, tema, top_k=5):
        # 1. BÚSQUEDA
        fuentes = self.investigador.buscar(tema)

        # 2. PRIMER BORRADOR
        draft = self.redactor.generar_resumen(fuentes)

        # 3. REVISIÓN
        review = self.revisor.evaluar_texto(draft)

        # 4. TEXTO FINAL
        final = (
            f"{draft}\n\n"
            "### Ajustes del revisor:\n"
            f"{review}\n"
        )

        return {
            "sources": fuentes,
            "draft": draft,
            "review": review,
            "final": final
        }
