# ============================================================
# Multi-Agent Research Lab (CrewAI / LangChain / HF Inference)
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
            "❌ ERROR: No se encontró HF_TOKEN en .env. "
            "Asegúrate de crear un archivo .env con:\nHF_TOKEN=tu_token_aqui"
        )
    return token


# ======================================================
# 1. AGENTE INVESTIGADOR
# ======================================================

class Investigador:
    def __init__(self, top_k=5):
        self.search = DuckDuckGoSearchAPIWrapper()   # Nueva API funcional
        self.top_k = top_k

    def buscar(self, query):
        """
        Realiza búsqueda web y devuelve títulos + snippets.
        """
        try:
            resultados = self.search.results(query, max_results=self.top_k)
            textos = []
            for r in resultados:
                titulo = r.get("title", "")
                snippet = r.get("body", "")
                textos.append(f"{titulo}\n{snippet}\n")

            return "\n".join(textos)

        except Exception as e:
            return f"Error en búsqueda: {e}"


# ======================================================
# 2. AGENTE REDACTOR (Modelo HF via API)
# ======================================================

class Redactor:
    def __init__(self, modelo="facebook/bart-large-cnn"):
        self.modelo = modelo
        self.client = InferenceClient(token=leer_token())

    def generar_resumen(self, texto):
        """
        Crea un resumen usando el modelo HF.
        """
        try:
            result = self.client.summarization(
                model=self.modelo,
                text=texto
            )

            # Formato puede variar según el modelo
            if isinstance(result, dict) and "summary_text" in result:
                return result["summary_text"]

            if hasattr(result, "summary_text"):
                return result.summary_text

            return str(result)

        except Exception as e:
            return f"Error en generación de resumen: {e}"


# ======================================================
# 3. AGENTE REVISOR (Simulado)
# ======================================================

class Revisor:
    def __init__(self):
        pass

    def evaluar_texto(self, texto):
        """
        Devuelve una crítica estilo LLM.
        """
        evaluacion = (
            "• El resumen es claro y mantiene coherencia general.\n"
            "• Se recomienda fortalecer el tono académico.\n"
            "• Podrías añadir ejemplos específicos para ilustrar los riesgos éticos.\n"
            "• Sugiero añadir una frase que conecte explícitamente las aplicaciones prácticas con "
            "los desafíos éticos discutidos."
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

        # 4. RESUMEN FINAL (añadir mejoras sugeridas)
        texto_final = (
            f"{draft}\n\n"
            "### Ajustes propuestos por el revisor:\n"
            f"{review}\n"
        )

        return {
            "sources": fuentes,
            "draft": draft,
            "review": review,
            "final": texto_final
        }