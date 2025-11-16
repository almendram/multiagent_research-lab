from typing import List, Dict
from crewai import Agent, Task, Crew
from crewai_tools import tool
from duckduckgo_search import DDGS
from huggingface_hub import InferenceClient

# ======================
# 1. Tool: Buscador Web
# ======================
@tool("busqueda_web")
def busqueda_web(query: str) -> str:
    """Realiza una búsqueda en DuckDuckGo y devuelve texto concatenado."""
    resultados = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=5):
            title = r.get("title", "")
            body = r.get("body", "")
            resultados.append(f"{title}: {body}")
    return "\n".join(resultados)

# =========================
# 2. Modelos Hugging Face
# =========================
summary_model = InferenceClient("facebook/bart-large-cnn")
review_model  = InferenceClient("microsoft/deberta-v3-small")

# ======================
# 3. Crear Agentes
# ======================
def build_agents():
    investigador = Agent(
        name="Investigador",
        role="Analista de información",
        goal="Buscar información confiable para el tema solicitado.",
        backstory="Experto en recuperación de información y análisis documental.",
        tools=[busqueda_web],
        verbose=True
    )

    redactor = Agent(
        name="Redactor",
        role="Redactor científico",
        goal="Escribir un resumen claro y estructurado basado en los hallazgos.",
        backstory="Especialista en comunicación científica y escritura académica.",
        llm=summary_model,
        verbose=True
    )

    revisor = Agent(
        name="Revisor",
        role="Corrector académico",
        goal="Revisar y mejorar el texto generado por claridad y precisión.",
        backstory="Editor profesional con experiencia en revisión científica.",
        llm=review_model,
        verbose=True
    )

    return investigador, redactor, revisor

# ======================
# 4. Flujo CrewAI
# ======================
def build_workflow(topic="Tema de prueba"):
    investigador, redactor, revisor = build_agents()

    t1 = Task(
        description=f"Investiga el siguiente tema y devuelve hallazgos clave: {topic}",
        agent=investigador,
        expected_output="Lista estructurada con hallazgos relevantes."
    )

    t2 = Task(
        description="Redacta un resumen de aproximadamente 500 palabras basado en los hallazgos.",
        agent=redactor,
        expected_output="Resumen estructurado en formato Markdown."
    )

    t3 = Task(
        description="Revisa y corrige el resumen para mejorar claridad, coherencia y precisión.",
        agent=revisor,
        expected_output="Versión final corregida del resumen en Markdown."
    )

    crew = Crew(
        agents=[investigador, redactor, revisor],
        tasks=[t1, t2, t3],
        verbose=True
    )

    return crew