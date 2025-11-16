%%writefile src/agents.py
from crewai import Agent, Task, Crew
from crewai_tools import tool
from duckduckgo_search import DDGS
from huggingface_hub import InferenceClient


# ======================
# 1. Tool: Buscador Web
# ======================
@tool("busqueda_web")
def busqueda_web(query: str) -> str:
    """Realiza búsqueda en DuckDuckGo y devuelve texto concatenado."""
    textos = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=5):
            textos.append(f"{r.get('title')}: {r.get('body')}")
    return "\n".join(textos)


# ======================
# 2. Modelos HF
# ======================
summary_model = InferenceClient("facebook/bart-large-cnn")
review_model  = InferenceClient("microsoft/deberta-v3-small")


# ======================
# 3. Agentes
# ======================
def build_agents():

    investigador = Agent(
        name="Investigador",
        role="Analista de información",
        goal="Buscar información precisa y confiable sobre el tema.",
        backstory="Experto en búsqueda avanzada y análisis documental.",
        tools=[busqueda_web],
        verbose=True
    )

    redactor = Agent(
        name="Redactor",
        role="Escritor científico",
        goal="Escribir un resumen estructurado de 500 palabras.",
        backstory="Especialista en comunicación clara y divulgación técnica.",
        llm=summary_model,
        verbose=True
    )

    revisor = Agent(
        name="Revisor",
        role="Verificador académico",
        goal="Revisar coherencia, estilo y precisión del texto.",
        backstory="Editor profesional con enfoque en investigación científica.",
        llm=review_model,
        verbose=True
    )

    return investigador, redactor, revisor


# ======================
# 4. Flujo con Crew
# ======================
def build_workflow(topic="Sesgo en los LLM"):
    investigador, redactor, revisor = build_agents()

    t1 = Task(
        description=f"Investiga el siguiente tema y devuelve puntos clave: {topic}",
        agent=investigador,
        expected_output="Lista clara de hallazgos relevantes."
    )

    t2 = Task(
        description="Redacta un resumen de 500 palabras usando los hallazgos.",
        agent=redactor,
        expected_output="Resumen en Markdown bien estructurado."
    )

    t3 = Task(
        description="Revisa y corrige el resumen para garantizar precisión y claridad.",
        agent=revisor,
        expected_output="Versión final corregida del resumen en Markdown."
    )

    crew = Crew(agents=[investigador, redactor, revisor], tasks=[t1, t2, t3], verbose=True)
    return crew