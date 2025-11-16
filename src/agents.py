# src/agents.py
from langchain import HuggingFaceHub, LLMChain, PromptTemplate
from langchain.chat_models import ChatOpenAI  # not used; we use HuggingFaceHub
from langchain_community.tools import DuckDuckGoSearchRun
import json
import os

# -------------------------------------------------
# UTIL: configura el cliente Hugging Face a través de LangChain
# -------------------------------------------------
def get_hf_llm(repo_id: str, hf_token: str, task: str = None, max_length: int = 512, temperature: float = 0.2):
    """
    Devuelve un wrapper LLM de LangChain que llama al endpoint de Hugging Face.
    repo_id: ejemplo "HuggingFaceH4/zephyr-7b-beta" o "google/flan-t5-small"
    """
    # HuggingFaceHub de LangChain utiliza huggingface_api_token en el init
    llm = HuggingFaceHub(repo_id=repo_id, huggingfacehub_api_token=hf_token, model_kwargs={"max_length": max_length, "temperature": temperature})
    return llm

# -------------------------------------------------
# AGENTE INVESTIGADOR: usa DuckDuckGoSearchRun (herramienta) para recuperar fragmentos
# -------------------------------------------------
def investigador_buscar(query: str, top_k: int = 5):
    """
    Realiza búsqueda con DuckDuckGoSearchRun y devuelve una lista de dicts: [{title, snippet, link}, ...]
    """
    search_tool = DuckDuckGoSearchRun()
    raw = search_tool.run(query)
    # DuckDuckGoSearchRun devuelve texto; lo encapsulamos en un único resultado si es string.
    # Para reproducibilidad guardamos cada resultado en una lista de dicts simple.
    results = []
    if isinstance(raw, str):
        # raw suele contener concatenación de títulos y snippets; lo devolvemos como un único fragmento
        results.append({"title": query, "snippet": raw, "link": ""})
    elif isinstance(raw, list):
        # si retorna lista, intentamos parsearla
        for r in raw[:top_k]:
            results.append({"title": r.get("title", ""), "snippet": r.get("snippet", r.get("body", "")), "link": r.get("link", "")})
    else:
        results.append({"title": query, "snippet": str(raw), "link": ""})
    return results

# -------------------------------------------------
# AGENTE ESCRITOR: genera borrador usando LLM de Hugging Face
# -------------------------------------------------
def escritor_generar_borrador(sources: list, hf_token: str, repo_id: str, word_target: int = 500):
    """
    sources: lista de dicts con 'title','snippet','link'
    repo_id: modelo en HF para generación (ej: "HuggingFaceH4/zephyr-7b-beta" o "google/flan-t5-small")
    Retorna Markdown string (borrador).
    """
    llm = get_hf_llm(repo_id=repo_id, hf_token=hf_token, max_length=1024, temperature=0.2)
    # Preparamos prompt
    aggregated = "\n\n".join([f"Title: {s.get('title','')}\nSnippet: {s.get('snippet','')}\nLink: {s.get('link','')}" for s in sources])
    prompt_template = """
Eres un escritor de investigación. Usando las siguientes fuentes (títulos y fragmentos), escribe un resumen de investigación en Markdown de aproximadamente {word_target} palabras sobre el tema. Estructura el resultado con:
# Introducción
# Hallazgos clave (lista con bullets)
# Desafíos éticos y técnicos
# Conclusión

Fuentes:
{sources}

Escribe en español y cita brevemente (p. ej., Fuente 1, Fuente 2) al final de cada hallazgo cuando sea posible.
"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["sources", "word_target"])
    chain = LLMChain(llm=llm, prompt=prompt)
    md = chain.run({"sources": aggregated, "word_target": word_target})
    return md

# -------------------------------------------------
# AGENTE REVISOR: evalúa veracidad y coherencia, sugiere cambios
# -------------------------------------------------
def revisor_evaluar(borrador_md: str, hf_token: str, repo_id: str):
    """
    Usa un modelo de clasificación/análisis para evaluar coherencia y veracidad.
    Devuelve un dict con observaciones y sugerencias.
    """
    llm = get_hf_llm(repo_id=repo_id, hf_token=hf_token, max_length=512, temperature=0.0)
    prompt_template = """
Eres un revisor experto. Analiza el siguiente borrador de investigación y responde en JSON con campos:
{
  "coherencia": "breve comentario",
  "veracidad": "breve comentario sobre posibles afirmaciones no verificadas",
  "estilo": "comentario sobre claridad y estructura",
  "sugerencias": ["lista de sugerencias concretas para mejorar el texto"]
}
Texto a revisar:
{borrador}
Responde sólo con JSON válido.
"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["borrador"])
    chain = LLMChain(llm=llm, prompt=prompt)
    raw = chain.run({"borrador": borrador_md})
    # intentamos parsear JSON; si falla, devolvemos texto libre
    try:
        eval_json = json.loads(raw)
    except Exception:
        eval_json = {"raw_output": raw}
    return eval_json

# -------------------------------------------------
# FLUJO COORDINADOR: ejecuta la secuencia Investigador -> Escritor -> Revisor -> Escritura final
# -------------------------------------------------
def ejecutar_flujo(topic_query: str, hf_token: str, writer_model: str, reviewer_model: str, save_md_path: str = "research_summary.md"):
    """
    Ejecuta el flujo completo. Devuelve la ruta del archivo MD final.
    """
    # 1) Investigador
    print("Investigador: buscando fuentes...")
    fuentes = investigador_buscar(topic_query, top_k=6)

    print(f"Investigador: recuperó {len(fuentes)} fragmentos.")
    # opcional: guardar fuentes
    try:
        with open("data/sources.jsonl", "w", encoding="utf-8") as f:
            for s in fuentes:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
    except Exception:
        pass

    # 2) Escritor (borrador)
    print("Escritor: generando borrador...")
    borrador = escritor_generar_borrador(fuentes, hf_token=hf_token, repo_id=writer_model, word_target=500)

    # 3) Revisor
    print("Revisor: evaluando borrador...")
    evaluacion = revisor_evaluar(borrador, hf_token=hf_token, repo_id=reviewer_model)

    # 4) Escritor: incorpora sugerencias (simplemente pide reescritura con sugerencias)
    # Si evaluacion incluye 'sugerencias', las aplicamos; sino las añadimos en texto de revisión.
    sugerencias_texto = ""
    if isinstance(evaluacion, dict) and "sugerencias" in evaluacion:
        sugerencias_texto = "\n".join([f"- {s}" for s in evaluacion.get("sugerencias", [])])
    else:
        sugerencias_texto = str(evaluacion.get("raw_output") if isinstance(evaluacion, dict) else evaluacion)

    # Creamos prompt de finalización:
    # reuse writer LLM
    print("Escritor: generando versión final incorporando sugerencias...")
    llm_writer = get_hf_llm(repo_id=writer_model, hf_token=hf_token, max_length=1024, temperature=0.15)
    final_prompt = PromptTemplate(
        template = """
Eres un escritor que va a producir la versión final del informe en Markdown (aprox 500 palabras), incorporando estas sugerencias del revisor:
{suggestions}

Aquí está el borrador original:
{draft}

Entrega el texto final en español, con la misma estructura:
# Introducción
# Hallazgos clave
# Desafíos éticos y técnicos
# Conclusión
""",
        input_variables=["suggestions","draft"]
    )
    chain_final = LLMChain(llm=llm_writer, prompt=final_prompt)
    final_md = chain_final.run({"suggestions": sugerencias_texto, "draft": borrador})

    # Guardamos archivo final
    with open(save_md_path, "w", encoding="utf-8") as f:
        f.write(final_md)

    print(f"Informe final guardado en {save_md_path}")
    # también devolvemos la evaluación para que quede registro
    return {"md_path": save_md_path, "evaluation": evaluacion, "draft": borrador, "final": final_md}
