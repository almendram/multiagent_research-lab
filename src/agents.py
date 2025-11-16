# src/agents.py

from langchain import HuggingFaceHub, LLMChain, PromptTemplate
from langchain_community.tools import DuckDuckGoSearchRun
import json

# -------------------------------------------------
# Cargar modelo HuggingFaceHub SIEMPRE con task fijo
# -------------------------------------------------
def cargar_llm(repo_id, hf_token, max_length=512, temperature=0.3):
    return HuggingFaceHub(
        repo_id=repo_id,
        huggingfacehub_api_token=hf_token,
        task="text2text-generation",      # ← FUNDAMENTAL
        model_kwargs={
            "max_length": max_length,
            "temperature": temperature,
        }
    )

# -------------------------------------------------
# AGENTE INVESTIGADOR
# -------------------------------------------------
def investigador_buscar(query: str, top_k: int = 5):
    search_tool = DuckDuckGoSearchRun()
    raw = search_tool.run(query)

    results = []
    if isinstance(raw, str):
        results.append({"title": query, "snippet": raw, "link": ""})
    elif isinstance(raw, list):
        for r in raw[:top_k]:
            results.append({
                "title": r.get("title", ""),
                "snippet": r.get("snippet", r.get("body", "")),
                "link": r.get("link", "")
            })
    else:
        results.append({"title": query, "snippet": str(raw), "link": ""})

    return results

# -------------------------------------------------
# AGENTE ESCRITOR
# -------------------------------------------------
def escritor_generar_borrador(sources: list, hf_token: str, repo_id: str, word_target: int = 500):
    llm = cargar_llm(repo_id, hf_token, max_length=1024, temperature=0.2)

    aggregated = "\n\n".join(
        [f"Title: {s['title']}\nSnippet: {s['snippet']}\nLink: {s['link']}" for s in sources]
    )

    prompt_template = """
Eres un escritor de investigación. Usando las siguientes fuentes, escribe un informe Markdown de {word_target} palabras:

# Introducción
# Hallazgos clave
# Desafíos éticos y técnicos
# Conclusión

Fuentes:
{sources}

Escribe en español.
"""

    prompt = PromptTemplate(template=prompt_template, input_variables=["sources", "word_target"])
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run({"sources": aggregated, "word_target": word_target})

# -------------------------------------------------
# AGENTE REVISOR
# -------------------------------------------------
def revisor_evaluar(borrador_md: str, hf_token: str, repo_id: str):
    llm = cargar_llm(repo_id, hf_token, max_length=512, temperature=0.0)

    prompt_template = """
Eres un revisor experto. Analiza el siguiente texto y responde sólo con JSON:

{
  "coherencia": "",
  "veracidad": "",
  "estilo": "",
  "sugerencias": []
}

Texto:
{borrador}
"""

    prompt = PromptTemplate(template=prompt_template, input_variables=["borrador"])
    chain = LLMChain(llm=llm, prompt=prompt)

    raw = chain.run({"borrador": borrador_md})

    try:
        return json.loads(raw)
    except:
        return {"raw_output": raw}

# -------------------------------------------------
# FLUJO COMPLETO
# -------------------------------------------------
def ejecutar_flujo(topic_query: str, hf_token: str, writer_model: str, reviewer_model: str, save_md_path: str = "research_summary.md"):
    print("Investigador: buscando fuentes...")
    fuentes = investigador_buscar(topic_query, top_k=6)
    print(f"Investigador: recuperó {len(fuentes)} fragmentos.")

    print("Escritor: generando borrador...")
    borrador = escritor_generar_borrador(fuentes, hf_token, writer_model)

    print("Revisor: evaluando borrador...")
    evaluacion = revisor_evaluar(borrador, hf_token, reviewer_model)

    sugerencias = "\n".join([f"- {s}" for s in evaluacion.get("sugerencias", [])])

    print("Escritor: generando versión final...")
    llm_writer = cargar_llm(writer_model, hf_token, max_length=1024, temperature=0.15)

    final_prompt = PromptTemplate(
        template="""
Reescribe este borrador incorporando las sugerencias:

Sugerencias:
{suggestions}

Borrador original:
{draft}

Mantén la estructura:
# Introducción
# Hallazgos clave
# Desafíos éticos y técnicos
# Conclusión
""",
        input_variables=["suggestions", "draft"]
    )

    chain_final = LLMChain(llm=llm_writer, prompt=final_prompt)
    final_md = chain_final.run({"suggestions": sugerencias, "draft": borrador})

    with open(save_md_path, "w", encoding="utf-8") as f:
        f.write(final_md)

    print(f"Informe final guardado en {save_md_path}")
    return {"md_path": save_md_path, "evaluation": evaluacion, "draft": borrador, "final": final_md}
