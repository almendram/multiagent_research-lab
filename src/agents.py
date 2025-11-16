# agents.py
"""
Multi-Agent Research System — Tarea 1 (LangChain + Hugging Face Inference API)
- Investigador: DuckDuckGoSearchRun (langchain_community.tools)
- Redactor: Hugging Face Inference API (facebook/bart-large-cnn) -> summarization
- Revisor: Reglas heurísticas + checks simples (robusto y reproducible)
- Coordinator: orquesta el bucle Investigador -> Redactor -> Revisor -> Rewriter final

Requisitos:
- Python 3.10+
- Exportar token HF en .env con la clave HF_TOKEN
"""

import os
from dotenv import load_dotenv
from typing import List, Dict, Any

# LangChain DuckDuckGo tool
from langchain_community.tools import DuckDuckGoSearchRun

# Hugging Face InferenceClient
from huggingface_hub import InferenceClient

load_dotenv()

def leer_token():
    token = os.getenv("HF_TOKEN")
    if not token:
        raise ValueError(
            "HF_TOKEN no encontrado. Crea un archivo .env con:\nHF_TOKEN=tu_token_aqui"
        )
    return token


# ----------------------------
# Investigador (buscador web)
# ----------------------------
class Investigador:
    def __init__(self, top_k: int = 5):
        self.top_k = top_k
        self.search_tool = DuckDuckGoSearchRun()

    def buscar(self, query: str) -> List[Dict[str, str]]:
        """
        Ejecuta una búsqueda y devuelve una lista de resultados con: title, snippet, url.
        DuckDuckGoSearchRun.run suele devolver texto; en algunos entornos devuelve json-like.
        Aquí intentamos parsear de forma robusta.
        """
        raw = self.search_tool.run(query)
        # raw suele ser string; intentamos heurística de parseo simple
        results = []
        try:
            # Si raw ya es lista/dict, devolver en formato unificado
            if isinstance(raw, (list, tuple)):
                for item in raw[: self.top_k]:
                    if isinstance(item, dict):
                        results.append({
                            "title": item.get("title", ""),
                            "snippet": item.get("body", item.get("snippet", "")),
                            "url": item.get("link", item.get("url", ""))
                        })
                    else:
                        # fallback: text
                        results.append({"title": str(item)[:80], "snippet": str(item), "url": ""})
                return results
            # Si es string: parse simple por líneas con http -> url heurística
            lines = [l.strip() for l in str(raw).splitlines() if l.strip()]
            current = {"title": "", "snippet": "", "url": ""}
            for line in lines:
                if line.startswith("http") or "http" in line:
                    # guardar bloque anterior
                    if any(current.values()):
                        results.append(current.copy())
                        current = {"title": "", "snippet": "", "url": ""}
                    current["url"] = line
                elif len(line.split()) <= 10 and not current["title"]:
                    current["title"] = line
                else:
                    # acumular como snippet
                    current["snippet"] = (current["snippet"] + " " + line).strip()
            if any(current.values()):
                results.append(current)
            # Si no hay resultados parseados, fallback: devolver raw entero en un solo item
            if not results:
                results = [{"title": query, "snippet": raw[:1000], "url": ""}]
            # cortar top_k
            return results[: self.top_k]
        except Exception as e:
            return [{"title": "error", "snippet": f"Error parseo DuckDuckGo: {e}", "url": ""}]


# ----------------------------
# Redactor (resúmenes con HF Inference API)
# ----------------------------
class Redactor:
    def __init__(self, modelo: str = "facebook/bart-large-cnn"):
        self.modelo = modelo
        self.client = InferenceClient(token=leer_token())

    def _build_prompt(self, topic: str, sources: List[Dict[str, str]]) -> str:
        """
        Construye un prompt claro para que el modelo genere un resumen.
        Aunque 'bart-large-cnn' funciona con summarization endpoint, pasamos
        un texto concatenado y meta-instrucciones por seguridad.
        """
        header = f"Genera un resumen en español de aprox. 400-600 palabras en formato Markdown sobre: {topic}\n\n"
        header += "Usa las siguientes fuentes (título / snippet / url). Cita con [número] cuando correspondan.\n\n"
        body = ""
        for i, s in enumerate(sources, start=1):
            body += f"[{i}] Título: {s.get('title','')}\nSnippet: {s.get('snippet','')}\nURL: {s.get('url','')}\n\n"
        instructions = (
            "Estructura (Markdown):\n# Introducción\n## Hallazgos clave\n## Desafíos éticos y técnicos\n## Conclusión\n\n"
            "Mantén un tono académico y claro. Si una afirmación no está en las fuentes, marca [sin cita]."
        )
        return header + body + instructions

    def generar_resumen(self, topic: str, sources: List[Dict[str, str]]) -> str:
        """
        Llama al endpoint de summarization. Si el cliente no soporta 'summarization' como método,
        se usa text_generation con prompts (safe fallback).
        """
        # Concatenar fuentes en un texto de entrada razonable (acortar si demasiado largo)
        concatenated = ""
        for s in sources:
            concatenated += f"{s.get('title','')}\n{s.get('snippet','')}\n{s.get('url','')}\n\n"
        prompt = self._build_prompt(topic, sources)

        try:
            # Intentar el endpoint de summarization (método ofrecido por InferenceClient)
            try:
                resp = self.client.summarization(
                    model=self.modelo,
                    text=concatenated,
                    max_length=500
                )
                if isinstance(resp, dict) and "summary_text" in resp:
                    return resp["summary_text"].strip()
                # si devuelve texto plano
                return str(resp).strip()
            except Exception:
                # Fallback: usar text_generation (algunos clientes/planes aceptan esto)
                resp2 = self.client.text_generation(model=self.modelo, inputs=prompt, max_new_tokens=500)
                # resp2 puede ser dict o list
                if isinstance(resp2, dict):
                    for k in ("generated_text", "text", "output"):
                        if k in resp2:
                            return resp2[k].strip()
                    return str(resp2).strip()
                elif isinstance(resp2, list) and len(resp2) > 0:
                    first = resp2[0]
                    if isinstance(first, dict) and "generated_text" in first:
                        return first["generated_text"].strip()
                    return str(first).strip()
                else:
                    return str(resp2).strip()

        except Exception as e:
            return f"Error en generación de resumen: {e}"


# ----------------------------
# Revisor (heurístico, robusto)
# ----------------------------
class Revisor:
    def __init__(self):
        pass

    def evaluar_texto(self, summary: str, sources: List[Dict[str, str]]) -> str:
        """
        Revisión simple y accionable:
        - Comprueba longitud y estructura (presencia de secciones)
        - Heurísticas para detectar declaraciones sin cita (busca 'sin cita')
        - Lista de sugerencias concretas
        """
        notes = []
        # estructura
        if "# Introducción" in summary or "## Hallazgos" in summary:
            notes.append("Estructura: Las secciones principales están presentes.")
        else:
            notes.append("Estructura: Falta alguna sección esperada (introducción/hallazgos/conclusión).")

        # longitud
        words = len(summary.split())
        if words < 350:
            notes.append(f"Longitud: corto ({words} palabras). Objetivo: 400-600 palabras.")
        elif words > 700:
            notes.append(f"Longitud: largo ({words} palabras). Recomendar acotar a 400-600 palabras.")
        else:
            notes.append(f"Longitud: adecuada ({words} palabras).")

        # citas heurísticas
        missing_cites = "sin cita" in summary.lower() or "[sin cita]" in summary.lower()
        if missing_cites:
            notes.append("Citas: Se detectaron marcadores de afirmaciones sin cita. Añadir referencias específicas donde proceda.")
        else:
            notes.append("Citas: No se detectaron marcadores obvios de 'sin cita' — verificar manualmente contra las fuentes.")

        # coherencia simple (frases largas sin puntos)
        long_sentences = [s for s in summary.split(".") if len(s.split()) > 80]
        if long_sentences:
            notes.append("Coherencia: Hay oraciones muy largas. Divídelas para mejorar claridad.")
        else:
            notes.append("Coherencia: Buen nivel de claridad en oraciones.")

        # sugerencias concretas (ejemplos)
        suggestions = [
            "1) Añadir al menos 2 citas directas a las fuentes enumeradas para las afirmaciones clave.",
            "2) Revisar secciones 'Desafíos éticos y técnicos' para incluir un ejemplo concreto (por ejemplo, un caso de uso clínico).",
            "3) Comprobar datos numéricos en las URLs originales para evitar inexactitudes."
        ]

        return "• " + "\n• ".join(notes) + "\n\nSugerencias:\n" + "\n".join(suggestions)


# ----------------------------
# Coordinator (orquesta el flujo)
# ----------------------------
class Coordinator:
    def __init__(self, investigador: Investigador, redactor: Redactor, revisor: Revisor):
        self.investigador = investigador
        self.redactor = redactor
        self.revisor = revisor

    def run(self, topic: str, query: str, top_k: int = 5) -> Dict[str, Any]:
        # 1. Investigación
        raw_sources = self.investigador.buscar(query)

        # Si raw_sources es texto/plural, intentar transformarlo a lista de dicts si no lo está
        sources_list = []
        if isinstance(raw_sources, list):
            sources_list = raw_sources
        else:
            # si es texto, creamos un solo item
            sources_list = [{"title": topic, "snippet": raw_sources, "url": ""}]

        # 2. Redactor - primer borrador (usa el topic y sources_list)
        draft = self.redactor.generar_resumen(topic, sources_list)

        # 3. Revisor - feedback
        review = self.revisor.evaluar_texto(draft, sources_list)

        # 4. Rewriter simple: pedimos al redactor que reescriba incorporando sugerencias
        rewrite_prompt = (
            "REQUERIMIENTO: Reescribe el resumen incorporando las siguientes sugerencias del revisor. "
            "Mantén formato Markdown y aprox. 400-600 palabras.\n\n"
            f"BORRADOR:\n{draft}\n\n"
            f"FEEDBACK:\n{review}\n\n"
        )
        # Intentamos usar text_generation as a rewrite (fallback)
        try:
            resp = self.redactor.client.text_generation(
                model=self.redactor.modelo,
                inputs=rewrite_prompt,
                max_new_tokens=600
            )
            final_text = ""
            if isinstance(resp, dict):
                for k in ("generated_text", "text", "output"):
                    if k in resp:
                        final_text = resp[k]
                        break
                if not final_text:
                    final_text = str(resp)
            elif isinstance(resp, list) and len(resp) > 0:
                first = resp[0]
                if isinstance(first, dict) and "generated_text" in first:
                    final_text = first["generated_text"]
                else:
                    final_text = str(first)
            else:
                final_text = str(resp)
        except Exception:
            # Si falla la generación de reescritura, usar el draft + review como fallback
            final_text = draft + "\n\n### Ajustes sugeridos:\n" + review

        full = (
            f"# Resumen final - {topic}\n\n"
            f"## Fuentes (resumen):\n"
        )
        for i, s in enumerate(sources_list, start=1):
            full += f"{i}. {s.get('title','(sin título)')} - {s.get('url','')}\n\n"
            full += f"> {s.get('snippet','')}\n\n"

        full += "\n## Resumen Final\n\n" + final_text + "\n\n"
        full += "## Feedback del Revisor\n\n" + review + "\n"

        return {
            "sources": sources_list,
            "draft": draft,
            "review": review,
            "final": full
        }