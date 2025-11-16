# src/agents.py
"""
Agentes del proyecto multi-agent_research-lab
- Investigador: busca fuentes en la web usando DuckDuckGo (langchain_community tool)
- Redactor: sintetiza / resume utilizando la API de inferencia de Hugging Face
- Revisor: evalúa coherencia/verdad; usa modelo HF si hay token, si no, devuelve evaluación heurística
- Coordinator: bucle simple Investigator -> Redactor -> Revisor -> Redactor final
"""

from typing import List, Dict, Optional
import os
from huggingface_hub import InferenceClient
from langchain_community.tools import DuckDuckGoSearchRun
from dotenv import load_dotenv
from textwrap import dedent

load_dotenv()  # carga .env si existe

# -------------------------
# 1) INVESTIGADOR
# -------------------------
class Investigador:
    def __init__(self, herramienta=None):
        # Usa la herramienta de DuckDuckGo por defecto
        self.search_tool = herramienta or DuckDuckGoSearchRun()
    
    def buscar_fuentes(self, tema: str, sitio_filter: Optional[str] = None, top_k: int = 5) -> List[Dict]:
        """
        Realiza una búsqueda y devuelve una lista de resultados resumidos.
        Cada elemento: {'title': ..., 'snippet': ..., 'link': ...}
        """
        query = tema
        if sitio_filter:
            query = f"{tema} site:{sitio_filter}"
        try:
            raw = self.search_tool.run(query)
        except Exception as e:
            return [{"title": "Error en búsqueda", "snippet": str(e), "link": ""}]
        
        # DuckDuckGoSearchRun suele devolver texto con fragmentos; intentamos parsearlo
        # Si raw es string, lo devolvemos como un único snippet
        results = []
        if isinstance(raw, str):
            results.append({"title": tema, "snippet": raw[:800], "link": ""})
        elif isinstance(raw, list):
            # cada item puede ser dict o string
            for item in raw[:top_k]:
                if isinstance(item, dict):
                    title = item.get("title") or item.get("heading") or ""
                    snippet = item.get("snippet") or item.get("body") or ""
                    link = item.get("link") or item.get("url") or ""
                    results.append({"title": title, "snippet": snippet, "link": link})
                else:
                    # item tipo string
                    results.append({"title": tema, "snippet": str(item)[:800], "link": ""})
        else:
            # estructura inesperada
            results.append({"title": tema, "snippet": str(raw)[:800], "link": ""})
        
        # Asegurar cantidad
        return results[:top_k]


# -------------------------
# 2) REDACTOR
# -------------------------
class Redactor:
    def __init__(self, modelo: str = "facebook/bart-large-cnn"):
        """
        Modelo por defecto: facebook/bart-large-cnn (para summarization)
        Usa InferenceClient de huggingface_hub.
        """

        self.modelo = modelo
        token = self._leer_token()
        self.client = InferenceClient(token=token) if token else None

    def _leer_token(self) -> Optional[str]:
        # lee variable HF_TOKEN desde .env o desde variable de entorno
        token = os.environ.get("HF_TOKEN")
        if token:
            return token
        # también buscar archivo .env con formato HF_TOKEN=xxx
        env_path = ".env"
        if os.path.exists(env_path):
            try:
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("HF_TOKEN"):
                            parts = line.split("=", 1)
                            if len(parts) == 2:
                                return parts[1].strip().strip('"').strip("'")
            except Exception:
                pass
        return None

    def generar_resumen(self, texto: str, max_tokens: int = 500) -> str:
        """
        Genera un resumen de texto (aprox. 500 palabras) usando la API de Hugging Face.
        Si no hay token o falla la llamada, se devuelve un fallback simple (truncado).
        """
        # Preparar prompt/inputs
        prompt_input = {
            "inputs": texto,
            "parameters": {"max_new_tokens": max_tokens}
        }

        if self.client:
            try:
                # try standard summarization
                result = self.client.summarization(model=self.modelo, **prompt_input)
            except Exception as e:
                # fallback: intentar text_generation con prompt de resumen
                try:
                    prompt = "Resume el siguiente texto en estilo académico (~500 palabras):\n\n" + texto
                    gen = self.client.text_generation(model=self.modelo, inputs=prompt, max_new_tokens=max_tokens)
                    # gen puede venir como dict o lista
                    if isinstance(gen, dict):
                        return gen.get("generated_text", str(gen))
                    if isinstance(gen, list) and len(gen) > 0:
                        return gen[0].get("generated_text", str(gen[0]))
                    return str(gen)
                except Exception as e2:
                    return f"Error en generación (fallback): {e2}"

            # Procesar respuesta de summarization
            if isinstance(result, dict) and "summary_text" in result:
                return result["summary_text"]
            if isinstance(result, list) and len(result) > 0:
                # puede venir como list de dicts
                item = result[0]
                if isinstance(item, dict) and "summary_text" in item:
                    return item["summary_text"]
                # si es string, devolverlo
                if isinstance(item, str):
                    return item
            # si result tiene attribute summary_text
            if hasattr(result, "summary_text"):
                return getattr(result, "summary_text")
            # fallback: string
            return str(result)
        else:
            # Sin token: fallback simple
            # Devuelve las primeras N palabras y una nota
            words = texto.split()
            summary_words = words[:400]  # fallback ~ 400 palabras cortas
            summary = " ".join(summary_words)
            note = ("\n\n[NOTA: no se encontró token de Hugging Face; "
                    "se devolvió un resumen truncado local como fallback]\n")
            return summary + note


# -------------------------
# 3) REVISOR
# -------------------------
class Revisor:
    def __init__(self, modelo_eval: str = "gpt2"):
        """
        Revisor: intenta usar un modelo HF para evaluar el texto si hay token.
        Por defecto modelo_eval es genérico; si no hay token, se usa evaluación heurística.
        """
        self.modelo_eval = modelo_eval
        token = os.environ.get("HF_TOKEN")
        if not token:
            # intenta leer desde .env
            token = Redactor()._leer_token()
        self.client = InferenceClient(token=token) if token else None

    def evaluar_texto(self, texto: str) -> str:
        """
        Devuelve una evaluación en formato bullet list.
        Si hay client, genera una evaluación automática con prompt; si no, devuelve heurística.
        """
        heuristic = dedent(
            "• El texto presenta una estructura clara y mantiene coherencia general.\n"
            "• Se recomienda reforzar el tono académico con transiciones más formales.\n"
            "• Añadir ejemplos concretos que conecten aplicaciones con desafíos éticos.\n"
            "• Sugerencia: incluir citas/links de respaldo y aclarar limitaciones metodológicas."
        )
        if self.client:
            try:
                prompt = (
                    "Eres un revisor académico. Analiza la coherencia, veracidad y estructura "
                    "del siguiente texto y sugiere mejoras en viñetas (máximo 6 viñetas):\n\n"
                    f"--- TEXTO A REVISAR ---\n{texto}\n\n--- FIN ---\n"
                )
                # Utilizamos text_generation para generar evaluación tipo LLM
                resp = self.client.text_generation(model=self.modelo_eval, inputs=prompt, max_new_tokens=200)
                # resp puede ser dict/list/str
                if isinstance(resp, dict) and "generated_text" in resp:
                    return resp["generated_text"]
                if isinstance(resp, list) and len(resp) > 0:
                    first = resp[0]
                    if isinstance(first, dict) and "generated_text" in first:
                        return first["generated_text"]
                    return str(first)
                if isinstance(resp, str):
                    return resp
                # fallback
                return heuristic
            except Exception:
                return heuristic
        else:
            return heuristic


# -------------------------
# COORDINADOR: Orquesta ciclo Investigar -> Redactar -> Revisar -> Redactar final
# -------------------------
class Coordinator:
    def __init__(self, investigador: Investigador, redactor: Redactor, revisor: Revisor):
        self.investigador = investigador
        self.redactor = redactor
        self.revisor = revisor

    def run(self, tema: str, top_k: int = 5) -> Dict[str, str]:
        """
        Ejecuta el flujo y devuelve dict con 'raw_sources', 'draft', 'review', 'final'
        """
        # 1) Investigación
        fuentes = self.investigador.buscar_fuentes(tema, top_k=top_k)

        # Concatenar snippets para redactor
        collected = "\n\n".join([f"Title: {f.get('title','')}\nLink: {f.get('link','')}\nSnippet: {f.get('snippet','')}" for f in fuentes])

        # 2) Primer borrador
        prompt_for_writer = (
            "Genera un resumen académico de investigación de aproximadamente 500 palabras "
            "en formato Markdown para el siguiente tema. Debe incluir: Introducción, "
            "Hallazgos clave, Desafíos éticos y técnicos, Conclusión.\n\n"
            f"TEMA: {tema}\n\nFUENTES (fragmentos):\n{collected}\n\n"
        )
        draft = self.redactor.generar_resumen(prompt_for_writer, max_tokens=500)

        # 3) Revisión
        review = self.revisor.evaluar_texto(draft)

        # 4) Redactor final: incorporar mejoras sugeridas (simple: añadimos comentario del revisor al prompt)
        prompt_final = (
            "Usando el siguiente borrador y las observaciones del revisor, produce la versión final "
            "del informe en Markdown de ~500 palabras. Integra las mejoras sugeridas:\n\n"
            f"BORRADOR:\n{draft}\n\nOBSERVACIONES:\n{review}\n\nVERSIÓN FINAL:\n"
        )
        final = self.redactor.generar_resumen(prompt_final, max_tokens=500)

        return {
            "raw_sources": fuentes,
            "draft": draft,
            "review": review,
            "final": final
        }