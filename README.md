# Multi-Agent Research System — Tarea 1

## Descripción
Este proyecto implementa un **flujo de trabajo multi-agente** en Python utilizando **LangChain** y la **API de inferencia de Hugging Face**.  
El sistema simula una investigación colaborativa sobre un tema de IA, donde cada agente tiene un rol específico:

| Agente      | Función |
|------------|---------|
| Investigador | Recupera fuentes de texto relevantes usando DuckDuckGoSearchRun. |
| Redactor     | Genera un resumen de investigación en Markdown (~500 palabras) con Hugging Face. |
| Revisor     | Evalúa coherencia, veracidad y estructura del resumen, y sugiere mejoras. |
| Coordinator | Orquesta la comunicación entre los agentes y produce el resumen final. |

**Tema de ejemplo:** `Impacto de los datos sintéticos en la atención médica`.

---

## Requisitos

- Python 3.10+
- Librerías: listadas en `requirements.txt`
- Token de Hugging Face (`HF_TOKEN`) guardado en `.env` o como variable de entorno.

Instalación de dependencias:

```bash
pip install -r requirements.txt