"""
Inspector de calidad (idea 3) — versión integrable.

A diferencia de inspectorcalidad.py (que recibía una imagen subida por Streamlit),
acá la entrada es un frame BGR de OpenCV: el frame que el loop de visión "congela"
cuando el operario hace el gesto de pieza terminada.

    veredicto = auditar_frame(frame_bgr)  ->  {"estado": "APROBADO", "motivo": "..."}
"""
import json
import re
import time

import cv2
from PIL import Image
from google import genai

from config import GEMINI_API_KEY, MODELO_LLM, MODELO_FALLBACK, PROMPT_AUDITOR

_cliente = genai.Client(api_key=GEMINI_API_KEY)

# Modelo principal + respaldo (sin duplicar si son iguales).
_MODELOS = list(dict.fromkeys([MODELO_LLM, MODELO_FALLBACK]))


def auditar_frame(frame_bgr):
    """Manda un frame al LLM multimodal y devuelve {'estado', 'motivo'}.

    Reintenta y cae a un modelo de respaldo si el principal está saturado
    (503/429 son habituales en vivo), para que la demo no se caiga.
    """
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    imagen = Image.fromarray(rgb)
    ultimo_error = "desconocido"
    for modelo in _MODELOS:
        for _ in range(2):
            try:
                resp = _cliente.models.generate_content(
                    model=modelo, contents=[PROMPT_AUDITOR, imagen]
                )
                return _parsear(resp.text)
            except Exception as e:
                ultimo_error = str(e)[:60]
                time.sleep(1.0)
    return {"estado": "ERROR", "objeto": "", "motivo": f"IA no disponible: {ultimo_error}"}


def _parsear(texto):
    """El modelo suele devolver JSON limpio, pero blindamos contra fences y ruido."""
    if not texto:
        return {"estado": "ERROR", "objeto": "", "motivo": "Respuesta vacia"}

    limpio = re.sub(r"```(json)?", "", texto).strip()
    match = re.search(r"\{.*\}", limpio, re.DOTALL)
    if match:
        try:
            datos = json.loads(match.group(0))
            return {
                "estado": _normalizar_estado(datos.get("estado", "")),
                "objeto": str(datos.get("objeto", "")).strip(),
                "motivo": str(datos.get("motivo", "")).strip(),
            }
        except json.JSONDecodeError:
            pass

    # Fallback: heurística sobre el texto crudo
    return {"estado": _normalizar_estado(limpio), "objeto": "", "motivo": limpio[:80]}


def _normalizar_estado(valor):
    v = str(valor).upper()
    if "RECHAZ" in v:
        return "RECHAZADO"
    if "APROB" in v:
        return "APROBADO"
    return "REVISAR"
