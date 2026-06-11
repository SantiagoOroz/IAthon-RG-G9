"""
Estado compartido entre el loop de visión (estacion.py) y el tablero (tablero.py).

Son dos procesos distintos (OpenCV vive en uno, Streamlit en otro), así que se
comunican por archivos:
  - estado.json  : snapshot actual (se sobreescribe de forma atómica)
  - eventos.jsonl: historial append-only (un evento JSON por línea)
"""
import json
import os
import tempfile
from datetime import datetime

from config import ESTADO_PATH, EVENTOS_PATH, CAPTURA_PATH, COMANDO_PATH


def estado_inicial():
    """Devuelve un snapshot fresco (nunca compartir el mismo dict entre corridas)."""
    return {
        "postura": {"estado": "—", "angulo": 0},
        "comando_actual": "Esperando comando...",
        "linea_activa": True,
        "inspeccionando": False,
        "ultima_inspeccion": None,  # {"estado","motivo","ts"}
        "contadores": {"aprobadas": 0, "rechazadas": 0, "alertas_postura": 0, "comandos": 0},
        "actualizado": None,
    }


def leer_estado():
    try:
        with open(ESTADO_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return estado_inicial()


def escribir_estado(estado):
    """Escritura atómica: el tablero nunca lee un JSON a medio escribir."""
    estado["actualizado"] = datetime.now().isoformat(timespec="seconds")
    carpeta = os.path.dirname(ESTADO_PATH) or "."
    fd, tmp = tempfile.mkstemp(dir=carpeta, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(estado, f, ensure_ascii=False)
        os.replace(tmp, ESTADO_PATH)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def registrar_evento(tipo, detalle):
    """Agrega un evento al historial. tipo: 'calidad' | 'comando' | 'postura'."""
    evento = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "tipo": tipo,
        "detalle": detalle,
    }
    with open(EVENTOS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(evento, ensure_ascii=False) + "\n")
    return evento


def leer_eventos(limite=50):
    try:
        with open(EVENTOS_PATH, encoding="utf-8") as f:
            lineas = [l for l in f if l.strip()]
        return [json.loads(l) for l in lineas[-limite:]]
    except Exception:
        return []


def enviar_comando_supervisor(accion):
    """El tablero deja una orden para la estación (p. ej. reactivar la línea)."""
    with open(COMANDO_PATH, "w", encoding="utf-8") as f:
        json.dump({"accion": accion}, f)


def tomar_comando_supervisor():
    """La estación lee y consume la orden pendiente (la borra). Devuelve la acción o None."""
    try:
        with open(COMANDO_PATH, encoding="utf-8") as f:
            accion = json.load(f).get("accion")
        os.remove(COMANDO_PATH)
        return accion
    except Exception:
        return None


def reset():
    """Deja la estación limpia (útil para arrancar una demo de cero)."""
    escribir_estado(estado_inicial())
    for ruta in (EVENTOS_PATH, CAPTURA_PATH, COMANDO_PATH):
        try:
            os.remove(ruta)
        except FileNotFoundError:
            pass
