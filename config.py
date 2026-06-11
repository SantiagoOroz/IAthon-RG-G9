"""
Configuración compartida de la "Estación de Trabajo Inteligente 360".

Todos los parámetros que tocarías para una demo (modelo, umbrales, sensibilidad
de gestos) viven acá, en un solo lugar.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# La key en el .env está como API_KEY (admitimos también GOOGLE_API_KEY por las dudas)
GEMINI_API_KEY = os.getenv("API_KEY") or os.getenv("GOOGLE_API_KEY")

# Cerebro LLM (visión + chatbot). gemini-2.5-flash = rápido y estable para demo en vivo.
# Para más precisión podés subir a "gemini-2.5-pro" o "gemini-3-flash-preview".
MODELO_LLM = "gemini-2.5-flash"
# Modelo de respaldo si el principal devuelve 503/429 en plena demo.
MODELO_FALLBACK = "gemini-flash-latest"

# --- Archivos de estado compartido (puente entre estacion.py y tablero.py) ---
ESTADO_PATH = BASE_DIR / "estado.json"        # snapshot actual de la estación
EVENTOS_PATH = BASE_DIR / "eventos.jsonl"     # historial (un evento por línea)
CAPTURA_PATH = BASE_DIR / "ultima_captura.jpg"  # frame congelado de la última auditoría
COMANDO_PATH = BASE_DIR / "comando_supervisor.json"  # órdenes supervisor -> estación (1 vía)

# --- Modelos de MediaPipe (se autodescargan si faltan) ---
POSE_MODEL_PATH = BASE_DIR / "pose_landmarker_full.task"
GESTO_MODEL_PATH = BASE_DIR / "gesture_recognizer.task"
POSE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"
GESTO_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task"

# --- Ergonomía ---
UMBRAL_POSTURA = 145          # grados oreja-hombro-cadera; por debajo = encorvado
UMBRAL_CUELLO_ADELANTADO = 0.35   # fracción del ancho de torso; >0.35 = cabeza proyectada
UMBRAL_HOMBROS_DESNIVELADOS = 0.12  # fracción del alto de torso; >0.12 = inclinación lateral
FRAMES_ALERTA_POSTURA = 15    # frames seguidos con riesgo antes de registrar una alerta

# --- Gestos / comandos (debounce para no disparar 30 veces por segundo) ---
FRAMES_CONFIRMACION_GESTO = 8   # frames sosteniendo el gesto para confirmarlo
COOLDOWN_COMANDO_SEG = 3.0      # segundos mínimos entre dos disparos del mismo comando

# Mapa de gestos -> acción industrial + color BGR para el overlay
GESTOS = {
    "Thumb_Up":    {"accion": "PIEZA TERMINADA  ->  Auditar calidad", "color": (0, 200, 0)},   # verde
    "Open_Palm":   {"accion": "DETENER LINEA",                        "color": (0, 0, 255)},   # rojo
    "Victory":     {"accion": "AVANZAR MANUAL",                       "color": (255, 200, 0)},  # celeste
    "Closed_Fist": {"accion": "LLAMAR SUPERVISOR",                    "color": (0, 165, 255)},  # naranja
}

# Prompt de sistema del auditor de calidad (idea 3)
PROMPT_AUDITOR = (
    "Sos un auditor de calidad estricto en una planta de ensamblaje electronico en "
    "Tierra del Fuego. Analizas el producto que el operario sostiene frente a la camara.\n"
    "Responde UNICAMENTE con un JSON valido (sin markdown, sin texto extra) con esta forma exacta:\n"
    '{"estado": "APROBADO" | "RECHAZADO", "motivo": "<explicacion breve, max 12 palabras>"}\n'
    "Es RECHAZADO si ves: soldadura fria, cable suelto, rayon, golpe, componente faltante o torcido."
)
