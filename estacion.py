"""
ESTACIÓN DE TRABAJO INTELIGENTE 360  —  loop de visión unificado.

Fusiona en un solo proceso las cuatro ideas que tienen las capas de cámara:
    · Postura  (idea 1): ángulo oreja-hombro-cadera -> alerta ergonómica.
    · Gestos   (idea 2): comandos manos libres (avanzar / detener / supervisor).
    · Calidad  (idea 3): el gesto "pulgar arriba" congela el frame y lo audita
                    con el LLM, SIN congelar el video (corre en un thread).
    · Estado   (-> idea 4): vuelca todo a estado.json / eventos.jsonl para el tablero.

Uso:
    python estacion.py
Salir: tecla 'q'.
"""
import os
import sys
import time
import queue
import threading
import urllib.request
from datetime import datetime

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import config
import estado as st
from inspector import auditar_frame

# La consola de Windows (cp1252) puede crashear al imprimir UTF-8; forzamos UTF-8.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# --------------------------------------------------------------------------- #
# Carga de modelos (autodescarga si faltan)
# --------------------------------------------------------------------------- #
def _asegurar_modelo(ruta, url, nombre):
    if not os.path.exists(ruta):
        print(f"Descargando modelo de {nombre}...")
        urllib.request.urlretrieve(url, ruta)
        print("  ...listo.")


_asegurar_modelo(config.POSE_MODEL_PATH, config.POSE_MODEL_URL, "postura")
_asegurar_modelo(config.GESTO_MODEL_PATH, config.GESTO_MODEL_URL, "gestos")

pose_detector = vision.PoseLandmarker.create_from_options(
    vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(config.POSE_MODEL_PATH)),
        running_mode=vision.RunningMode.IMAGE,
    )
)
gesto_recognizer = vision.GestureRecognizer.create_from_options(
    vision.GestureRecognizerOptions(
        base_options=python.BaseOptions(model_asset_path=str(config.GESTO_MODEL_PATH)),
        running_mode=vision.RunningMode.IMAGE,
    )
)


# --------------------------------------------------------------------------- #
# Inspección de calidad en segundo plano (no bloquea el video)
# --------------------------------------------------------------------------- #
_cola_resultados = queue.Queue()   # veredictos que el loop todavía no incorporó
_insp_event = threading.Event()    # marca si hay una auditoría en curso


def _worker_inspeccion(frame_limpio):
    veredicto = auditar_frame(frame_limpio)
    _cola_resultados.put(veredicto)
    _insp_event.clear()


def disparar_inspeccion(frame_limpio):
    """Lanza la auditoría en un thread y guarda el frame congelado para el tablero.

    Solo el loop principal llama a esta función, así que el chequeo+set no necesita lock.
    """
    if _insp_event.is_set():
        return False
    _insp_event.set()
    cv2.imwrite(str(config.CAPTURA_PATH), frame_limpio)
    threading.Thread(
        target=_worker_inspeccion, args=(frame_limpio.copy(),), daemon=True
    ).start()
    return True


# --------------------------------------------------------------------------- #
# Geometría
# --------------------------------------------------------------------------- #
def calcular_angulo(a, b, c):
    """Ángulo en el vértice b."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    rad = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    ang = np.abs(rad * 180.0 / np.pi)
    return 360.0 - ang if ang > 180.0 else ang


def evaluar_postura(landmarks, w, h):
    """
    3 métricas bilaterales para detectar riesgo ergonómico:
        1. Ángulo espinal   — promedio izq+der (oreja-hombro-cadera)
        2. Cabeza adelantada — desplazamiento X oreja vs hombro (visible en perfil)
        3. Hombros desnivelados — asimetría vertical entre ambos hombros

    Retorna (riesgo: bool, angulo: int, puntos_px: dict, detalle: str)
    """
    MIN_VIS = 0.25
    INDICES = {
        "oreja_izq": 7,  "oreja_der": 8,
        "hombro_izq": 11, "hombro_der": 12,
        "cadera_izq": 23, "cadera_der": 24,
    }
    try:
        lm  = {k: landmarks[v] for k, v in INDICES.items()}
        vis = {k: lm[k].visibility > MIN_VIS for k in lm}
        px  = {k: (int(lm[k].x * w), int(lm[k].y * h)) for k in lm if vis[k]}

        problemas = []
        angulos   = []

        # 1. Ángulo espinal: usa el lado con mayor visibilidad mínima.
        # Promediar ambos lados introduce ruido cuando uno está oculto (persona de perfil).
        candidatos = []
        for ore, hom, cad in [
            ("oreja_izq", "hombro_izq", "cadera_izq"),
            ("oreja_der", "hombro_der", "cadera_der"),
        ]:
            if all(vis.get(k) for k in (ore, hom, cad)):
                confianza = min(lm[ore].visibility, lm[hom].visibility, lm[cad].visibility)
                ang = calcular_angulo(
                    [lm[ore].x, lm[ore].y],
                    [lm[hom].x, lm[hom].y],
                    [lm[cad].x, lm[cad].y],
                )
                candidatos.append((confianza, ang))
        angulo_prom = int(max(candidatos, key=lambda x: x[0])[1]) if candidatos else 0
        if angulo_prom and angulo_prom < config.UMBRAL_POSTURA:
            problemas.append("espalda encorvada")

        # 2. Cabeza proyectada hacia adelante
        torso_ancho = max(
            abs(lm["hombro_izq"].x - lm["hombro_der"].x)
            if vis["hombro_izq"] and vis["hombro_der"] else 0,
            0.05,
        )
        for ore_k, hom_k in [("oreja_izq", "hombro_izq"), ("oreja_der", "hombro_der")]:
            if vis.get(ore_k) and vis.get(hom_k):
                if abs(lm[ore_k].x - lm[hom_k].x) / torso_ancho > config.UMBRAL_CUELLO_ADELANTADO:
                    problemas.append("cabeza adelantada")
                    break

        # 3. Asimetría de hombros (solo cuando la persona está de frente)
        # En perfil, los hombros se superponen en X y la métrica no tiene sentido.
        if vis["hombro_izq"] and vis["hombro_der"]:
            ancho_hombros = abs(lm["hombro_izq"].x - lm["hombro_der"].x)
            if ancho_hombros >= config.UMBRAL_FRONTAL_MIN:
                torso_alto = max(
                    abs(lm["hombro_izq"].y - lm["cadera_izq"].y)
                    if vis["cadera_izq"] else 0,
                    0.05,
                )
                if abs(lm["hombro_izq"].y - lm["hombro_der"].y) / torso_alto > config.UMBRAL_HOMBROS_DESNIVELADOS:
                    problemas.append("hombros desnivelados")

        detalle = " + ".join(problemas) if problemas else "OK"
        return bool(problemas), angulo_prom, px, detalle

    except Exception:
        return False, 0, {}, "sin datos"


# Conexiones del esqueleto completo de MediaPipe Pose (33 landmarks)
_POSE_CONN = [
    (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),          # cara
    (9,10),                                                     # boca
    (11,12),(11,13),(13,15),(15,17),(15,19),(15,21),(17,19),   # brazo izq
    (12,14),(14,16),(16,18),(16,20),(16,22),(18,20),           # brazo der
    (11,23),(12,24),(23,24),                                    # torso
    (23,25),(24,26),(25,27),(26,28),                           # muslos
    (27,29),(28,30),(29,31),(30,32),(27,31),(28,32),           # tobillos/pies
]


def _dibujar_esqueleto(img, landmarks, postura_riesgo, w, h):
    """Esqueleto completo: verde si postura OK, rojo si hay riesgo."""
    MIN_VIS = 0.25
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    vis = [lm.visibility > MIN_VIS for lm in landmarks]
    color = (0, 0, 210) if postura_riesgo else (0, 210, 0)
    for a, b in _POSE_CONN:
        if a < len(pts) and b < len(pts) and vis[a] and vis[b]:
            cv2.line(img, pts[a], pts[b], color, 2, cv2.LINE_AA)
    for i, p in enumerate(pts):
        if vis[i]:
            cv2.circle(img, p, 3, (245, 117, 66), -1)


# --------------------------------------------------------------------------- #
# Dibujo del overlay
# --------------------------------------------------------------------------- #
def dibujar_overlay(img, datos):
    h, w = img.shape[:2]

    # Borde de alerta (prioridad: auditando > línea detenida > mala postura)
    if datos["inspeccionando"]:
        cv2.rectangle(img, (0, 0), (w, h), (0, 200, 255), 12)
    elif not datos["linea_activa"]:
        cv2.rectangle(img, (0, 0), (w, h), (0, 0, 255), 12)
    elif datos["postura_riesgo"]:
        cv2.rectangle(img, (0, 0), (w, h), (0, 0, 255), 8)

    # Banner superior: estado ergonómico
    col_post = (0, 0, 255) if datos["postura_riesgo"] else (0, 170, 0)
    if datos["postura_riesgo"]:
        txt_post = "RIESGO: " + datos.get("detalle_postura", "postura incorrecta").upper()
    else:
        txt_post = "POSTURA OK"
    if datos["angulo"]:
        txt_post += f"  ({datos['angulo']}g)"
    cv2.rectangle(img, (0, 0), (w, 50), col_post, -1)
    cv2.putText(img, txt_post, (12, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

    # Puntos ergonómicos clave (resaltados encima del esqueleto completo)
    for p in datos.get("puntos_px", {}).values():
        cv2.circle(img, p, 9, (245, 117, 66), -1)
        cv2.circle(img, p, 9, (255, 255, 255), 2)

    # Banner inferior: comando manos libres
    cv2.rectangle(img, (0, h - 50), (w, h), (30, 30, 30), -1)
    cv2.putText(img, f"COMANDO: {datos['comando']}", (12, h - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, datos["color_comando"], 2, cv2.LINE_AA)

    # Línea detenida (cartel central)
    if not datos["linea_activa"]:
        cv2.putText(img, "LINEA DETENIDA", (int(w * 0.20), int(h * 0.5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 4, cv2.LINE_AA)

    # Panel última inspección (abajo a la derecha)
    insp = datos["ultima_inspeccion"]
    if datos["inspeccionando"]:
        cv2.putText(img, "AUDITANDO CALIDAD...", (int(w * 0.30), 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 3, cv2.LINE_AA)
    elif insp:
        col = (0, 170, 0) if insp["estado"] == "APROBADO" else (0, 0, 255)
        obj = insp.get("objeto", "")
        alto = 95 if obj else 70
        pw, px, py = 360, w - 370, h - alto - 60
        cv2.rectangle(img, (px, py), (px + pw, py + alto), (20, 20, 20), -1)
        cv2.rectangle(img, (px, py), (px + pw, py + alto), col, 2)
        cv2.putText(img, f"CALIDAD: {insp['estado']}", (px + 10, py + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2, cv2.LINE_AA)
        if obj:
            cv2.putText(img, f"OBJETO: {obj[:32]}", (px + 10, py + 53),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 220, 255), 1, cv2.LINE_AA)
            motivo_y = py + 78
        else:
            motivo_y = py + 55
        cv2.putText(img, insp["motivo"][:42], (px + 10, motivo_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)

    # Contadores (arriba a la derecha)
    c = datos["contadores"]
    resumen = f"OK:{c['aprobadas']}  RECH:{c['rechazadas']}  POST:{c['alertas_postura']}"
    cv2.putText(img, resumen, (w - 360, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)


# --------------------------------------------------------------------------- #
# Loop principal
# --------------------------------------------------------------------------- #
def main():
    st.reset()  # demo limpia en cada arranque
    estado = st.estado_inicial()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se pudo abrir la webcam (cámara 0). ¿Está en uso por otra app?")
        return

    # Estado de debounce
    gesto_candidato, contador_gesto = "None", 0
    ultimo_comando_ts = {}
    frames_mala_postura = 0
    mala_postura_desde = None  # timestamp en que empezó la mala postura cruda (countdown anti-parpadeo)
    ultimo_ts_postura = time.time()  # para acumular tiempo en cada estado de postura
    frame_idx = 0
    puntos_px = {}
    detalle_postura = "sin datos"
    color_comando = (200, 200, 200)  # color del último comando (persiste en pantalla)

    print("Estacion 360 en marcha. Tecla 'q' para salir.")
    print("Gestos: pulgar arriba = pieza + auditar | palma abierta = detener | "
        "victoria = avanzar manual | puno cerrado = supervisor")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)          # espejo: intuitivo para el operario
        limpio = frame.copy()               # sin overlay -> esto ve el LLM
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # ---- Capa postura ----
        postura_riesgo = False
        angulo_int = 0
        h_fr, w_fr = frame.shape[:2]
        res_pose = pose_detector.detect(mp_img)
        riesgo_crudo = False
        if res_pose.pose_landmarks:
            riesgo_crudo, angulo_int, puntos_px, detalle_postura = evaluar_postura(
                res_pose.pose_landmarks[0], w_fr, h_fr
            )

        # Countdown anti-parpadeo: la mala postura debe sostenerse SEG_CONFIRMAR_POSTURA
        # segundos seguidos antes de marcarse en rojo. Así una oscilación breve alrededor
        # del umbral no hace titilar la postura entre buena y mala.
        ahora_post = time.time()
        if riesgo_crudo:
            if mala_postura_desde is None:
                mala_postura_desde = ahora_post
            postura_riesgo = (ahora_post - mala_postura_desde) >= config.SEG_CONFIRMAR_POSTURA
        else:
            mala_postura_desde = None
            postura_riesgo = False

        if res_pose.pose_landmarks:
            _dibujar_esqueleto(frame, res_pose.pose_landmarks[0], postura_riesgo, w_fr, h_fr)

        # Acumular tiempo en cada estado. Si no se detecta cuerpo, el frame no cuenta
        # ni como rojo ni como verde (se excluye del porcentaje).
        dt = ahora_post - ultimo_ts_postura
        ultimo_ts_postura = ahora_post
        if res_pose.pose_landmarks:
            clave = "riesgo" if postura_riesgo else "ok"
            estado["postura_tiempo"][clave] = round(estado["postura_tiempo"][clave] + dt, 1)

        if postura_riesgo:
            frames_mala_postura += 1
            if frames_mala_postura == config.FRAMES_ALERTA_POSTURA:
                estado["contadores"]["alertas_postura"] += 1
                st.registrar_evento("postura", {"angulo": angulo_int, "detalle": detalle_postura})
        else:
            frames_mala_postura = 0
        estado["postura"] = {"estado": "RIESGO" if postura_riesgo else "OK", "angulo": angulo_int, "detalle": detalle_postura}

        # ---- Capa gestos (con debounce) ----
        res_gesto = gesto_recognizer.recognize(mp_img)
        gesto_actual = "None"
        if res_gesto.gestures:
            cand = res_gesto.gestures[0][0].category_name
            if cand in config.GESTOS:
                gesto_actual = cand

        if gesto_actual == gesto_candidato:
            contador_gesto += 1
        else:
            gesto_candidato, contador_gesto = gesto_actual, 1

        confirmado = None
        if (contador_gesto == config.FRAMES_CONFIRMACION_GESTO
                and gesto_candidato in config.GESTOS):
            ahora = time.time()
            if ahora - ultimo_comando_ts.get(gesto_candidato, 0) > config.COOLDOWN_COMANDO_SEG:
                confirmado = gesto_candidato
                ultimo_comando_ts[gesto_candidato] = ahora

        if confirmado:
            info = config.GESTOS[confirmado]
            estado["comando_actual"] = info["accion"]
            color_comando = info["color"]
            estado["contadores"]["comandos"] += 1
            st.registrar_evento("comando", {"gesto": confirmado, "accion": info["accion"]})
            if confirmado == "Thumb_Up":
                # El operario aprueba la pieza directamente (sin IA)
                veredicto = {
                    "estado": "APROBADO",
                    "motivo": "aprobado por operario",
                    "ts": datetime.now().isoformat(timespec="seconds"),
                }
                estado["ultima_inspeccion"] = veredicto
                estado["contadores"]["aprobadas"] += 1
                st.registrar_evento("calidad", veredicto)
            elif confirmado == "Thumb_Down":
                # El operario tiene dudas: manda la pieza a revisión de la IA
                disparar_inspeccion(limpio)
            elif confirmado == "Open_Palm":
                estado["linea_activa"] = False
            st.escribir_estado(estado)

        # ---- Órdenes del supervisor (tablero -> estación) ----
        orden = st.tomar_comando_supervisor()
        if orden == "reactivar_linea":
            estado["linea_activa"] = True
            estado["comando_actual"] = "LINEA REACTIVADA (supervisor)"
            color_comando = (0, 200, 0)
            st.registrar_evento("comando", {"gesto": "supervisor", "accion": "Reactivar linea"})
            st.escribir_estado(estado)

        # ---- Incorporar veredicto de calidad si el thread terminó ----
        try:
            pendiente = _cola_resultados.get_nowait()
        except queue.Empty:
            pendiente = None
        if pendiente is not None:
            pendiente["ts"] = datetime.now().isoformat(timespec="seconds")
            estado["ultima_inspeccion"] = pendiente
            if pendiente["estado"] == "APROBADO":
                estado["contadores"]["aprobadas"] += 1
            elif pendiente["estado"] == "RECHAZADO":
                estado["contadores"]["rechazadas"] += 1
            st.registrar_evento("calidad", pendiente)
            st.escribir_estado(estado)
        estado["inspeccionando"] = _insp_event.is_set()

        # ---- Overlay + persistencia periódica ----
        dibujar_overlay(frame, {
            "postura_riesgo": postura_riesgo,
            "angulo": angulo_int,
            "puntos_px": puntos_px,
            "detalle_postura": detalle_postura,
            "comando": estado["comando_actual"],
            "color_comando": color_comando,
            "linea_activa": estado["linea_activa"],
            "inspeccionando": estado["inspeccionando"],
            "ultima_inspeccion": estado["ultima_inspeccion"],
            "contadores": estado["contadores"],
        })

        frame_idx += 1
        if frame_idx % 5 == 0:
            st.escribir_estado(estado)

        cv2.imshow("Estacion 360 - Vision (operario)", frame)
        if cv2.waitKey(5) & 0xFF == ord("q"):
            break

    estado["comando_actual"] = "Estacion apagada"
    st.escribir_estado(estado)
    pose_detector.close()
    gesto_recognizer.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
