import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os
import urllib.request

# 1. Descarga automática del modelo de IA (Para que funcione "Copy-Paste")
MODEL_PATH = "pose_landmarker_full.task"
if not os.path.exists(MODEL_PATH):
    print("Descargando modelo de MediaPipe... Por favor espera unos segundos.")
    url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"
    urllib.request.urlretrieve(url, MODEL_PATH)
    print("¡Descarga completada con éxito!")

# 2. Función matemática para el ángulo de la columna
def calcular_angulo(a, b, c):
    a = np.array(a) # Oreja
    b = np.array(b) # Hombro (Vértice)
    c = np.array(c) # Cadera
    
    radianes = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angulo = np.abs(radianes * 180.0 / np.pi)
    
    if angulo > 180.0:
        angulo = 360.0 - angulo
        
    return angulo

# 3. Configurar el nuevo inicializador de MediaPipe Tasks
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE # Modo síncrono ideal para loops simples
)

# Inicializar el detector de posturas
detector = vision.PoseLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: 
        break

    # La nueva API exige procesar con la clase mp.Image en formato RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    # Ejecutar la inferencia de forma directa
    detection_result = detector.detect(mp_image)
    
    # Creamos una copia para dibujar las alertas con OpenCV
    image = frame.copy()

    # 4. Procesamiento de puntos de referencia (Landmarks)
    if detection_result.pose_landmarks:
        try:
            # Extraemos el esqueleto de la primera persona detectada
            landmarks = detection_result.pose_landmarks[0]
            
            # Índices oficiales de MediaPipe: Oreja Izq (7), Hombro Izq (11), Cadera Izq (23)
            oreja_lm = landmarks[7]
            hombro_lm = landmarks[11]
            cadera_lm = landmarks[23]
            
            # Coordenadas normalizadas (de 0 a 1)
            oreja = [oreja_lm.x, oreja_lm.y]
            hombro = [hombro_lm.x, hombro_lm.y]
            cadera = [cadera_lm.x, cadera_lm.y]
            
            # Calcular ángulo ergonómico
            angulo_postura = calcular_angulo(oreja, hombro, cadera)
            
            # --- LÓGICA DE NEGOCIO (Umbral) ---
            umbral_riesgo = 145 
            
            if angulo_postura > umbral_riesgo:
                color_alerta = (0, 200, 0) # Verde si está recto
                texto_estado = "POSTURA OK"
            else:
                color_alerta = (0, 0, 255) # Rojo si se encorva
                texto_estado = "ALERTA: RIESGO ERGONOMICO"
                # Rectángulo de alerta que encuadra toda la pantalla
                cv2.rectangle(image, (0, 0), (image.shape[1], image.shape[0]), color_alerta, 10)
                
            # Mapear las posiciones normalizadas a píxeles reales de tu pantalla
            hombro_px = (int(hombro[0] * image.shape[1]), int(hombro[1] * image.shape[0]))
            oreja_px = (int(oreja[0] * image.shape[1]), int(oreja[1] * image.shape[0]))
            cadera_px = (int(cadera[0] * image.shape[1]), int(cadera[1] * image.shape[0]))
            
            # Dibujar el esqueleto de perfil de manera personalizada (Más limpio para la demo)
            cv2.line(image, oreja_px, hombro_px, (255, 255, 255), 2)
            cv2.line(image, hombro_px, cadera_px, (255, 255, 255), 2)
            cv2.circle(image, oreja_px, 6, (245, 117, 66), -1)
            cv2.circle(image, hombro_px, 6, (245, 117, 66), -1)
            cv2.circle(image, cadera_px, 6, (245, 117, 66), -1)
            
            # Desplegar los grados calculados sobre el hombro
            cv2.putText(image, f"{int(angulo_postura)} grados", hombro_px, 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                    
            # Interfaz gráfica: Banner superior informativo
            cv2.rectangle(image, (0, 0), (image.shape[1], 60), color_alerta, -1)
            cv2.putText(image, texto_estado, (10, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3, cv2.LINE_AA)

        except Exception as e:
            pass # Ignora errores momentáneos si el cuerpo sale de cuadro
        
    # 5. Desplegar la ventana en vivo
    cv2.imshow('Monitor Ergonomico - IA-THON', image)
    
    # Salir presionando la tecla 'q'
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Liberar recursos de forma segura
detector.close()
cap.release()
cv2.destroyAllWindows()