import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import urllib.request

# 1. Descarga automática del modelo de Reconocimiento de Gestos
MODEL_PATH = "gesture_recognizer.task"
if not os.path.exists(MODEL_PATH):
    print("Descargando modelo de Gestos... Por favor espera.")
    url = "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task"
    urllib.request.urlretrieve(url, MODEL_PATH)
    print("¡Descarga completada!")

# 2. Configurar el inicializador
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.GestureRecognizerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE
)
recognizer = vision.GestureRecognizer.create_from_options(options)

# 3. Diccionario de "Comandos Industriales"
# Mapeamos los gestos en inglés que detecta la IA a acciones de la fábrica
comandos_industriales = {
    "Thumb_Up": ("PIEZA APROBADA (Siguiente)", (0, 255, 0)),     # Verde
    "Open_Palm": ("ALERTA: LINEA DETENIDA", (0, 0, 255)),        # Rojo
    "Victory": ("AVANZAR MANUAL DE ENSAMBLE", (255, 200, 0)),    # Celeste
    "Closed_Fist": ("SOLICITANDO SUPERVISOR...", (0, 165, 255)), # Naranja
    "None": ("Esperando comando...", (200, 200, 200))            # Gris
}

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: 
        break

    # Modo espejo para que sea intuitivo para el usuario
    frame = cv2.flip(frame, 1)
    
    # Preparar imagen para MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    # Ejecutar la detección de gestos
    resultado = recognizer.recognize(mp_image)
    
    imagen_salida = frame.copy()
    gesto_detectado = "None"

    # 4. Procesar si se encontró una mano y un gesto
    if resultado.gestures and len(resultado.gestures) > 0:
        # Tomar el gesto con mayor confianza
        gesto_detectado = resultado.gestures[0][0].category_name
        
        # Si el modelo detecta un gesto que no mapeamos, lo ignoramos
        if gesto_detectado not in comandos_industriales:
            gesto_detectado = "None"

    # 5. Interfaz Gráfica del Prototipo
    texto_comando, color = comandos_industriales[gesto_detectado]
    
    # Dibujar Banner de Estado
    cv2.rectangle(imagen_salida, (0, 0), (imagen_salida.shape[1], 80), (30, 30, 30), -1)
    
    if gesto_detectado != "None":
        # Bordes iluminados si hay un comando activo
        cv2.rectangle(imagen_salida, (0, 0), (imagen_salida.shape[1], imagen_salida.shape[0]), color, 8)
        
    cv2.putText(imagen_salida, texto_comando, (20, 55), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3, cv2.LINE_AA)

    # Mostrar la cámara
    cv2.imshow('Controlador de Estacion Manos Libres', imagen_salida)
    
    # Salir con 'q'
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

recognizer.close()
cap.release()
cv2.destroyAllWindows()