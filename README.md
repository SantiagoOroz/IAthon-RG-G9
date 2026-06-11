# 🏭 Estación de Trabajo Inteligente 360

Una sola cámara sobre la estación del operario fueguino, **cuatro capas de IA** sobre el mismo frame de video:

| Capa | Qué hace | Disparador |
|------|----------|-----------|
| 🦴 **Salud** (ergonomía) | Detecta encorvado y avisa | Continuo (postura en tiempo real) |
| ✋ **Control** (manos libres) | Comandos sin tocar nada | Gestos de la mano |
| 🔍 **Calidad** (auditor IA) | Audita la pieza con un LLM multimodal | Gesto "pulgar arriba" |
| 📊 **Supervisión** | Tablero en vivo + chatbot | Espejo del estado |

El gesto que confirma *"pieza terminada"* es el mismo que **congela el frame y dispara la auditoría de calidad**: un solo movimiento del operario alimenta trazabilidad, control de calidad y el tablero del supervisor.

---

## 🚀 Cómo correr la demo

> Dependencias ya instaladas en la notebook del equipo. Si las necesitás de nuevo: `pip install -r requirements.txt`
> La API key de Gemini se lee del archivo `.env` (variable `API_KEY`).

Hacen falta **2 terminales**:

**Terminal 1 — la estación (cámara del operario):**
```bash
python estacion.py
```

**Terminal 2 — el tablero (pantalla del supervisor):**
```bash
streamlit run tablero.py
```

Cada vez que arranca `estacion.py` la estación se reinicia limpia (contadores en cero), ideal para la demo.

---

## 🤚 Gestos (mano frente a la cámara)

| Gesto | Acción |
|-------|--------|
| 👍 Pulgar arriba | **Pieza aprobada** → el operario la da por buena directamente |
| 👎 Pulgar abajo | **Enviar a revisión** → congela el frame y lo audita con la IA |
| ✋ Palma abierta | **Detener la línea** |
| ✌️ Victoria | **Avanzar manual** de ensamble |
| ✊ Puño cerrado | **Llamar al supervisor** |

El supervisor puede **reactivar la línea** desde el tablero (botón que aparece cuando está detenida).

> Los gestos tienen *debounce*: hay que sostenerlos ~8 frames y hay un cooldown de 3 s entre repeticiones, para no disparar la IA decenas de veces por segundo.

---

## 🎬 Guion sugerido para el pitch (5 min)

1. **Salud:** el operario se sienta a "ensamblar". Se encorva → la pantalla se pone roja con *ALERTA ERGONÓMICA*.
2. **Control:** hace ✌️ → "Avanzar manual". Hace ✋ → *LÍNEA DETENIDA*.
3. **Calidad:** sostiene una placa frente a la cámara y hace 👍 → el frame se congela, "AUDITANDO…", y aparece el veredicto **APROBADO / RECHAZADO** del LLM.
4. **Supervisión:** en la otra pantalla, el tablero ya sumó la pieza. El supervisor escribe en el chatbot *"¿cuántas piezas se rechazaron y por qué se detuvo la línea?"* y la IA responde con los datos en vivo. Reactiva la línea con un click.

Para probar el inspector sin placas físicas hay imágenes de ejemplo en el repo (`producto-defectuoso-*`).

---

## 🗂️ Arquitectura (archivos)

```
config.py      → parámetros (modelo, umbrales, gestos, paths). Tocás todo desde acá.
estado.py      → puente entre los 2 procesos: estado.json + eventos.jsonl (atómico).
inspector.py   → auditar_frame(frame) -> {estado, motivo}  (Gemini multimodal).
estacion.py    → loop de visión: postura + gestos + trigger de calidad (en thread).
tablero.py     → Streamlit: métricas en vivo + gráfico + chatbot del supervisor.
```

Los dos procesos no comparten memoria: `estacion.py` **escribe** el estado y `tablero.py` lo **lee** (con polling cada 2 s). Las órdenes del supervisor van por un canal aparte (`comando_supervisor.json`) para que nunca se pisen.

> **Nota técnica:** se corren dos modelos livianos de MediaPipe (pose + gestos) sobre el mismo frame, en lugar de `mediapipe.solutions.holistic`, porque Holistic no devuelve gestos con nombre (`Thumb_Up`, etc.). La auditoría LLM corre en un thread aparte para no congelar el video.
