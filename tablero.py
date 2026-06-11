"""
TABLERO DEL SUPERVISOR (idea 4)  —  Estación 360.

Panel web que es el "espejo en tiempo real" de la estación: lee el estado que
escribe estacion.py y lo muestra en vivo, más un chatbot que responde sobre los
datos en vivo + el conocimiento de la planta.

Uso (en otra terminal, con estacion.py corriendo):
    streamlit run tablero.py
"""
import os

import streamlit as st
from google import genai

import config
import estado as st_estado

st.set_page_config(page_title="Estación 360 — Supervisor", layout="wide")

_cliente = genai.Client(api_key=config.GEMINI_API_KEY)


@st.cache_data
def cargar_contexto_planta():
    try:
        with open("contexto_demo.txt", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "Sin documentación de planta cargada."


def responder_supervisor(pregunta, estado, eventos, contexto):
    """Chatbot del supervisor: razona sobre los datos EN VIVO + el contexto fijo."""
    c = estado["contadores"]
    resumen_vivo = (
        f"Línea: {'ACTIVA' if estado['linea_activa'] else 'DETENIDA'}. "
        f"Postura actual: {estado['postura']['estado']} ({estado['postura']['angulo']}°). "
        f"Piezas aprobadas: {c['aprobadas']}, rechazadas: {c['rechazadas']}, "
        f"alertas posturales: {c['alertas_postura']}, comandos: {c['comandos']}. "
        f"Última inspección: {estado.get('ultima_inspeccion')}."
    )
    ultimos = "\n".join(
        f"- {e['ts']} [{e['tipo']}] {e['detalle']}" for e in eventos[-15:]
    ) or "Sin eventos todavía."

    sistema = (
        "Sos el asistente del supervisor de una línea de ensamble electrónico en "
        "Tierra del Fuego. Respondé en español, corto y concreto. Si la respuesta "
        "está en los datos en vivo, usalos; no inventes números.\n\n"
        f"=== DATOS EN VIVO DE LA ESTACIÓN ===\n{resumen_vivo}\n\n"
        f"=== ÚLTIMOS EVENTOS ===\n{ultimos}\n\n"
        f"=== CONOCIMIENTO DE PLANTA ===\n{contexto}"
    )
    resp = _cliente.models.generate_content(
        model=config.MODELO_LLM, contents=[sistema, pregunta]
    )
    return resp.text


# --------------------------------------------------------------------------- #
# Panel en vivo (se auto-refresca cada 2 s sin recargar toda la página)
# --------------------------------------------------------------------------- #
@st.fragment(run_every=2)
def panel_vivo():
    estado = st_estado.leer_estado()
    eventos = st_estado.leer_eventos()
    c = estado["contadores"]

    # --- Estado de la línea ---
    izq, der = st.columns([3, 1])
    with izq:
        if estado["linea_activa"]:
            st.success("🟢 LÍNEA ACTIVA")
        else:
            st.error("🔴 LÍNEA DETENIDA por el operario")
    with der:
        if not estado["linea_activa"]:
            if st.button("▶ Reactivar línea", type="primary", use_container_width=True):
                st_estado.enviar_comando_supervisor("reactivar_linea")
                st.toast("Orden enviada a la estación")

    # --- Métricas ---
    total = c["aprobadas"] + c["rechazadas"]
    tasa = f"{(100 * c['aprobadas'] / total):.0f}%" if total else "—"
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("✅ Aprobadas", c["aprobadas"])
    m2.metric("❌ Rechazadas", c["rechazadas"])
    m3.metric("📈 Tasa aprobación", tasa)
    m4.metric("⚠️ Alertas posturales", c["alertas_postura"])

    col_a, col_b = st.columns(2)

    # --- Evolución de calidad (gráfico temporal) ---
    with col_a:
        st.caption("Evolución de calidad")
        calidad = [e for e in eventos if e["tipo"] == "calidad"]
        if calidad:
            apro, rech, a, r = [], [], 0, 0
            for e in calidad:
                if e["detalle"].get("estado") == "APROBADO":
                    a += 1
                elif e["detalle"].get("estado") == "RECHAZADO":
                    r += 1
                apro.append(a)
                rech.append(r)
            st.line_chart({"Aprobadas": apro, "Rechazadas": rech})
        else:
            st.info("Esperando la primera auditoría de calidad...")

    # --- Última inspección (foto congelada + veredicto) ---
    with col_b:
        st.caption("Última inspección")
        insp = estado.get("ultima_inspeccion")
        if estado["inspeccionando"]:
            st.warning("🔎 Auditando pieza...")
        if insp:
            if insp["estado"] == "APROBADO":
                st.success(f"**{insp['estado']}** — {insp['motivo']}")
            elif insp["estado"] == "RECHAZADO":
                st.error(f"**{insp['estado']}** — {insp['motivo']}")
            else:
                st.info(f"**{insp['estado']}** — {insp['motivo']}")
        if os.path.exists(config.CAPTURA_PATH):
            st.image(str(config.CAPTURA_PATH), caption="Frame auditado", use_container_width=True)

    # --- Postura y comando actual ---
    p = estado["postura"]
    estado_post = "🟢 OK" if p["estado"] == "OK" else "🔴 RIESGO"
    detalle_p = p.get("detalle", "")
    post_txt = f"**Postura:** {estado_post} ({p['angulo']}°)"
    if detalle_p and detalle_p not in ("OK", "sin datos"):
        post_txt += f" — _{detalle_p}_"
    st.write(post_txt + f"  ·  **Último comando:** {estado['comando_actual']}")

    # --- Log de eventos ---
    with st.expander("📋 Historial de eventos", expanded=False):
        if eventos:
            filas = [
                {"hora": e["ts"][11:], "tipo": e["tipo"], "detalle": str(e["detalle"])}
                for e in reversed(eventos)
            ]
            st.dataframe(filas, use_container_width=True, hide_index=True)
        else:
            st.write("Sin eventos todavía.")

    st.caption(f"Actualizado: {estado.get('actualizado') or '—'}")


# --------------------------------------------------------------------------- #
# Render
# --------------------------------------------------------------------------- #
st.title("🏭 Estación de Trabajo Inteligente 360 — Tablero del Supervisor")

panel_vivo()

st.divider()
st.subheader("🤖 Asistente del supervisor")

if "chat" not in st.session_state:
    st.session_state.chat = []

for msg in st.session_state.chat:
    with st.chat_message(msg["rol"]):
        st.markdown(msg["texto"])

pregunta = st.chat_input("Ej: ¿cuántas piezas se rechazaron? ¿por qué se detuvo la línea?")
if pregunta:
    st.session_state.chat.append({"rol": "user", "texto": pregunta})
    with st.chat_message("user"):
        st.markdown(pregunta)
    with st.chat_message("assistant"):
        with st.spinner("Consultando datos de la estación..."):
            try:
                respuesta = responder_supervisor(
                    pregunta,
                    st_estado.leer_estado(),
                    st_estado.leer_eventos(),
                    cargar_contexto_planta(),
                )
            except Exception as e:
                respuesta = f"Error consultando la IA: {e}"
            st.markdown(respuesta)
    st.session_state.chat.append({"rol": "assistant", "texto": respuesta})
