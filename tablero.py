"""
TABLERO DEL SUPERVISOR (idea 4)  —  Estación 360.

Panel web que es el "espejo en tiempo real" de la estación: lee el estado que
escribe estacion.py y lo muestra en vivo, más un chatbot que responde sobre los
datos en vivo + el conocimiento de la planta.

Uso (en otra terminal, con estacion.py corriendo):
    streamlit run tablero.py
"""
import base64
import os
from collections import Counter

import pandas as pd
import streamlit as st
from google import genai

import config
import estado as st_estado

LOGO_PATH = "logoOpiVision.png"

st.set_page_config(
    page_title="OptiVision — Estación 360",
    page_icon=LOGO_PATH if os.path.exists(LOGO_PATH) else "👁️",
    layout="wide",
)

_cliente = genai.Client(api_key=config.GEMINI_API_KEY)

# Paleta de marca OptiVision (naranja + azul)
AZUL = "#1E5BB8"
NARANJA = "#FF7A18"

# --------------------------------------------------------------------------- #
# Estilo visual (tema naranja + azul, tarjetas y animaciones)
# --------------------------------------------------------------------------- #
_ESTILOS = """
<style>
:root{
  --ov-orange:#FF7A18; --ov-orange-soft:#FF9F45;
  --ov-blue:#10315B;  --ov-blue2:#1E5BB8;
}
.block-container{padding-top:1.6rem;}
h2,h3,h4{color:var(--ov-blue)!important;font-weight:800;}

/* Hero / encabezado */
.ov-hero{
  display:flex;align-items:center;gap:22px;
  background:linear-gradient(120deg,var(--ov-blue) 0%,var(--ov-blue2) 55%,var(--ov-orange) 135%);
  padding:20px 28px;border-radius:20px;margin-bottom:20px;
  box-shadow:0 8px 26px rgba(16,49,91,.28);
}
.ov-logo-wrap{background:#fff;border-radius:18px;padding:8px;flex:0 0 auto;
  box-shadow:0 4px 14px rgba(0,0,0,.20);}
.ov-logo{width:78px;height:78px;display:block;border-radius:12px;}
.ov-hero-text h1{color:#fff;margin:0;font-size:1.75rem;font-weight:800;letter-spacing:.3px;}
.ov-hero-text p{color:#ffe9d6;margin:5px 0 0;font-size:1.02rem;font-weight:500;}
.ov-live{color:#fff;background:rgba(255,122,24,.95);padding:3px 12px;border-radius:20px;
  font-size:.78rem;margin-left:10px;font-weight:700;animation:ovpulse 1.6s infinite;}
@keyframes ovpulse{0%,100%{opacity:1}50%{opacity:.3}}

/* Tarjetas de métricas */
[data-testid="stMetric"]{
  background:linear-gradient(135deg,#ffffff,#eef5ff);
  border-radius:16px;padding:14px 18px;
  border-top:4px solid var(--ov-orange);
  box-shadow:0 3px 14px rgba(16,49,91,.10);
}
[data-testid="stMetricValue"]{color:var(--ov-blue);font-weight:800;}
[data-testid="stMetricLabel"]{color:#5b6b7b;font-weight:600;}

/* Estado de la línea */
.ov-status{padding:15px 20px;border-radius:14px;font-weight:800;font-size:1.08rem;
  box-shadow:0 4px 14px rgba(16,49,91,.14);}
.ov-status-on{background:linear-gradient(135deg,var(--ov-blue),var(--ov-blue2));color:#fff;}
.ov-status-off{background:linear-gradient(135deg,#C0392B,var(--ov-orange));color:#fff;
  animation:ovpulse 1.4s infinite;}

/* Botones */
.stButton>button{background:var(--ov-orange);color:#fff;border:none;border-radius:12px;
  font-weight:700;padding:.55rem 1rem;transition:transform .08s ease,background .15s ease;}
.stButton>button:hover{background:#e96a0c;color:#fff;transform:translateY(-1px);}

/* Sub-títulos de sección */
.ov-section{display:flex;align-items:center;gap:8px;margin:6px 0 2px;
  font-weight:800;color:var(--ov-blue);font-size:1.12rem;}
.ov-section::before{content:"";width:6px;height:22px;border-radius:4px;
  background:linear-gradient(var(--ov-orange),var(--ov-blue2));}

/* Evitar que la página se atenúe en gris durante el auto-refresh (estado "stale") */
[data-stale="true"], .stApp [data-stale="true"],
.element-container[data-stale="true"], div[data-stale]{
  opacity:1 !important; transition:none !important; filter:none !important;
}
[data-testid="stStatusWidget"]{display:none !important;}
</style>
"""


@st.cache_data
def _logo_base64():
    try:
        with open(LOGO_PATH, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except FileNotFoundError:
        return ""


def encabezado():
    """Hero con logo OptiVision + título y badge EN VIVO."""
    b64 = _logo_base64()
    logo = (f'<img src="data:image/png;base64,{b64}" class="ov-logo"/>'
            if b64 else '<div class="ov-logo">👁️</div>')
    st.markdown(
        f"""
        <div class="ov-hero">
          <div class="ov-logo-wrap">{logo}</div>
          <div class="ov-hero-text">
            <h1>Estación de Trabajo Inteligente 360</h1>
            <p>OptiVision · Tablero del Supervisor
               <span class="ov-live">● EN VIVO</span></p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


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
            st.markdown('<div class="ov-status ov-status-on">🟢 LÍNEA ACTIVA</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<div class="ov-status ov-status-off">🔴 LÍNEA DETENIDA por el operario</div>',
                        unsafe_allow_html=True)
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
        st.markdown('<div class="ov-section">Evolución de calidad</div>', unsafe_allow_html=True)
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
            st.line_chart({"Aprobadas": apro, "Rechazadas": rech},
                          color=[AZUL, NARANJA])
        else:
            st.info("Esperando la primera auditoría de calidad...")

    # --- Última inspección (foto congelada + veredicto) ---
    with col_b:
        st.caption("Última inspección")
        insp = estado.get("ultima_inspeccion")
        if estado["inspeccionando"]:
            st.warning("🔎 Auditando pieza...")
        if insp:
            obj = insp.get("objeto", "")
            etiqueta = f"**{insp['estado']}**"
            if obj:
                etiqueta += f" · 📦 {obj}"
            etiqueta += f" — {insp['motivo']}"
            if insp["estado"] == "APROBADO":
                st.success(etiqueta)
            elif insp["estado"] == "RECHAZADO":
                st.error(etiqueta)
            else:
                st.info(etiqueta)
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

    # --- Análisis de postura (gráficos) ---
    st.markdown('<div class="ov-section">🧍 Análisis de postura</div>', unsafe_allow_html=True)
    col_p1, col_p2 = st.columns(2)

    # Errores posturales más comunes (cada alerta puede combinar varios problemas)
    with col_p1:
        st.caption("Errores posturales más comunes")
        errores = Counter()
        for e in eventos:
            if e["tipo"] == "postura":
                detalle = str(e["detalle"].get("detalle", ""))
                for parte in detalle.split(" + "):
                    parte = parte.strip()
                    if parte and parte not in ("OK", "sin datos"):
                        errores[parte] += 1
        if errores:
            df_err = pd.DataFrame(
                {"veces": list(errores.values())},
                index=list(errores.keys()),
            ).sort_values("veces", ascending=False)
            st.bar_chart(df_err, horizontal=True, color=NARANJA)
            mas_comun = df_err.index[0]
            st.caption(f"Más frecuente: **{mas_comun}** ({int(df_err.iloc[0]['veces'])} alertas)")
        else:
            st.info("Sin alertas posturales todavía.")

    # Porcentaje del tiempo en postura de riesgo (excluye frames sin cuerpo)
    with col_p2:
        st.caption("Tiempo en postura de riesgo")
        pt = estado.get("postura_tiempo", {"ok": 0.0, "riesgo": 0.0})
        t_ok, t_riesgo = float(pt.get("ok", 0)), float(pt.get("riesgo", 0))
        total_t = t_ok + t_riesgo
        if total_t > 0:
            pct_rojo = 100 * t_riesgo / total_t
            st.metric("% del tiempo en rojo", f"{pct_rojo:.0f}%",
                      help="Sobre el tiempo con cuerpo detectado. Los frames sin cuerpo no cuentan.")
            st.progress(min(int(pct_rojo), 100))
            df_t = pd.DataFrame(
                {"segundos": [round(t_ok, 1), round(t_riesgo, 1)]},
                index=["🟢 Correcta", "🔴 Riesgo"],
            )
            st.bar_chart(df_t, horizontal=True, color=AZUL)
        else:
            st.info("Aún no se acumuló tiempo de postura (sin cuerpo detectado).")

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
st.markdown(_ESTILOS, unsafe_allow_html=True)
if os.path.exists(LOGO_PATH):
    st.logo(LOGO_PATH)
encabezado()

panel_vivo()

st.divider()
st.markdown('<div class="ov-section">🤖 Asistente del supervisor</div>', unsafe_allow_html=True)

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
