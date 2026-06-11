import streamlit as st
from google import genai
from PIL import Image

# 1. Configuración de la página web
st.set_page_config(page_title="Auditor IA Fueguino", layout="centered")
st.title("🔍 Inspector de Calidad en Línea")
st.markdown("Arrastra la imagen del componente al salir de la línea de ensamble para una auditoría automatizada.")

# 2. Configuración de la IA con el NUEVO SDK
API_KEY = "TU_API_KEY_DE_GOOGLE_AQUI"
cliente = genai.Client(api_key=API_KEY)

# 3. Interfaz: Subir archivo
archivo_subido = st.file_uploader("Cargar imagen del producto (JPG/PNG)", type=["jpg", "jpeg", "png"])

if archivo_subido is not None:
    imagen = Image.open(archivo_subido)
    st.image(imagen, caption="Componente en evaluación", use_column_width=True)
    
    if st.button("Ejecutar Auditoría de Calidad", type="primary"):
        with st.spinner("Analizando soldaduras, conexiones y estado físico..."):
            
            prompt = """
            Eres un auditor de calidad estricto en una planta de ensamblaje electrónico en Tierra del Fuego.
            Analiza esta imagen y genera un reporte técnico breve (máximo 4 líneas).
            Indica claramente al principio: [APROBADO] o [RECHAZADO].
            Si es rechazado, describe exactamente dónde está el defecto visual.
            """
            
            try:
                # Llamada a la IA usando el nuevo formato
                respuesta = cliente.models.generate_content(
                    model='gemini-1.5-flash',
                    contents=[prompt, imagen]
                )
                
                st.success("Auditoría completada.")
                st.markdown("### Reporte Oficial:")
                st.write(respuesta.text)
                
            except Exception as e:
                st.error(f"Error de conexión con la IA: {e}")