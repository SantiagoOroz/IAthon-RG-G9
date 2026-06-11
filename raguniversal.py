import streamlit as st
import os
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

st.set_page_config(page_title="Asistente Corporativo IA", layout="wide")

# --- CONFIGURACIÓN: Pon tu API KEY de Google aquí ---
os.environ["GOOGLE_API_KEY"] = "TU_API_KEY_DE_GOOGLE_AQUI"
ARCHIVO_CONOCIMIENTO = "contexto_demo.txt" 

st.title("🤖 Copiloto Experto (RAG con Gemini)")

@st.cache_resource
def inicializar_base_conocimiento():
    if not os.path.exists(ARCHIVO_CONOCIMIENTO):
        with open(ARCHIVO_CONOCIMIENTO, "w", encoding="utf-8") as f:
            f.write("Articulo 1 de la Ley Fueguina: Los componentes electrónicos importados no pagan aranceles si se ensamblan localmente. Residuos: La empresa RioGrandeTech descarta 50kg de cartón diarios.")
    
    with open(ARCHIVO_CONOCIMIENTO, "r", encoding="utf-8") as f:
        texto_crudo = f.read()
        
    separador = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    textos = separador.split_text(texto_crudo)
    
    # Usamos los Embeddings gratuitos de Google
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_texts(textos, embeddings)
    
    # Usamos Gemini Flash como cerebro
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)
    
    cadena_qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())
    return cadena_qa

con_spinner = st.spinner("Indexando documentos locales...")
with con_spinner:
    motor_qa = inicializar_base_conocimiento()

if "mensajes" not in st.session_state:
    st.session_state.mensajes = []

for msg in st.session_state.mensajes:
    with st.chat_message(msg["rol"]):
        st.markdown(msg["contenido"])

pregunta = st.chat_input("Escribe tu consulta aquí...")

if pregunta:
    st.session_state.mensajes.append({"rol": "user", "contenido": pregunta})
    with st.chat_message("user"):
        st.markdown(pregunta)
        
    with st.chat_message("assistant"):
        with st.spinner("Buscando en la base documental..."):
            prompt_final = "Responde como un experto técnico fueguino basado SOLO en la base de datos: " + pregunta
            respuesta = motor_qa.invoke(prompt_final)
            
            texto_respuesta = respuesta['result']
            st.markdown(texto_respuesta)
            
    st.session_state.mensajes.append({"rol": "assistant", "contenido": texto_respuesta})