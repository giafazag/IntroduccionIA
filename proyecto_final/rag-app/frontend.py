import streamlit as st
import requests

# Configuración de la URL del Backend FastAPI
BACKEND_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="RAG System (Google AI + ChromaDB)", layout="wide")
st.title("🤖 Sistema RAG")

# Barra lateral para la carga de documentos
with st.sidebar:
    st.header("📂 Gestión de Documentos")
    uploaded_file = st.file_uploader("Sube un archivo PDF de conocimiento", type=["pdf"])
    
    if uploaded_file is not None:
        if st.button("Procesar e Indexar Documento"):
            with st.spinner("El backend está extrayendo y vectorizando el contenido..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                try:
                    response = requests.post(f"{BACKEND_URL}/upload", files=files)
                    res_data = response.json()
                    
                    if res_data.get("status") == "success":
                        st.success(res_data.get("message"))
                    else:
                        st.error(f"Error: {res_data.get('message')}")
                except Exception as e:
                    st.error(f"No se pudo conectar con el servidor Backend: {e}")

# Historial de Chat usando Session State de Streamlit
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar el historial en pantalla
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Capturar nueva pregunta del usuario
if user_query := st.chat_input("Hazle una pregunta a tus documentos vectorizados..."):
    # Mostrar mensaje del usuario inmediatamente
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    # Consultar al Backend
    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            try:
                response = requests.post(f"{BACKEND_URL}/query", data={"question": user_query})
                res_data = response.json()
                
                if res_data.get("status") == "success":
                    answer = res_data.get("answer")
                    st.markdown(answer)
                    
                    # Mostrar fuentes de manera discreta si existen
                    sources = res_data.get("sources", [])
                    if sources and "no encuentro esa información" not in answer.lower()::
                        source_files = list(set([s['source'] for s in sources]))
                        st.caption(f"📚 *Fuentes consultadas: {', '.join(source_files)}*")
                        
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    error_msg = f"Error del backend: {res_data.get('message')}"
                    st.error(error_msg)
            except Exception as e:
                st.error(f"Error de comunicación: {e}")
