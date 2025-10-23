# ~/agente_sql/frontend/app.py (Versión con Mejoras Visuales)

import streamlit as st
import pandas as pd
import requests
import io
import os

st.set_page_config(page_title="Autoconsulta IA", page_icon="🤖", layout="wide")

AGENT_AVATAR = "https://images.seeklogo.com/logo-png/49/2/sodimac-warehouse-logo-png_seeklogo-494670.png"

# --- CAMBIO PRINCIPAL (1/3): Añadir CSS para el color del botón de descarga ---
st.markdown("""
<style>
    /* Estilos generales del mensaje de chat */
    .stChatMessage { 
        border-radius: 10px; 
        padding: 1rem; 
        margin-bottom: 1rem; 
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.1); 
    }
    
    /* Estilo para el contenedor del log de pensamiento */
    .st-emotion-cache-1629p8f pre { 
        white-space: pre-wrap !important; 
        word-wrap: break-word !important; 
    }

    /* Estilo específico para el botón de descarga */
    .stDownloadButton button {
        background-color: #f0f2f6; /* Gris claro, similar al input */
        color: #31333F; /* Color de texto oscuro para contraste */
        border: 1px solid #f0f2f6;
        border-radius: 0.5rem;
        padding: 0.4rem 0.8rem;
    }
    .stDownloadButton button:hover {
        background-color: #e6e8eb; /* Un gris un poco más oscuro al pasar el mouse */
        border: 1px solid #e6e8eb;
        color: #31333F;
    }
</style>
""", unsafe_allow_html=True)

st.title("Autoconsulta IA")
st.caption("Realizado por Business Analytics - BI Chile")

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000/query")

@st.cache_data
def to_excel(df: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Resultados')
    return output.getvalue()

with st.sidebar:
    st.header("Opciones")
    if st.button("🧹 Limpiar Historial de Chat"):
        st.session_state.clear()
        st.rerun()
    
    log_expander = st.expander("Log de Pensamiento del Agente", expanded=False)

if "last_log" in st.session_state and st.session_state.last_log:
    log_expander.code(st.session_state.last_log, language='text')

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hola! te puedo ayudar a descargar bases de datos, cuéntame que necesitas?"}]

for i, msg in enumerate(st.session_state.messages):
    avatar_url = AGENT_AVATAR if msg["role"] == "assistant" else "🧑‍💻"
    with st.chat_message(msg["role"], avatar=avatar_url):
        if "content" in msg:
            st.markdown(msg["content"])
        
        if "answer_text" in msg and msg["answer_text"]:
            st.markdown(msg["answer_text"])
        
        if "table_data" in msg and msg["table_data"]:
            df = pd.DataFrame(msg["table_data"])
            
            # --- CAMBIO PRINCIPAL (2/3): Mostrar estructura de la tabla (columnas x filas) ---
            rows, cols = df.shape
            st.caption(f"{cols} columnas x {rows} filas")
            
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            st.download_button(
                label="Descargar Excel",
                data=to_excel(df),
                file_name=f"resultado_{i}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheet.sheet",
                key=f"download_{i}"
            )

if prompt := st.chat_input("Ej: Dame una base de la familia 0415 con la venta, contribucion y margen en el año 2024, por grupo y conjunto"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.rerun()

if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    user_prompt = st.session_state.messages[-1]["content"]
    
    assistant_message = {"role": "assistant"}
    with st.chat_message("assistant", avatar=AGENT_AVATAR):
        with st.spinner("Analizando, consultando y validando..."):
            try:
                payload = {"question": user_prompt}
                response = requests.post(BACKEND_URL, json=payload, timeout=300)
                response.raise_for_status()
                data = response.json()
                
                assistant_message["answer_text"] = data.get("answer_text")
                assistant_message["table_data"] = data.get("table_data")
                st.session_state.last_log = data.get("reasoning", "No se recibió log del agente.")

            except requests.exceptions.RequestException as e:
                error_message = f"**Error de Conexión:** No se pudo comunicar con el backend.\n\n*Detalles: {e}*"
                st.error(error_message)
                assistant_message["content"] = error_message
                st.session_state.last_log = str(e)
            except Exception as e:
                error_message = f"**Ocurrió un error inesperado:**\n\n*Detalles: {e}*"
                st.error(error_message)
                assistant_message["content"] = error_message
                st.session_state.last_log = str(e)
    
    st.session_state.messages.append(assistant_message)
    st.rerun()

