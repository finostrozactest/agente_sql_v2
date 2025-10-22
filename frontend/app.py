# ~/agente_sql/frontend/app.py (Versión Final 100% Completa)

import streamlit as st
import pandas as pd
import requests
import io
import os

st.set_page_config(page_title="Asistente de Datos v2.0", page_icon="✅", layout="wide")

st.markdown("""
<style>
    .stChatMessage { border-radius: 10px; padding: 1rem; margin-bottom: 1rem; box-shadow: 0 4px 8px 0 rgba(0,0,0,0.1); }
    .st-emotion-cache-1629p8f pre { white-space: pre-wrap !important; word-wrap: break-word !important; }
</style>
""", unsafe_allow_html=True)

st.title("✅ Asistente de Análisis de Datos con Validación")
st.caption("Impulsado por Google Gemini, LangChain y Cloud Run.")

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
    st.session_state.messages = [{"role": "assistant", "content": "¡Hola! Soy tu asistente. La base de datos está lista. ¿Qué te gustaría saber?"}]

for i, msg in enumerate(st.session_state.messages):
    avatar = "🧑‍💻" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        if "content" in msg:
            st.markdown(msg["content"])
        
        if "answer_text" in msg and msg["answer_text"]:
            st.markdown(msg["answer_text"])
        
        if "table_data" in msg and msg["table_data"]:
            df = pd.DataFrame(msg["table_data"])
            st.caption(f"Mostrando {len(df)} filas.")
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            st.download_button(
                label="📥 Descargar Resultado (Excel)",
                data=to_excel(df),
                file_name=f"resultado_{i}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheet.sheet",
                key=f"download_{i}"
            )

if prompt := st.chat_input("Ej: ¿Top 5 clientes en Francia por gasto total?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.rerun()

if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    user_prompt = st.session_state.messages[-1]["content"]
    
    assistant_message = {"role": "assistant"}
    with st.chat_message("assistant", avatar="🤖"):
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

