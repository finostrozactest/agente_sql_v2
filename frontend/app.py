# ~/agente_sql/frontend/app.py (Versión Final y Mejorada)

import streamlit as st
import pandas as pd
import requests
import io
import os

# --- 1. CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="Asistente de Datos v2.0",
    page_icon="✅",
    layout="wide"
)

# --- 2. ESTILOS CSS PERSONALIZADOS ---
st.markdown("""
<style>
    .stChatMessage {
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.1);
    }
    .st-emotion-cache-1629p8f pre {
        white-space: pre-wrap !important;
        word-wrap: break-word !important;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. TÍTULOS Y CABECERA ---
st.title("✅ Asistente de Análisis de Datos con Validación")
st.header("Versión 2.0 - Interfaz de Chat Mejorada")
st.caption("Impulsado por Google Gemini, LangChain y Cloud Run.")

# --- 4. LÓGICA DE LA APLICACIÓN ---
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000/query")

@st.cache_data
def to_excel(df: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Resultados')
    return output.getvalue()

# --- 5. INTERFAZ DE USUARIO ---

with st.sidebar:
    st.header("Opciones")
    if st.button("🧹 Limpiar Historial de Chat"):
        st.session_state.messages = [{"role": "assistant", "content": "¡Hola! El historial ha sido limpiado. ¿En qué puedo ayudarte ahora?"}]
        st.rerun()
    st.session_state.log_expander = st.expander("Log de Pensamiento del Agente (Última Consulta)", expanded=False)

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "¡Hola! Soy tu asistente. La base de datos está lista. ¿Qué te gustaría saber?"}]

for i, msg in enumerate(st.session_state.messages):
    avatar = "🧑‍💻" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        if "content" in msg:
            st.markdown(msg["content"])
        
        if "df_data" in msg and msg["df_data"]:
            df = pd.DataFrame(msg["df_data"])
            
            # --- NUEVA FUNCIONALIDAD: Mostrar el total de filas ---
            st.caption(f"Mostrando {len(df)} filas.")
            
            # --- FUNCIONALIDAD MEJORADA: Tabla interactiva tipo BigQuery ---
            st.dataframe(df, use_container_width=True)
            
            st.download_button(
                label="📥 Descargar Excel",
                data=to_excel(df),
                file_name=f"resultado_{i}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheet.sheet",
                key=f"download_{i}"
            )

        if "verdict" in msg and msg["verdict"]:
            st.info(f"**Veredicto del Validador:**\n{msg['verdict']}")

if prompt := st.chat_input("Ej: ¿Top 5 clientes en Francia por gasto total?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.rerun()

if st.session_state.messages[-1]["role"] == "user":
    user_prompt = st.session_state.messages[-1]["content"]
    
    assistant_message = {"role": "assistant"}
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Consultando la base de datos y validando..."):
            try:
                payload = {"question": user_prompt}
                response = requests.post(BACKEND_URL, json=payload, timeout=300) # Timeout de 5 minutos
                response.raise_for_status()
                data = response.json()
                
                assistant_message["content"] = data.get("answer_text", "No se recibió texto.")
                assistant_message["df_data"] = data.get("table_data", [])
                assistant_message["verdict"] = data.get("verdict", "")
                
                st.session_state.log_expander.code(data.get("reasoning", ""), language='text')

            except requests.exceptions.RequestException as e:
                error_message = f"**Error de Conexión:** No se pudo comunicar con el servicio de backend. Por favor, verifica que esté funcionando.\n\n*Detalles: {e}*"
                st.error(error_message)
                assistant_message["content"] = error_message
            except Exception as e:
                error_message = f"**Ocurrió un error inesperado:**\n\n*Detalles: {e}*"
                st.error(error_message)
                assistant_message["content"] = error_message
    
    st.session_state.messages.append(assistant_message)
    st.rerun()
