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
    /* Estilo general para los mensajes del chat */
    .stChatMessage {
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* ¡SOLUCIÓN DEFINITIVA PARA EL AJUSTE DE TEXTO EN EL LOG! */
    /* Se aplica a los bloques de código (pre) dentro del sidebar de Streamlit */
    .st-emotion-cache-1629p8f pre {
        white-space: pre-wrap !important;
        word-wrap: break-word !important;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. TÍTULOS Y MARCADOR DE VERSIÓN ---
st.title("✅ Asistente de Análisis de Datos con Validación")
st.header("Versión 2.0 - Interfaz de Chat Mejorada") # <-- MARCADOR VISUAL
st.caption("Impulsado por Google Gemini y LangChain.")

# --- 4. LÓGICA DE LA APLICACIÓN ---
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000/query")

# Función para convertir DataFrame a Excel (cacheada para eficiencia)
@st.cache_data
def to_excel(df: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Resultados')
    return output.getvalue()

# --- 5. INTERFAZ DE USUARIO ---

# --- Barra Lateral (Sidebar) ---
with st.sidebar:
    st.header("Opciones")
    if st.button("🧹 Limpiar Historial de Chat"):
        st.session_state.messages = [{"role": "assistant", "content": "¡Hola! El historial ha sido limpiado. ¿En qué puedo ayudarte?"}]
        st.rerun()
    
    # Se crea el contenedor para el log aquí, se llenará después
    st.session_state.log_expander = st.expander("Log de Pensamiento del Agente", expanded=False)

# --- Lógica del Chat ---

# Inicializar el historial de mensajes si no existe
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "¡Hola! Soy tu asistente. La base de datos está lista. ¿Qué te gustaría saber?"}]

# Mostrar todos los mensajes del historial
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"], avatar="🧑‍💻" if msg["role"] == "user" else "🤖"):
        # Contenido del mensaje: texto, tabla y botón de descarga
        if "text" in msg:
            st.write(msg["text"])
        
        if "df_data" in msg and msg["df_data"]:
            df = pd.DataFrame(msg["df_data"])
            # FUNCIONALIDAD: Tabla interactiva tipo BigQuery
            st.dataframe(df, use_container_width=True)
            # FUNCIONALIDAD: Botón de descarga para cada tabla
            st.download_button(
                label="📥 Descargar Excel",
                data=to_excel(df),
                file_name=f"resultado_{i}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheet.sheet",
                key=f"download_{i}"
            )
            
        if "content" in msg:
            st.write(msg["content"])
        
        # Mostrar el veredicto del validador
        if "verdict" in msg:
            st.info(f"**Veredicto del Validador:**\n{msg['verdict']}")

# Input del usuario
if prompt := st.chat_input("Ej: ¿Top 5 clientes en Francia por gasto total?"):
    st.session_state.messages.append({"role": "user", "text": prompt})
    st.rerun() # Recargar la app para mostrar el mensaje del usuario inmediatamente

# Procesar y mostrar la respuesta del asistente (si el último mensaje es del usuario)
if st.session_state.messages[-1]["role"] == "user":
    user_prompt = st.session_state.messages[-1]["text"]
    
    assistant_message = {"role": "assistant"}
    
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Analizando..."):
            try:
                payload = {"question": user_prompt}
                response = requests.post(BACKEND_URL, json=payload, timeout=590)
                response.raise_for_status()
                data = response.json()
                
                # Guardar los datos en la estructura del mensaje
                assistant_message["text"] = data.get("answer_text")
                assistant_message["df_data"] = data.get("table_data")
                assistant_message["verdict"] = data.get("verdict")
                
                # Actualizar el log en el sidebar
                st.session_state.log_expander.code(data.get("reasoning", "No se recibió log."), language='text')

            except requests.exceptions.RequestException as e:
                assistant_message["text"] = f"Error de conexión: {e}"
            except Exception as e:
                assistant_message["text"] = f"Error inesperado: {e}"
                
    st.session_state.messages.append(assistant_message)
    st.rerun() # Recargar para mostrar la respuesta completa del asistente
