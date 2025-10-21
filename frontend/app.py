import streamlit as st
import pandas as pd
import requests
import io
import os

# --- Configuración de la Página de Streamlit ---
st.set_page_config(
    page_title="Asistente de Análisis de Datos",
    page_icon="✅",
    layout="wide"
)

# --- Estilos CSS ---
st.markdown("""
<style>
    /* Estilo para los mensajes del chat */
    .stChatMessage {
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.1);
    }
    .st-emotion-cache-janbn0 { /* Clases específicas de Streamlit para ajustar sombras */
        box-shadow: none;
    }
    .st-emotion-cache-4oy321 {
        border-radius: 10px;
    }
    
    /* --- ¡SOLUCIÓN PARA EL AJUSTE DE TEXTO! --- */
    /* Esto fuerza al texto dentro de bloques de código (como el log) a ajustarse */
    pre {
        white-space: pre-wrap !important;
        word-wrap: break-word !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("✅ Asistente de Análisis de Datos con Validación")
st.caption("Impulsado por Google Gemini y LangChain. Rápido, preciso y con doble comprobación.")

# --- Lógica de la Aplicación ---
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000/query")

@st.cache_data
def to_excel(df_to_convert):
    """Convierte un DataFrame a un archivo Excel en memoria."""
    output = io.BytesIO()
    # Usamos openpyxl como motor, que es moderno y robusto
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_to_convert.to_excel(writer, index=False, sheet_name='Resultado')
    return output.getvalue()

# --- Interfaz de Usuario ---

# --- BARRA LATERAL (SIDEBAR) ---
with st.sidebar:
    st.header("Opciones")
    if st.button("🧹 Limpiar Historial de Chat"):
        st.session_state.messages = [{"role": "assistant", "content": "¡Hola! El historial ha sido limpiado. ¿En qué te puedo ayudar ahora?"}]
        st.rerun()
    
    # El expansor para el log se crea aquí, pero se llenará más tarde
    st.session_state.log_expander = st.expander("Log de Pensamiento del Agente (Última Consulta)", expanded=False)


# --- LÓGICA DEL CHAT ---

# Inicializar historial de chat si no existe
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "¡Hola! Soy tu asistente de análisis de datos. ¿Qué te gustaría saber?"}]

# Mostrar todos los mensajes del historial en cada recarga
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"], avatar="🧑‍💻" if msg["role"] == "user" else "🤖"):
        
        # Muestra el contenido principal del mensaje
        if "text" in msg:
            st.write(msg["text"])
            
        # --- FUNCIONALIDAD: TABLA INTERACTIVA Y BOTÓN DE EXCEL ---
        if "df" in msg and msg["df"] is not None and msg["df"]:
            df_to_show = pd.DataFrame(msg["df"])
            # st.dataframe es el componente que crea la tabla interactiva (ordenable)
            st.dataframe(df_to_show, use_container_width=True)
            # El botón de descarga se crea para cada resultado que contenga una tabla
            st.download_button(
                label="📥 Descargar Resultado (Excel)",
                data=to_excel(df_to_show),
                file_name=f"resultado_consulta_{i}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheet.sheet",
                key=f"download_{i}" # Una clave única es importante para cada botón
            )
            
        # Para el mensaje de bienvenida inicial que no tiene 'text' ni 'df'
        elif "content" in msg:
            st.write(msg["content"])
        
        # Muestra el veredicto del validador si existe
        if "verdict" in msg:
            st.info(f"**Veredicto del Validador:**\n{msg['verdict']}")

# Input del usuario al final de la página
if prompt := st.chat_input("Ej: ¿Cuáles son los 5 productos más vendidos y sus cantidades?"):
    # Añadir y mostrar el mensaje del usuario inmediatamente
    st.session_state.messages.append({"role": "user", "text": prompt})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.write(prompt)

    # Preparar un placeholder para la respuesta del asistente
    assistant_message = {"role": "assistant"}
    st.session_state.messages.append(assistant_message)
    
    # Mostrar la respuesta del asistente
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Consultando la base de datos y analizando..."):
            try:
                payload = {"question": prompt}
                response = requests.post(BACKEND_URL, json=payload, timeout=590)
                response.raise_for_status() # Lanza un error si la respuesta no es 200 (ej. 404, 500)
                
                data = response.json()
                answer_text = data.get("answer_text", "No se recibió texto de respuesta.")
                df_result_list = data.get("table_data", [])
                reasoning = data.get("reasoning", "No se recibió el log de razonamiento.")
                verdict = data.get("verdict", "No se recibió el veredicto.")

                # 1. Mostrar la respuesta principal del agente
                st.write(answer_text)
                
                # 2. Mostrar la tabla y el botón de descarga si hay datos
                if df_result_list:
                    df_to_show = pd.DataFrame(df_result_list)
                    st.dataframe(df_to_show, use_container_width=True)
                    st.download_button(
                        label="📥 Descargar Resultado (Excel)",
                        data=to_excel(df_to_show),
                        file_name="resultado_consulta_actual.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheet.sheet",
                        key="download_current"
                    )

                # 3. Mostrar el veredicto del validador
                st.info(f"**Veredicto del Validador:**\n{verdict}")

                # 4. Actualizar el log en la barra lateral
                st.session_state.log_expander.code(reasoning, language='text')

                # 5. Guardar la respuesta completa en el historial de sesión para futuras recargas
                assistant_message["text"] = answer_text
                assistant_message["df"] = df_result_list
                assistant_message["verdict"] = verdict

            except requests.exceptions.RequestException as e:
                error_message = f"Error de conexión con el backend: {e}"
                st.error(error_message)
                assistant_message["text"] = error_message
            except Exception as e:
                error_message = f"Ocurrió un error inesperado: {e}"
                st.error(error_message)
                assistant_message["text"] = error_message

