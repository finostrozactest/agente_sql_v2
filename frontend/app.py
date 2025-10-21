import streamlit as st
import pandas as pd
import requests
import io
import os
import re

# --- 1. CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="Asistente de Datos v3.0",
    page_icon="✅",
    layout="wide"
)

# --- 2. TÍTULOS Y MARCADOR DE VERSIÓN ---
st.title("✅ Asistente de Análisis de Datos con Validación")
st.header("Versión 3.0 - Funcionalidades de UX Mejoradas") # <-- NUEVO MARCADOR VISUAL
st.caption("Impulsado por Google Gemini y LangChain.")

# --- 3. LÓGICA DE LA APLICACIÓN ---
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000/query")

@st.cache_data
def to_excel(df: pd.DataFrame) -> bytes:
    """Convierte un DataFrame a un archivo Excel en memoria."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Resultados')
    return output.getvalue()

def parse_markdown_table_to_df(markdown_text: str):
    """
    Función para encontrar una tabla Markdown en el texto y convertirla a un DataFrame de Pandas.
    Esta es la clave para poder descargar el resultado.
    """
    table_regex = re.compile(r"(\|.*\|(?:\n\|.*\|)+)")
    table_match = table_regex.search(markdown_text)
    if not table_match:
        return None
    
    table_str = table_match.group(0)
    
    try:
        lines = table_str.strip().split("\n")
        if len(lines) > 1 and all(c in '|-: ' for c in lines[1]):
            del lines[1]
        
        csv_like = "\n".join([line.strip().strip('|').replace('|', ',') for line in lines])
        df = pd.read_csv(io.StringIO(csv_like))
        
        # Limpiar espacios en blanco de las cabeceras y datos
        df.columns = df.columns.str.strip()
        df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
        return df
    except Exception:
        return None

# --- 4. INTERFAZ DE USUARIO ---

# --- Barra Lateral (Sidebar) ---
with st.sidebar:
    st.header("Opciones")
    if st.button("🧹 Limpiar Historial de Chat"):
        st.session_state.messages = [{"role": "assistant", "content": "¡Hola! Historial limpiado. ¿En qué puedo ayudarte?"}]
        st.rerun()

    # Contenedor para el log del agente
    st.session_state.log_container = st.container()

# --- Lógica del Chat ---

# Inicializar el historial de mensajes
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "¡Hola! Soy tu asistente de análisis de datos. ¿Qué te gustaría saber?"}]

# Mostrar todos los mensajes del historial
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"], avatar="🧑‍💻" if msg["role"] == "user" else "🤖"):
        # La respuesta completa (incluyendo texto y tabla markdown) se renderiza aquí
        st.markdown(msg["content"])
        
        # Si el mensaje del asistente tiene una tabla, muestra el botón de descarga
        if msg["role"] == "assistant" and "df_for_download" in msg:
            if msg["df_for_download"] is not None:
                st.download_button(
                    label="📥 Descargar Excel",
                    data=to_excel(msg["df_for_download"]),
                    file_name=f"resultado_{i}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheet.sheet",
                    key=f"download_{i}"
                )

# Input del usuario
if prompt := st.chat_input("Ej: ¿Top 10 productos más vendidos en Reino Unido?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.rerun()

# Procesar y mostrar la respuesta del asistente (si el último mensaje es del usuario)
if st.session_state.messages[-1]["role"] == "user":
    user_prompt = st.session_state.messages[-1]["content"]
    
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Analizando..."):
            try:
                payload = {"question": user_prompt}
                response = requests.post(BACKEND_URL, json=payload, timeout=590)
                response.raise_for_status()
                data = response.json()
                
                # El backend ahora nos da el texto, que puede incluir una tabla markdown
                full_answer_text = data.get("answer_text", "No se recibió respuesta.")
                reasoning = data.get("reasoning", "No se recibió log.")
                verdict = data.get("verdict", "No se recibió veredicto.")
                
                # Combinamos la respuesta y el veredicto en un solo bloque de markdown
                final_content = f"{full_answer_text}\n\n---\n\n**Veredicto del Validador:**\n{verdict}"
                
                # Mostramos la respuesta completa
                st.markdown(final_content)
                
                # FUNCIONALIDAD: Parsear la tabla para el botón de descarga
                df_to_download = parse_markdown_table_to_df(full_answer_text)
                if df_to_download is not None:
                    st.download_button(
                        label="📥 Descargar Excel",
                        data=to_excel(df_to_download),
                        file_name="resultado_actual.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheet.sheet",
                        key="download_current"
                    )

                # FUNCIONALIDAD: Mostrar el log del agente con ajuste de texto
                with st.session_state.log_container:
                    with st.expander("Log de Pensamiento del Agente (Última Consulta)", expanded=False):
                        # Este truco de CSS es la solución definitiva para el ajuste de texto
                        st.markdown(f'```text\n{reasoning}\n```')

                # Guardamos el contenido completo y el df para descarga en el historial
                assistant_message = {
                    "role": "assistant",
                    "content": final_content,
                    "df_for_download": df_to_download
                }
                st.session_state.messages.append(assistant_message)

            except requests.exceptions.RequestException as e:
                error_message = f"Error de conexión con el backend: {e}"
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})
            except Exception as e:
                error_message = f"Ocurrió un error inesperado: {e}"
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})

    # No es necesario un st.rerun() aquí, ya que la respuesta se muestra en el mismo flujo
