import streamlit as st
import requests
import pandas as pd
import io
import os

# --- Configuración de la Página ---
st.set_page_config(
    page_title="Agente de Análisis de Datos",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Estilo CSS para un look futurista ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&display=swap');

    body {
        color: #e0e0e0;
        background-color: #10101a;
    }
    .stApp {
        background: url("https://www.transparenttextures.com/patterns/cubes.png");
        background-color: #10101a;
    }
    .stTextInput > div > div > input, .stTextArea > div > div > textarea {
        background-color: #1e1e2f;
        color: #00ffcc;
        border: 1px solid #00ffcc;
        border-radius: 10px;
        font-family: 'Orbitron', sans-serif;
    }
    .stButton > button {
        background-color: #00ffcc;
        color: #10101a;
        border: none;
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: bold;
        font-family: 'Orbitron', sans-serif;
        box-shadow: 0 0 15px #00ffcc;
        transition: all 0.3s ease-in-out;
    }
    .stButton > button:hover {
        box-shadow: 0 0 25px #00ffcc, 0 0 5px #ffffff;
        transform: scale(1.05);
    }
    h1, h2, h3 {
        font-family: 'Orbitron', sans-serif;
        color: #00ffcc;
        text-shadow: 0 0 10px #00ffcc;
    }
    .stExpander {
        background-color: #1e1e2f;
        border: 1px solid #00ffcc;
        border-radius: 10px;
    }
    .stDataFrame {
        border: 2px solid #00ffcc;
        border-radius: 10px;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)


# --- Lógica de la Aplicación ---

# Obtener la URL del backend desde una variable de entorno, con un valor por defecto para desarrollo local
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000/query")

# Función para convertir DataFrame a Excel en memoria
@st.cache_data
def to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Resultados')
    processed_data = output.getvalue()
    return processed_data

# --- Interfaz de Usuario ---
st.title("🤖 Agente Interactivo de Análisis de Datos")
st.markdown("### Haz una pregunta en lenguaje natural sobre la base de datos de transacciones de E-commerce.")

# Inicializar estado de sesión
if 'response_data' not in st.session_state:
    st.session_state.response_data = None

# Área para la pregunta del usuario
with st.form(key='query_form'):
    user_question = st.text_area(
        "Escribe tu pregunta aquí:",
        height=100,
        placeholder="Ej: ¿Cuáles son los 5 productos más vendidos en Francia?"
    )
    submit_button = st.form_submit_button(label='Preguntar al Agente')


# Procesamiento al enviar la pregunta
if submit_button and user_question:
    with st.spinner('El agente está pensando... 🧠'):
        try:
            payload = {"question": user_question}
            response = requests.post(BACKEND_URL, json=payload, timeout=300)
            
            if response.status_code == 200:
                st.session_state.response_data = response.json()
            else:
                st.error(f"Error del servidor ({response.status_code}): {response.text}")
                st.session_state.response_data = None

        except requests.exceptions.RequestException as e:
            st.error(f"Error de conexión con el backend: {e}")
            st.session_state.response_data = None

# Mostrar resultados si existen
if st.session_state.response_data:
    res = st.session_state.response_data
    
    st.divider()

    # Sección de Respuesta Final
    st.subheader("Respuesta del Agente")
    st.info(res['answer'])

    # Sección de Razonamiento
    with st.expander("Ver el razonamiento del Agente (Pensamiento y SQL) ⚙️"):
        st.code(res['reasoning'], language='text')
    
    # Sección de Datos y Descarga
    if res['table_data']:
        st.subheader("Datos Resultantes")
        df = pd.DataFrame(res['table_data'])
        st.dataframe(df, use_container_width=True)
        
        excel_data = to_excel(df)
        st.download_button(
            label="📥 Descargar como Excel",
            data=excel_data,
            file_name="resultados_agente.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    else:
        st.warning("El agente no devolvió datos tabulares para esta pregunta.")