

# ~/agente_sql/backend/main.py (Versión Final Definitiva - Anti-errores de ruta y CSV)

import re
import io
import os
import pandas as pd
from contextlib import redirect_stdout, asynccontextmanager

# Import crucial para buscar el secreto
from google.cloud import secretmanager

from sqlalchemy import create_engine
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from langchain_community.utilities import SQLDatabase
from langchain_google_genai import ChatGoogleGenera_iveAI
from langchain.agents import create_sql_agent, AgentType
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer_text: str
    table_data: list[dict]
    reasoning: str
    verdict: str

def get_api_key_from_secret_manager():
    """Obtiene la API Key desde Secret Manager al iniciar la app."""
    try:
        project_id = os.getenv('GCP_PROJECT', 'project-agentes')
        secret_id = "gemini-api-key"
        client = secretmanager.SecretManagerServiceClient()
        name = f"projects/{project_id}/secrets/{secret_id}/versions/latest"
        print(f"Accediendo al secreto: {name}")
        response = client.access_secret_version(request={"name": name})
        api_key = response.payload.data.decode("UTF-8")
        os.environ['GOOGLE_API_KEY'] = api_key
        print("API Key cargada exitosamente desde Secret Manager.")
    except Exception as e:
        print(f"Error CRÍTICO al obtener la API Key de Secret Manager: {e}")
        raise RuntimeError("No se pudo obtener la GOOGLE_API_KEY desde Secret Manager.") from e

def load_and_prepare_data():
    """
    Carga y prepara los datos del CSV usando una ruta absoluta para ser
    compatible con cualquier entorno de producción.
    """
    try:
        # --- INICIO DE LA CORRECCIÓN CLAVE ---
        # 1. Obtiene la ruta del directorio donde se encuentra este script (main.py)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        # 2. Construye la ruta completa y correcta al archivo CSV
        local_file = os.path.join(base_dir, "transaccional_dummy.csv")
        # --- FIN DE LA CORRECCIÓN CLAVE ---

        print(f"Cargando datos desde la ruta absoluta: {local_file}")
        if not os.path.exists(local_file):
            raise FileNotFoundError(f"Error CRÍTICO: No se encontró el archivo en la ruta '{local_file}'.")

        try:
            df = pd.read_csv(local_file, sep=None, engine='python', on_bad_lines='skip')
        except UnicodeDecodeError:
            print("Fallo al leer con UTF-8. Reintentando con codificación 'latin-1'.")
            df = pd.read_csv(local_file, sep=None, engine='python', on_bad_lines='skip', encoding='latin-1')
        except Exception as e:
            print(f"Ocurrió un error de parseo inesperado: {e}")
            raise

        print("Datos cargados exitosamente.")
        df.columns = [str(c) for c in df.columns]
        df.columns = df.columns.str.strip().str.lower().str.replace(r'\s+', '_', regex=True).str.replace(r'[^a-zA-Z0-9_]', '', regex=True)
        print("Datos preparados. Nombres de columnas finales:", df.columns.to_list())
        return df
    except Exception as e:
        raise RuntimeError(f"Fallo al cargar o preparar los datos: {e}")

def create_db_engine(df):
    try:
        engine = create_engine("sqlite:///:memory:")
        table_name = "transacciones"
        df.to_sql(table_name, engine, index=False, if_exists="replace")
        print(f"Base de datos en memoria creada con la tabla '{table_name}'.")
        return engine
    except Exception as e:
        raise RuntimeError(f"Fallo al crear el motor de base de datos: {e}")

def parse_response_to_df(response_text: str):
    table_regex = re.compile(r"(\|.*\|(?:\n\|.*\|)+)")
    table_match = table_regex.search(response_text)
    if not table_match: return response_text, []
    table_str = table_match.group(0)
    text_part = response_text.replace(table_str, "").strip()
    try:
        lines = table_str.strip().split("\n")
        if len(lines) > 1 and all(c in '|-: ' for c in lines[1]): del lines[1]
        csv_like = "\n".join([line.strip().strip('|').replace('|', ',') for line in lines])
        df = pd.read_csv(io.StringIO(csv_like), skipinitialspace=True)
        df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
        return text_part, df.to_dict(orient='records')
    except Exception:
        return response_text, []

app_state = {}

class QueryMaster:
    def __init__(self, analyst_agent):
        self.analyst_agent = analyst_agent
    def run_query(self, question: str):
        log_io = io.StringIO()
        try:
            with redirect_stdout(log_io):
                analyst_response = self.analyst_agent.invoke({"input": question})
        except Exception as e:
            print(f"Error en el agente analista: {e}")
            raise
        reasoning_log = log_io.getvalue()
        analyst_answer_raw = analyst_response.get("output", "No se pudo generar una respuesta.")
        text_part, table_data = parse_response_to_df(analyst_answer_raw)
        clean_log = re.sub(r'\x1B(?:[@-Z\\-_]|\[[0-9;?]*[ -/]*[@-~])', '', reasoning_log)
        sql_matches = re.findall(r"Action Input: (SELECT .*?)(?:\n|$)", clean_log, re.DOTALL)
        sql_query = sql_matches[-1].strip() if sql_matches else "No se ejecutó una consulta SQL."
        return { "answer_text": text_part, "table_data": table_data, "reasoning": clean_log, "sql_query": sql_query }

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Iniciando el servidor...")
    get_api_key_from_secret_manager()
    
    data_df = load_and_prepare_data()
    engine = create_db_engine(data_df)
    llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash-lite", temperature=0)
    db = SQLDatabase(engine=engine)
    prefix = """
    Eres un asistente experto en análisis de datos. Tu objetivo es responder preguntas generando y ejecutando consultas SQL.
    REGLAS DE NEGOCIO:
    1. CÁLCULO DE VALORES: Para calcular totales monetarios, infiere las columnas de cantidad y precio y multiplícalas.
    2. FILTROS DE TEXTO: Usa siempre comillas simples para valores de texto en cláusulas WHERE.
    """
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    analyst_agent_executor = create_sql_agent(llm=llm, toolkit=toolkit, verbose=True, prefix=prefix, handle_parsing_errors="Tuve un problema para interpretar la consulta. Por favor, reformula tu pregunta.", agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION)
    validator_template = """
    Valida la siguiente consulta SQL basada en la pregunta del usuario. Responde en español y en una línea.
    Regla: Si la pregunta pide un total de ventas/gasto, la consulta DEBE incluir una multiplicación.
    Pregunta: "{question}"
    Consulta: "{sql_query}"
    Veredicto (APROBADO o RECHAZADO con breve explicación):
    """
    validator_prompt = PromptTemplate(template=validator_template, input_variables=["question", "sql_query"])
    validator_chain = validator_prompt | llm | StrOutputParser()
    app_state['query_master'] = QueryMaster(analyst_agent_executor)
    app_state['validator_chain'] = validator_chain
    print("¡El servidor está listo para recibir peticiones!")
    yield
    app_state.clear()
    print("Apagando el servidor.")

app = FastAPI(lifespan=lifespan)

@app.get("/")
def read_root():
    return {"status": "El backend del Agente de Datos está funcionando."}

# Ruta de salud para App Engine (buena práctica)
@app.get("/_ah/health")
def health_check():
    return {"status": "ok"}

@app.post("/query", response_model=QueryResponse)
async def handle_query(request: QueryRequest):
    query_master = app_state.get('query_master')
    validator_chain = app_state.get('validator_chain')
    if not query_master or not validator_chain: raise HTTPException(status_code=503, detail="El servicio no está listo.")
    if not request.question: raise HTTPException(status_code=400, detail="La pregunta no puede estar vacía.")
    original_question = request.question
    fixed_instruction = "Genera una tabla de datos como resultado. Responde completamente en español, inc
