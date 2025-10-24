# ~/agente_sql/backend/main.py (Versión Final Definitiva - Anti-errores de CSV)

import re
import io
import os
import pandas as pd
from contextlib import redirect_stdout, asynccontextmanager

from sqlalchemy import create_engine
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from langchain_community.utilities import SQLDatabase
from langchain_google_genai import ChatGoogleGenerativeAI
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

def load_and_prepare_data():
    local_file = "transaccional_dummy.csv"
    try:
        print(f"Cargando datos desde el archivo local: {local_file}")
        if not os.path.exists(local_file):
            raise FileNotFoundError(f"Error CRÍTICO: No se encontró el archivo '{local_file}'.")

        # --- ¡SOLUCIÓN DEFINITIVA PARA ERRORES DE PARSEO DE CSV! ---
        # on_bad_lines='skip': Esta es la clave. Ignora cualquier línea que tenga un número incorrecto de columnas.
        # engine='python': Usamos el motor de Python que es más flexible.
        # sep=None: Dejamos que autodetecte el separador (coma o punto y coma).
        # El bloque try/except maneja la codificación de caracteres.
        try:
            df = pd.read_csv(local_file, sep=None, engine='python', on_bad_lines='skip')
        except UnicodeDecodeError:
            print("Fallo al leer con UTF-8. Reintentando con codificación 'latin-1'.")
            df = pd.read_csv(local_file, sep=None, engine='python', on_bad_lines='skip', encoding='latin-1')
        except Exception as e:
            # Captura cualquier otro error de parseo que no sea de codificación
            print(f"Ocurrió un error de parseo inesperado: {e}")
            raise

        print("Datos cargados exitosamente (algunas filas podrían haberse omitido si estaban malformadas).")
        print(f"DataFrame cargado con la forma: {df.shape}")
        
        print("Limpiando nombres de columnas para compatibilidad con SQL.")
        df.columns = [str(c) for c in df.columns]
        df.columns = df.columns.str.strip().str.lower().str.replace(r'\s+', '_', regex=True).str.replace(r'[^a-zA-Z0-9_]', '', regex=True)

        print("Datos preparados. Nombres de columnas finales:", df.columns.to_list())
        return df
    except Exception as e:
        print(f"Error CRÍTICO durante la carga o preparación de datos: {e}")
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
    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError("FATAL: La variable de entorno GOOGLE_API_KEY no se encontró.")
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

@app.post("/query", response_model=QueryResponse)
async def handle_query(request: QueryRequest):
    query_master = app_state.get('query_master')
    validator_chain = app_state.get('validator_chain')
    if not query_master or not validator_chain: raise HTTPException(status_code=503, detail="El servicio no está listo.")
    if not request.question: raise HTTPException(status_code=400, detail="La pregunta no puede estar vacía.")
    original_question = request.question
    fixed_instruction = "Genera una tabla de datos como resultado. Responde completamente en español, incluyendo los encabezados de la tabla. La pregunta es:"
    modified_question = f"{fixed_instruction} {original_question}"
    print(f"\n--- [INPUT ORIGINAL]: {original_question} ---")
    print(f"--- [INPUT MODIFICADO PARA EL AGENTE]: {modified_question} ---")
    try:
        run_result = query_master.run_query(modified_question)
        sql_query = run_result["sql_query"]
        verdict = validator_chain.invoke({"question": original_question, "sql_query": sql_query})
        full_log = f"{run_result['reasoning']}\n--- Validador ---\nPregunta: {original_question}\nConsulta: {sql_query}\nVeredicto: {verdict}"
        final_text = "" if run_result["table_data"] else run_result["answer_text"]
        return QueryResponse(answer_text=final_text, table_data=run_result["table_data"], reasoning=full_log, verdict=verdict)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Ocurrió un error interno en el backend: {e}")


