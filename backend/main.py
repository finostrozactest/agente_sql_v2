# ~/agente_sql/backend/main.py (Versión Final Definitiva y Robusta)

import re
import io
import os
import pandas as pd
import ast
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
    try:
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
        print(f"Cargando datos desde la URL: {url}")
        df = pd.read_excel(url)
        print("Datos cargados. Iniciando limpieza...")
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        df.dropna(subset=['CustomerID'], inplace=True)
        df['CustomerID'] = df['CustomerID'].astype(int)
        print("Datos cargados y preparados exitosamente.")
        return df
    except Exception as e:
        print(f"Error CRÍTICO durante la carga de datos: {e}")
        return None

def create_db_engine(df):
    try:
        engine = create_engine("sqlite:///:memory:")
        df.to_sql("transacciones", engine, index=False, if_exists="replace")
        print("Base de datos en memoria creada y poblada.")
        return engine
    except Exception as e:
        print(f"Error al crear la base de datos en memoria: {e}")
        return None

def extract_sql_observation(log: str, query: str) -> pd.DataFrame:
    """
    Estrategia robusta: Extrae la observación de datos y los nombres de las columnas
    directamente del log de pensamiento del agente.
    """
    try:
        observation_match = re.search(r"Observation:\s*(\[.*?\])", log, re.DOTALL)
        if not observation_match:
            return pd.DataFrame()

        data_str = observation_match.group(1)
        data = ast.literal_eval(data_str)

        columns_match = re.search(r"SELECT\s+(.*?)\s+FROM", query, re.IGNORECASE | re.DOTALL)
        if not columns_match:
            return pd.DataFrame()

        columns_str = columns_match.group(1).replace('"', '')
        columns = [col.split(' as ')[-1].strip() for col in columns_str.split(',')]

        return pd.DataFrame(data, columns=columns)
    except Exception as e:
        print(f"No se pudo extraer la tabla del log, se usará la respuesta final. Error: {e}")
        return pd.DataFrame()

app_state = {}

class QueryMaster:
    def __init__(self, analyst_agent, validator_chain):
        self.analyst_agent = analyst_agent
        self.validator_chain = validator_chain

    def run_query(self, question: str):
        log_io = io.StringIO()
        try:
            with redirect_stdout(log_io):
                analyst_response = self.analyst_agent.invoke({"input": question})
        except Exception as e:
            print(f"Error en el agente analista: {e}")
            raise HTTPException(status_code=500, detail=f"El agente analista falló: {e}")

        analyst_answer_raw = analyst_response.get("output", "No se pudo generar una respuesta.")
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-9;?]*[ -/]*[@-~])')
        clean_log = ansi_escape.sub('', log_io.getvalue())

        sql_matches = re.findall(r"Action Input: (SELECT .*?)(?:\n|$)", clean_log, re.DOTALL)
        sql_query = sql_matches[-1].strip() if sql_matches else "No se ejecutó una consulta SQL directa."

        verdict = self.validator_chain.invoke({ "question": question, "sql_query": sql_query })
        
        full_log = f"{clean_log}\n--- Interacción con el Validador ---\nPregunta: {question}\nConsulta: {sql_query}\nVeredicto: {verdict}"
        
        df = extract_sql_observation(clean_log, sql_query)
        
        if not df.empty:
            answer_text = f"He encontrado {len(df)} resultados para tu consulta."
            table_data = df.to_dict(orient='records')
        else:
            answer_text = analyst_answer_raw
            table_data = []

        return {
            "answer_text": answer_text,
            "table_data": table_data,
            "reasoning": full_log,
            "verdict": verdict
        }
        
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Iniciando el servidor...")
    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError("FATAL: La variable de entorno GOOGLE_API_KEY no se encontró.")
    
    ecommerce_data = load_and_prepare_data()
    if ecommerce_data is None: raise RuntimeError("FATAL: No se pudieron cargar los datos.")

    engine = create_db_engine(ecommerce_data)
    if engine is None: raise RuntimeError("FATAL: No se pudo crear la base de datos.")

    llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash-lite", temperature=0)
    db = SQLDatabase(engine=engine)

    prefix = """
    Eres un asistente experto en análisis de datos que trabaja con una base de datos SQLite. Tu objetivo es responder a las preguntas del usuario generando y ejecutando consultas SQL.
    REGLAS ESTRICTAS:
    1. CÁLCULO DE VENTAS: Para calcular el total de ventas o el gasto, SIEMPRE debes multiplicar 'Quantity' por 'UnitPrice'.
    2. FILTROS DE TEXTO: Cuando filtres por un valor de texto (ej. un país o un nombre), SIEMPRE debes usar comillas simples alrededor del valor en la cláusula WHERE. Ejemplo: `WHERE Country = 'Brazil'`.
    3. Tu respuesta final debe estar COMPLETAMENTE en español.
    """
    
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    analyst_agent_executor = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=True,
        prefix=prefix,
        handle_parsing_errors="Tuve un problema para interpretar la consulta. Por favor, reformula tu pregunta en español.",
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    )

    validator_template = """
    Tu única tarea es actuar como un validador de consultas SQL. Basado en la pregunta del usuario y la consulta SQL generada, proporciona un veredicto en UNA SOLA LÍNEA y en español.
    Regla de negocio: Si la pregunta implica calcular ventas, la consulta DEBE multiplicar 'Quantity' por 'UnitPrice'.
    Pregunta: "{question}"
    Consulta: "{sql_query}"
    Evalúa si la consulta responde correctamente a la pregunta y cumple la regla de negocio.
    Responde ÚNICAMENTE con "APROBADO" si es correcta, o "RECHAZADO" con una explicación técnica muy breve (máximo 10 palabras).
    Veredicto:
    """
    validator_prompt = PromptTemplate(template=validator_template, input_variables=["question", "sql_query"])
    validator_chain = validator_prompt | llm | StrOutputParser()

    app_state['query_master'] = QueryMaster(analyst_agent_executor, validator_chain)
    print("¡El servidor está listo para recibir peticiones!")
    yield
    app_state.clear()

app = FastAPI(lifespan=lifespan)

@app.get("/")
def read_root():
    return {"status": "El backend del Agente de Datos está funcionando."}

@app.post("/query", response_model=QueryResponse)
async def handle_query(request: QueryRequest):
    print(f"\n--- [NUEVA PETICIÓN]: {request.question} ---")
    query_master = app_state.get('query_master')
    if not query_master:
        raise HTTPException(status_code=503, detail="El servicio no está listo.")
    if not request.question:
        raise HTTPException(status_code=400, detail="La pregunta no puede estar vacía.")
    try:
        result = query_master.run_query(request.question)
        return QueryResponse(**result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Ocurrió un error interno en el backend: {e}")


