# ~/agente_sql/backend/main.py (Versión Final y Corregida)

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

# --- MODELOS DE DATOS ---
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer_text: str
    table_data: list[dict]
    reasoning: str
    verdict: str

# --- FUNCIONES DE UTILIDAD ---
def load_and_prepare_data():
    """Carga y prepara los datos del e-commerce desde la URL pública."""
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
    """Crea una base de datos SQLite en memoria a partir de un DataFrame."""
    try:
        engine = create_engine("sqlite:///:memory:")
        df.to_sql("transacciones", engine, index=False, if_exists="replace")
        print("Base de datos en memoria creada y poblada.")
        return engine
    except Exception as e:
        print(f"Error al crear la base de datos en memoria: {e}")
        return None

# --- FUNCIÓN QUE FALTABA Y CAUSABA EL NameError ---
def parse_response_to_df(response_text: str):
    """Extrae texto y una tabla Markdown de la respuesta y la convierte a lista de diccionarios."""
    table_regex = re.compile(r"(\|.*\|(?:\n\|.*\|)+)")
    table_match = table_regex.search(response_text)
    
    if not table_match:
        return response_text, []

    table_str = table_match.group(0)
    text_part = response_text.replace(table_str, "").strip()

    try:
        lines = table_str.strip().split("\n")
        if len(lines) > 1 and all(c in '|-: ' for c in lines[1]):
            del lines[1]
        
        csv_like = "\n".join([line.strip().strip('|').replace('|', ',') for line in lines])
        df = pd.read_csv(io.StringIO(csv_like), skipinitialspace=True)
        df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
        return text_part, df.to_dict(orient='records')
    except Exception as e:
        print(f"Error al parsear la tabla markdown: {e}")
        return response_text, []

# --- LÓGICA PRINCIPAL DEL AGENTE ---
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
        answer_text, table_data = parse_response_to_df(analyst_answer_raw)
        
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        clean_log = ansi_escape.sub('', log_io.getvalue())

        sql_matches = re.findall(r"Action Input: (SELECT .*?)(?:\n|$)", clean_log, re.DOTALL)
        sql_query = sql_matches[-1].strip() if sql_matches else "No se ejecutó una consulta SQL directa."

        verdict = self.validator_chain.invoke({
            "question": question,
            "sql_query": sql_query,
            "agent_response": analyst_answer_raw
        })
        print(f"\n--- [Veredicto del Validador] ---\n{verdict}")

        return {
            "answer_text": analyst_answer_raw,
            "table_data": table_data,
            "reasoning": clean_log,
            "verdict": verdict
        }
        
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Iniciando el servidor...")
    
    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError("FATAL: La variable de entorno GOOGLE_API_KEY no se encontró.")
    
    ecommerce_data = load_and_prepare_data()
    if ecommerce_data is None: 
        raise RuntimeError("FATAL: No se pudieron cargar los datos, el backend no puede iniciar.")

    engine = create_db_engine(ecommerce_data)
    if engine is None: 
        raise RuntimeError("FATAL: No se pudo crear la base de datos, el backend no puede iniciar.")

    llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash-lite", temperature=0)
    db = SQLDatabase(engine=engine)

    prefix = """
    Eres un asistente experto en análisis de datos que trabaja con una base de datos SQLite.
    La base de datos contiene una única tabla llamada 'transacciones' con información de ventas de e-commerce.
    Tu objetivo es responder a las preguntas del usuario generando y ejecutando consultas SQL.
    Regla de negocio importante: Para calcular el total de ventas o el gasto, siempre debes multiplicar la columna 'Quantity' por 'UnitPrice'.
    Cuando la respuesta sea una tabla de datos, preséntala en formato Markdown.
    """
    
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    analyst_agent_executor = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=True,
        prefix=prefix,
        handle_parsing_errors="Tuve un problema para interpretar la consulta. Por favor, reformula tu pregunta.",
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        max_iterations=10
    )

    validator_template = """
    Eres un experto en SQL... (tu template de validación va aquí)
    """
    validator_prompt = PromptTemplate(template=validator_template, input_variables=["question", "sql_query", "agent_response"])
    validator_chain = validator_prompt | llm | StrOutputParser()

    app_state['query_master'] = QueryMaster(analyst_agent_executor, validator_chain)
    print("¡El servidor está listo para recibir peticiones!")
    yield
    print("Apagando el servidor.")
    app_state.clear()


# --- RUTAS DE LA API ---
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
