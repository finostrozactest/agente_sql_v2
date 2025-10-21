import os
import pandas as pd
import re
import io
from contextlib import redirect_stdout
from sqlalchemy import create_engine
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import functools

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_sql_agent, AgentExecutor
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase

# --- Modelos de Datos Pydantic para la API ---
class QueryRequest(BaseModel):
    question: str

# El backend ahora devuelve una estructura más simple
class QueryResponse(BaseModel):
    answer_text: str # Contendrá texto y la tabla markdown juntos
    reasoning: str
    verdict: str

# --- Lógica de la Aplicación ---

@functools.lru_cache(maxsize=None)
def load_and_prepare_data():
    """Carga, limpia y prepara los datos para la base de datos."""
    print("Iniciando la carga y preparación de datos...")
    ecommerce_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
    try:
        df = pd.read_excel(ecommerce_url)
        df.dropna(subset=['CustomerID'], inplace=True)
        df = df[df['Quantity'] > 0]
        df['CustomerID'] = df['CustomerID'].astype(int)
        df.rename(columns={'InvoiceNo': 'InvoiceID', 'StockCode': 'StockCode', 'Description': 'Description', 'Quantity': 'Quantity', 'InvoiceDate': 'InvoiceDate', 'UnitPrice': 'UnitPrice', 'CustomerID': 'CustomerID', 'Country': 'Country'}, inplace=True)
        print("Limpieza de datos completada.")
        return df
    except Exception as e:
        print(f"Error crítico al cargar o limpiar los datos: {e}")
        return None

def create_db_engine(df):
    if df is None: return None
    try:
        engine = create_engine("sqlite:///:memory:")
        df.to_sql("transacciones", engine, index=False, if_exists="replace")
        print("Base de datos en memoria creada y poblada.")
        return engine
    except Exception as e:
        print(f"Error al crear la base de datos en memoria: {e}")
        return None

app_state = {}

class QueryMaster:
    def __init__(self, analyst_agent: AgentExecutor, validator_chain):
        self.analyst_agent = analyst_agent
        self.validator_chain = validator_chain

    def run_query(self, question: str):
        print("--- [Paso 1: Iniciando Agente Analista] ---")
        
        log_io = io.StringIO()
        try:
            with redirect_stdout(log_io):
                analyst_response = self.analyst_agent.invoke({"input": question})
            log_output = log_io.getvalue()
        except Exception as e:
            print(f"Error en el agente analista: {e}")
            raise

        # La respuesta del agente ya contiene el texto y la tabla markdown juntos
        analyst_answer_raw = analyst_response.get("output", "No se pudo generar una respuesta.")
        
        print("\n--- [Paso 2: Iniciando Agente Validador] ---")
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        clean_log = ansi_escape.sub('', log_output)
        
        sql_matches = re.findall(r"Action Input: (SELECT .*?)(?:\n|$)", clean_log, re.DOTALL)
        sql_query = sql_matches[-1].strip() if sql_matches else "No se ejecutó una consulta SQL directa."

        verdict = self.validator_chain.invoke({
            "user_question": question,
            "sql_query": sql_query,
            "analyst_answer": analyst_answer_raw
        })
        print(f"\n--- [Veredicto del Validador] ---\n{verdict}")

        # Devolvemos la respuesta cruda del agente, que es lo que el frontend espera
        return {
            "reasoning": clean_log,
            "answer_text": analyst_answer_raw,
            "verdict": verdict
        }

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Iniciando el servidor FastAPI...")
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key: raise ValueError("La variable de entorno GOOGLE_API_KEY no está configurada.")
    os.environ["GOOGLE_API_KEY"] = google_api_key

    ecommerce_data = load_and_prepare_data()
    if ecommerce_data is None: raise RuntimeError("No se pudieron cargar los datos.")
    
    engine = create_db_engine(ecommerce_data)
    if engine is None: raise RuntimeError("No se pudo crear la base de datos.")

    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0)
    db = SQLDatabase(engine=engine)
    
    prefix = """
    Eres un analista de datos experto que trabaja con una base de datos SQLite.
    Tu objetivo es responder a las preguntas del usuario generando y ejecutando consultas SQL.
    Regla de negocio importante: Para calcular el total de ventas o el gasto, siempre debes multiplicar la columna 'Quantity' por 'UnitPrice'.
    Cuando la respuesta sea una tabla de datos, preséntala en formato Markdown.
    Siempre responde en español.
    """
    
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    analyst_agent_executor = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=True,
        prefix=prefix,
        handle_parsing_errors="Tuve un problema para interpretar la consulta. Por favor, reformula tu pregunta.",
        max_iterations=10
    )

    validator_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template("Eres un analista de datos experto y validador."),
        HumanMessagePromptTemplate.from_template("Pregunta: {user_question}\nConsulta SQL: {sql_query}\nRespuesta del Agente: {analyst_answer}\n\n**Veredicto (sé breve, directo y responde en español):**")
    ])
    validator_chain = validator_prompt | llm | StrOutputParser()
    
    app_state['query_master'] = QueryMaster(analyst_agent_executor, validator_chain)
    print("¡El servidor está listo!")
    yield
    print("Apagando el servidor.")
    app_state.clear()


app = FastAPI(lifespan=lifespan)

@app.post("/query", response_model=QueryResponse)
async def handle_query(request: QueryRequest):
    print(f"\n--- [NUEVA PETICIÓN]: {request.question} ---")
    query_master = app_state.get('query_master')
    if not query_master:
        raise HTTPException(status_code=503, detail="El servicio no está listo.")
    try:
        result = query_master.run_query(request.question)
        return QueryResponse(**result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Ocurrió un error interno en el backend: {e}")
