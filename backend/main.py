# main.py (Final, con tu lógica de carga de datos)

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
# main.py -> REEMPLAZA TU FUNCIÓN con esta para la prueba

def load_and_prepare_data():
    """
    FUNCIÓN DE PRUEBA: Carga datos desde una URL pública conocida.
    Si el deploy funciona con esto, el problema está en tu lógica de carga original.
    """
    try:
        # Esta es una fuente de datos pública y clásica para e-commerce
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
        print(f"Cargando datos de prueba desde: {url}")
        df = pd.read_excel(url)
        
        # Limpieza básica de los datos de prueba
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        df.dropna(subset=['CustomerID'], inplace=True)
        df['CustomerID'] = df['CustomerID'].astype(int)
        print("Datos de PRUEBA cargados y preparados exitosamente.")
        return df
    except Exception as e:
        print(f"Error CRÍTICO durante la carga de datos de PRUEBA: {e}")
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

# ... (El resto del código de `parse_response_to_df`, `QueryMaster`, etc. no necesita cambios y puede permanecer como en la respuesta anterior) ...
# Pega aquí el resto de las funciones: parse_response_to_df, QueryMaster, etc.
# El resto del código de la respuesta anterior es correcto.

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
            log_output = log_io.getvalue()
        except Exception as e:
            print(f"Error en el agente analista: {e}")
            raise HTTPException(status_code=500, detail=f"El agente analista falló: {e}")

        analyst_answer_raw = analyst_response.get("output", "No se pudo generar una respuesta.")
        _, table_data = parse_response_to_df(analyst_answer_raw)
        
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        clean_log = ansi_escape.sub('', log_output)

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
    
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise ValueError("FATAL: La variable de entorno GOOGLE_API_KEY no se encontró. Asegúrate de que el secreto esté montado.")
    
    # No es necesario hacer os.environ["GOOGLE_API_KEY"] = google_api_key,
    # ya que las librerías de Google leen la variable de entorno directamente.

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
    Eres un experto en SQL y analista de datos. Tu tarea es validar si la consulta SQL generada por un agente de IA responde correctamente a la pregunta original del usuario.
    Regla de negocio importante: Para calcular el total de ventas o el gasto, la consulta SQL DEBE multiplicar 'Quantity' por 'UnitPrice'.
    Pregunta del Usuario: "{question}"
    Consulta SQL generada: "{sql_query}"
    Respuesta generada por el agente: "{agent_response}"
    Evalúa lo siguiente:
    1.  ¿La consulta SQL responde directamente a la pregunta del usuario?
    2.  ¿Cumple con la regla de negocio sobre 'Quantity' * 'UnitPrice' si la pregunta implica un total de ventas o gasto?
    3.  ¿La respuesta final es coherente con la consulta y la pregunta?
    Proporciona un veredicto final en una sola línea: "APROBADO" si todo es correcto, o "RECHAZADO" con una breve explicación si hay un error.
    Veredicto:
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
# ... (Pega aquí tus rutas @app.get y @app.post) ...
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



