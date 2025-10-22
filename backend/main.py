# ~/agente_sql/backend/main.py (Versión Final Definitiva y Robusta)

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

def parse_response_to_df(response_text: str):
    table_regex = re.compile(r"(\|.*\|(?:\n\|.*\|)+)")
    table_match = table_regex.search(response_text)
    
    if not table_match:
        return response_text, []

    table_str = table_match.group(0)
    text_part = ""

    try:
        lines = table_str.strip().split("\n")
        if len(lines) > 1 and all(c in '|-: ' for c in lines[1]):
            del lines[1]
        
        csv_like = "\n".join([line.strip().strip('|').replace('|', ',') for line in lines])
        df = pd.read_csv(io.StringIO(csv_like), skipinitialspace=True)
        df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
        return text_part, df.to_dict(orient='records')
    except Exception as e:
        print(f"Error al parsear tabla markdown, devolviendo texto original. Error: {e}")
        return response_text, []

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
        text_part, table_data = parse_response_to_df(analyst_answer_raw)
        
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-9;?]*[ -/]*[@-~])')
        clean_log = ansi_escape.sub('', log_io.getvalue())

        sql_matches = re.findall(r"Action Input: (SELECT .*?)(?:\n|$)", clean_log, re.DOTALL)
        sql_query = sql_matches[-1].strip() if sql_matches else "No se ejecutó una consulta SQL directa."

        verdict = self.validator_chain.invoke({ "question": question, "sql_query": sql_query })
        full_log = f"{clean_log}\n--- Interacción con el Validador ---\nPregunta: {question}\nConsulta: {sql_query}\nVeredicto: {verdict}"
        
        return {
            "answer_text": text_part,
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
    
    # --- PROMPT DE SISTEMA (PREFIX) REFORZADO Y DEFINITIVO ---
    prefix = """
    Eres un bot de extracción de datos SQL altamente especializado. Tu ÚNICO propósito es recibir una pregunta en lenguaje natural y devolver el resultado de la consulta SQL correspondiente como una ÚNICA tabla en formato Markdown.

    SIGUE ESTAS REGLAS INQUEBRANTABLES:

    REGLA #1: FORMATO DE SALIDA
    Tu respuesta final DEBE ser solo la tabla Markdown. NO incluyas NINGÚN texto introductorio, explicaciones, saludos o resúmenes. La tabla es tu única salida permitida.

    REGLA #2: IDIOMA
    La tabla resultante y, de manera CRÍTICA, todos sus encabezados de columna, DEBEN estar completamente en español. Traduce los nombres de las columnas de la base de datos a un español claro y legible. Ejemplo: 'Country' se convierte en 'País', 'SUM(Quantity * UnitPrice)' se convierte en 'Venta_Total'.

    REGLA #3: CÁLCULOS
    Para cualquier cálculo de "ventas", "gasto", "ingresos" o "total", SIEMPRE multiplica 'Quantity' por 'UnitPrice'.

    A continuación se muestran ejemplos del comportamiento correcto:

    <ejemplo>
    Pregunta: "Top 3 países por ventas"
    Respuesta:
    | País          | Venta_Total |
    |---------------|-------------|
    | Reino Unido   | 8187806.24  |
    | Países Bajos  | 284661.54   |
    | EIRE          | 263276.82   |
    </ejemplo>

    <ejemplo>
    Pregunta: "dame los datos del cliente con id 12350"
    Respuesta:
    | InvoiceNo | StockCode | Description                          | Quantity | InvoiceDate               | UnitPrice | CustomerID | Country       |
    |-----------|-----------|--------------------------------------|----------|---------------------------|-----------|------------|---------------|
    | 536378    | 21578     | ROUND SNACK BOXES SET OF 4 FRUITS    | 6        | 2010-12-01 09:37:00       | 2.95      | 12350      | Channel Islands |
    | 536378    | 21992     | VINTAGE PAISLEY STATIONERY SET       | 12       | 2010-12-01 09:37:00       | 2.55      | 12350      | Channel Islands |
    </ejemplo>

    Ahora, procesa la siguiente pregunta del usuario. Recuerda, tu única respuesta debe ser la tabla en Markdown y en español.
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
    print("Apagando el servidor.")
    app_state.clear()

app = FastAPI(lifespan=lifespan)

@app.get("/")
def read_root():
    return {"status": "El backend del Agente de Datos está funcionando."}

@app.post("/query", response_model=QueryResponse)
async def handle_query(request: QueryRequest):
    # Se elimina la modificación del input. Se envía la pregunta original directamente.
    print(f"\n--- [NUEVA PETICIÓN]: {request.question} ---")
    query_master = app_state.get('query_master')
    if not query_master:
        raise HTTPException(status_code=503, detail="El servicio no está listo.")
    if not request.question:
        raise HTTPException(status_code=400, detail="La pregunta no puede estar vacía.")
    
    try:
        # Enviamos la pregunta original y sin modificar al agente.
        result = query_master.run_query(request.question)
        return QueryResponse(**result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Ocurrió un error interno en el backend: {e}")

