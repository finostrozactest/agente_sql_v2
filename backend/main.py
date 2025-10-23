# ~/agente_sql/backend/main.py (Versión Final con Carga de Excel Local)

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

# --- CAMBIO PRINCIPAL (3/3): Cargar datos desde un archivo Excel local ---
def load_and_prepare_data():
    local_file = "transaccional_dummy.xlsx"
    try:
        print(f"Cargando datos desde el archivo local: {local_file}")
        if not os.path.exists(local_file):
            error_msg = f"Error CRÍTICO: No se encontró el archivo '{local_file}' en el directorio del backend."
            print(error_msg)
            raise FileNotFoundError(error_msg)
            
        df = pd.read_excel(local_file)
        print("Datos cargados. Iniciando limpieza si es necesario...")
        # Asumiendo que tu Excel ya tiene las columnas necesarias con los tipos de datos correctos.
        # Si las columnas se llaman diferente, ajústalas aquí.
        # Ejemplo de limpieza (ajústalo a tu archivo real):
        if 'InvoiceDate' in df.columns:
            df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        if 'CustomerID' in df.columns:
            df.dropna(subset=['CustomerID'], inplace=True)
            df['CustomerID'] = df['CustomerID'].astype(int)
        
        print("Datos cargados y preparados exitosamente.")
        return df
    except FileNotFoundError as e:
        # Este error es específico para que sepas que falta el archivo
        raise RuntimeError(e)
    except Exception as e:
        print(f"Error CRÍTICO durante la carga de datos del archivo local: {e}")
        return None

def create_db_engine(df):
    try:
        engine = create_engine("sqlite:///:memory:")
        # El nombre de la tabla ahora será el nombre del archivo sin extensión
        table_name = "transaccional_dummy"
        df.to_sql(table_name, engine, index=False, if_exists="replace")
        print(f"Base de datos en memoria creada y poblada con la tabla '{table_name}'.")
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
        final_text = "" if table_data else text_part
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-9;?]*[ -/]*[@-~])')
        clean_log = ansi_escape.sub('', log_io.getvalue())
        sql_matches = re.findall(r"Action Input: (SELECT .*?)(?:\n|$)", clean_log, re.DOTALL)
        sql_query = sql_matches[-1].strip() if sql_matches else "No se ejecutó una consulta SQL directa."
        verdict = self.validator_chain.invoke({ "question": question, "sql_query": sql_query })
        full_log = f"{clean_log}\n--- Interacción con el Validador ---\nPregunta: {question}\nConsulta: {sql_query}\nVeredicto: {verdict}"
        return { "answer_text": final_text, "table_data": table_data, "reasoning": full_log, "verdict": verdict }

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
    REGLAS DE NEGOCIO OBLIGATORIAS:
    1. CÁLCULO DE VENTAS: Para calcular el total de ventas o el gasto, SIEMPRE debes multiplicar 'Quantity' por 'UnitPrice'.
    2. FILTROS DE TEXTO: Cuando filtres por un valor de texto (ej. un país), SIEMPRE debes usar comillas simples. Ejemplo: `WHERE Country = 'France'`.
    """
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    analyst_agent_executor = create_sql_agent( llm=llm, toolkit=toolkit, verbose=True, prefix=prefix, handle_parsing_errors="Tuve un problema para interpretar la consulta. Por favor, reformula tu pregunta en español.", agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION)
    validator_template = """
    Tu única tarea es actuar como un validador de consultas SQL. Basado en la pregunta del usuario y la consulta SQL generada, proporciona un veredicto en UNA SOLA LÍNEA y en español. Obliga al analista a responder con formato tabla Markdown y en español.
    Regla de negocio: Si la pregunta implica calcular el margen, la consulta DEBE dividir 'contribucion' por 'venta'.
    Pregunta: "{question}"
    Consulta: "{sql_query}"
    Evalúa si la consulta responde correctamente a la pregunta y cumple la regla de negocio.
    Si te preguntan por CLACOM, se refiere aluniverso de categorias, que son las columnas departamento, familia, subfamilia, grupo, conjunto
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
    query_master = app_state.get('query_master')
    if not query_master:
        raise HTTPException(status_code=503, detail="El servicio no está listo.")
    if not request.question:
        raise HTTPException(status_code=400, detail="La pregunta no puede estar vacía.")
    fixed_instruction = "Genera una tabla de datos como resultado. Responde completamente en español. La pregunta es:"
    modified_question = f"{fixed_instruction} {request.question}"
    print(f"\n--- [INPUT ORIGINAL]: {request.question} ---")
    print(f"--- [INPUT MODIFICADO PARA EL AGENTE]: {modified_question} ---")
    try:
        result = query_master.run_query(modified_question)
        return QueryResponse(**result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Ocurrió un error interno en el backend: {e}")


