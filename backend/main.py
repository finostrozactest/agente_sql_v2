import os
import pandas as pd
from sqlalchemy import create_engine, text, inspect
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import functools

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_sql_agent
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase

# --- Modelos de Datos Pydantic para la API ---
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    reasoning: str
    answer: str
    table_data: list[dict]

# --- Lógica de la Aplicación (adaptada de Colab) ---

# Usamos caché para asegurar que los datos se descarguen y procesen solo una vez.
@functools.lru_cache(maxsize=None)
def load_and_prepare_data():
    """Carga, limpia y prepara los datos para la base de datos."""
    print("Iniciando la carga y preparación de datos...")
    ecommerce_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
    try:
        df = pd.read_excel(ecommerce_url)
        print("Datos cargados exitosamente desde la URL.")
        
        # Limpieza de datos
        df.dropna(subset=['CustomerID'], inplace=True)
        df = df[df['Quantity'] > 0]
        df['CustomerID'] = df['CustomerID'].astype(int)
        df.rename(
            columns={
                'InvoiceNo': 'InvoiceID', 'StockCode': 'StockCode', 'Description': 'Description',
                'Quantity': 'Quantity', 'InvoiceDate': 'InvoiceDate', 'UnitPrice': 'UnitPrice',
                'CustomerID': 'CustomerID', 'Country': 'Country'
            },
            inplace=True
        )
        print("Limpieza de datos completada.")
        return df
    except Exception as e:
        print(f"Error crítico al cargar o limpiar los datos: {e}")
        return None

def create_db_engine(df):
    """Crea una base de datos SQLite en memoria y carga el DataFrame."""
    if df is None:
        return None
    try:
        engine = create_engine("sqlite:///:memory:")
        df.to_sql("transacciones", engine, index=False, if_exists="replace")
        print("Base de datos en memoria creada y poblada.")
        return engine
    except Exception as e:
        print(f"Error al crear la base de datos en memoria: {e}")
        return None

# Variable global para almacenar los recursos (agentes, db, etc.)
app_state = {}

class QueryMaster:
    """Orquestador para los agentes de IA."""
    def __init__(self, analyst_agent, validator_chain, db_engine):
        self.analyst_agent = analyst_agent
        self.validator_chain = validator_chain
        self.db_engine = db_engine

    def _get_sql_from_thought_process(self, thought_process):
        """Extrae la última consulta SQL del razonamiento del agente."""
        try:
            # Busca la última consulta SQL en el razonamiento
            sql_query_marker = "SQLQuery:"
            last_occurrence = thought_process.rfind(sql_query_marker)
            if last_occurrence == -1:
                return None
            
            thought_after_marker = thought_process[last_occurrence + len(sql_query_marker):]
            # La consulta termina en el siguiente "SQLResult:" o al final del string
            sql_query = thought_after_marker.split("SQLResult:")[0].strip()
            return sql_query
        except Exception:
            return None

    def run_query(self, question: str):
        print("--- [Paso 1: Iniciando Agente Analista] ---")
        try:
            analyst_response = self.analyst_agent.invoke({"input": question})
        except Exception as e:
            print(f"Error en el agente analista: {e}")
            raise HTTPException(status_code=500, detail=f"El agente analista falló: {e}")

        analyst_thought_process = analyst_response.get("intermediate_steps", "No disponible")
        analyst_answer = analyst_response["output"]

        print("\n--- [Paso 2: Iniciando Agente Validador] ---")
        validation_result = self.validator_chain.invoke({
            "user_question": question,
            "analyst_thought_process": analyst_thought_process,
            "analyst_answer": analyst_answer
        })

        print("\n--- [Veredicto del Validador] ---\n")
        print(validation_result)

        # Extraer y ejecutar la consulta SQL para obtener datos tabulares
        table_data = []
        sql_query = self._get_sql_from_thought_process(str(analyst_thought_process))
        if sql_query:
            print(f"Ejecutando SQL extraída para obtener datos de tabla: {sql_query}")
            try:
                with self.db_engine.connect() as connection:
                    result_df = pd.read_sql_query(text(sql_query), connection)
                    table_data = result_df.to_dict(orient='records')
            except Exception as e:
                print(f"No se pudo ejecutar la SQL para obtener datos tabulares. Error: {e}")
        else:
             print("No se encontró una consulta SQL en el razonamiento del agente.")


        return {
            "reasoning": str(analyst_thought_process),
            "answer": analyst_answer,
            "table_data": table_data
        }

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Lógica que se ejecuta al iniciar la aplicación ---
    print("Iniciando el servidor FastAPI...")
    # 1. Configurar la API Key de Google
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise ValueError("La variable de entorno GOOGLE_API_KEY no está configurada.")
    os.environ["GOOGLE_API_KEY"] = google_api_key

    # 2. Cargar y preparar datos
    ecommerce_data = load_and_prepare_data()
    if ecommerce_data is None:
        raise RuntimeError("No se pudieron cargar los datos, el backend no puede iniciar.")
    
    # 3. Crear la base de datos
    engine = create_db_engine(ecommerce_data)
    if engine is None:
         raise RuntimeError("No se pudo crear la base de datos, el backend no puede iniciar.")

    # 4. Configurar Agentes de IA
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    db = SQLDatabase(engine=engine)
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    
    custom_error_message = """
    ERROR DE ANÁLISIS: El formato de tu respuesta anterior no fue correcto.
    Recuerda usar el formato requerido. La última acción SIEMPRE debe ser:
    1. Una herramienta con 'Action:' y 'Action Input:'.
    2. La respuesta final, que DEBE empezar con 'Final Answer:'.
    Inténtalo de nuevo.
    """
    
    analyst_agent_executor = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=True,
        agent_type="zero-shot-react-description",
        handle_parsing_errors=custom_error_message
    )

    validator_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template("Eres un experto analista de datos y tu rol es validar la respuesta de otro agente de IA. Revisa la Pregunta Original, la Consulta SQL y la Respuesta Generada. Si todo es correcto, responde con 'APROBADO'. Si encuentras algún error, explica detalladamente cuál es el problema."),
        HumanMessagePromptTemplate.from_template("Pregunta Original: {user_question}\n\nProceso y SQL: {analyst_thought_process}\n\nRespuesta Generada: {analyst_answer}\n\nTu Veredicto:"),
    ])

    validator_chain = validator_prompt | llm | StrOutputParser()
    
    # 5. Instanciar el Orquestador
    app_state['query_master'] = QueryMaster(analyst_agent_executor, validator_chain, engine)
    print("¡El servidor está listo para recibir peticiones!")
    
    yield
    
    # --- Lógica que se ejecuta al apagar la aplicación (limpieza) ---
    print("Apagando el servidor FastAPI.")
    app_state.clear()


app = FastAPI(lifespan=lifespan)

@app.get("/")
def read_root():
    return {"status": "El backend del Agente de Datos está funcionando."}

@app.post("/query", response_model=QueryResponse)
async def handle_query(request: QueryRequest):
    query_master = app_state.get('query_master')
    if not query_master:
        raise HTTPException(status_code=503, detail="El servicio no está listo. Inténtalo de nuevo en unos momentos.")
    
    if not request.question:
        raise HTTPException(status_code=400, detail="La pregunta no puede estar vacía.")

    try:
        result = query_master.run_query(request.question)
        return QueryResponse(**result)
    except HTTPException as http_exc:
        raise http_exc # Re-lanzar excepciones HTTP
    except Exception as e:
        print(f"Error inesperado al procesar la consulta: {e}")
        raise HTTPException(status_code=500, detail=f"Ocurrió un error interno: {e}")
