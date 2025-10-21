
import re
import io
from contextlib import redirect_stdout
from sqlalchemy import create_engine, text
from sqlalchemy import create_engine
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
@@ -20,9 +20,9 @@
class QueryRequest(BaseModel):
    question: str

# El backend ahora devuelve una estructura más simple
class QueryResponse(BaseModel):
    answer_text: str
    table_data: list[dict]
    answer_text: str # Contendrá texto y la tabla markdown juntos
    reasoning: str
    verdict: str

@@ -56,35 +56,6 @@ def create_db_engine(df):
        print(f"Error al crear la base de datos en memoria: {e}")
        return None

def parse_response_to_df(response_text: str):
    """Extrae texto y una tabla Markdown de la respuesta del agente y la convierte en DataFrame."""
    table_regex = re.compile(r"(\|.*\|(?:\n\|.*\|)+)")
    table_match = table_regex.search(response_text)
    if not table_match:
        return response_text, []
    
    table_str = table_match.group(0)
    text_part = response_text.replace(table_str, "").strip()
    
    try:
        lines = table_str.strip().split("\n")
        # Eliminar la línea de separación de Markdown si existe
        if len(lines) > 1 and all(c in '|-: ' for c in lines[1]):
            del lines[1]
        
        # Convertir a formato CSV-like
        csv_like = "\n".join([line.strip().strip('|').replace('|', ',') for line in lines])
        df = pd.read_csv(io.StringIO(csv_like))
        
        # Limpiar espacios en blanco en las columnas
        df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
        
        final_text = text_part if text_part else "Aquí están los resultados de tu consulta:"
        return final_text, df.to_dict(orient='records')
    except Exception as e:
        print(f"Error al parsear la tabla markdown: {e}")
        return response_text, []

app_state = {}

class QueryMaster:
@@ -97,25 +68,20 @@ def run_query(self, question: str):

        log_io = io.StringIO()
        try:
            # Captura la salida verbosa del agente
            with redirect_stdout(log_io):
                analyst_response = self.analyst_agent.invoke({"input": question})
            log_output = log_io.getvalue()
        except Exception as e:
            print(f"Error en el agente analista: {e}")
            raise HTTPException(status_code=500, detail=f"El agente analista falló: {e}")
            raise

        # La respuesta del agente ya contiene el texto y la tabla markdown juntos
        analyst_answer_raw = analyst_response.get("output", "No se pudo generar una respuesta.")

        print("\n--- [Paso 2: Parseando la respuesta del agente] ---")
        answer_text, table_data = parse_response_to_df(analyst_answer_raw)
        
        print("\n--- [Paso 3: Iniciando Agente Validador] ---")
        # Limpiar el log de colores ANSI para la validación
        print("\n--- [Paso 2: Iniciando Agente Validador] ---")
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        clean_log = ansi_escape.sub('', log_output)

        # Extraer la última consulta SQL del log
        sql_matches = re.findall(r"Action Input: (SELECT .*?)(?:\n|$)", clean_log, re.DOTALL)
        sql_query = sql_matches[-1].strip() if sql_matches else "No se ejecutó una consulta SQL directa."

@@ -126,10 +92,10 @@ def run_query(self, question: str):
        })
        print(f"\n--- [Veredicto del Validador] ---\n{verdict}")

        # Devolvemos la respuesta cruda del agente, que es lo que el frontend espera
        return {
            "reasoning": clean_log,
            "answer_text": answer_text,
            "table_data": table_data,
            "answer_text": analyst_answer_raw,
            "verdict": verdict
        }

@@ -141,17 +107,16 @@ async def lifespan(app: FastAPI):
    os.environ["GOOGLE_API_KEY"] = google_api_key

    ecommerce_data = load_and_prepare_data()
    if ecommerce_data is None: raise RuntimeError("No se pudieron cargar los datos, el backend no puede iniciar.")
    if ecommerce_data is None: raise RuntimeError("No se pudieron cargar los datos.")

    engine = create_db_engine(ecommerce_data)
    if engine is None: raise RuntimeError("No se pudo crear la base de datos, el backend no puede iniciar.")
    if engine is None: raise RuntimeError("No se pudo crear la base de datos.")

    llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash-lite", temperature=0)
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0)
    db = SQLDatabase(engine=engine)

prefix = """
Eres un asistente experto en análisis de datos que trabaja con una base de datos SQLite.
La base de datos contiene una única tabla llamada 'transacciones' con información de ventas de e-commerce.
Eres un analista de datos experto que trabaja con una base de datos SQLite.
Tu objetivo es responder a las preguntas del usuario generando y ejecutando consultas SQL.
Regla de negocio importante: Para calcular el total de ventas o el gasto, siempre debes multiplicar la columna 'Quantity' por 'UnitPrice'.
Cuando la respuesta sea una tabla de datos, preséntala en formato Markdown.
"""



@@ -164,7 +129,7 @@ async def lifespan(app: FastAPI):
        toolkit=toolkit,
        verbose=True,
        prefix=prefix,
        handle_parsing_errors="Tuve un problema para interpretar la consulta. Por favor, intenta reformular tu pregunta de una manera más simple.",
        handle_parsing_errors="Tuve un problema para interpretar la consulta. Por favor, reformula tu pregunta.",
        max_iterations=10
    )

@@ -175,32 +140,25 @@ async def lifespan(app: FastAPI):
    validator_chain = validator_prompt | llm | StrOutputParser()

    app_state['query_master'] = QueryMaster(analyst_agent_executor, validator_chain)
    app_state['full_data'] = ecommerce_data # Guardamos los datos completos para descarga
    print("¡El servidor está listo para recibir peticiones!")
    print("¡El servidor está listo!")
    yield
    print("Apagando el servidor FastAPI.")
    print("Apagando el servidor.")
    app_state.clear()


app = FastAPI(lifespan=lifespan)

@app.get("/")
def read_root():
    return {"status": "El backend del Agente de Datos está funcionando."}

@app.post("/query", response_model=QueryResponse)
async def handle_query(request: QueryRequest):
    print(f"\n--- [NUEVA PETICIÓN RECIBIDA]: {request.question} ---")
    print(f"\n--- [NUEVA PETICIÓN]: {request.question} ---")
    query_master = app_state.get('query_master')
    if not query_master:
        raise HTTPException(status_code=503, detail="El servicio no está listo.")
    if not request.question:
        raise HTTPException(status_code=400, detail="La pregunta no puede estar vacía.")
    try:
        result = query_master.run_query(request.question)
        print("QueryMaster ha finalizado. Devolviendo respuesta.")
        return QueryResponse(**result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Ocurrió un error interno: {e}")
        raise HTTPException(status_code=500, detail=f"Ocurrió un error interno en el backend: {e}")


    
