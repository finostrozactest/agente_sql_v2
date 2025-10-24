# Ruta: frontend/Dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# No necesitamos EXPOSE ni CMD, el entrypoint del app.yaml se encargará.
```2.  **Reemplaza COMPLETAMENTE el contenido de `frontend/app.yaml`** con esta configuración.
```yaml
# Ruta: frontend/app.yaml
# --- CONFIGURACIÓN PARA APP ENGINE STANDARD CON DOCKER ---
service: default
runtime: custom # Obligatorio para usar Docker

env_variables:
  BACKEND_URL: "https://backend-dot-project-agentes.ue.r.appspot.com/query"

# El entrypoint que se ejecutará dentro del contenedor
entrypoint: streamlit run app.py --server.port 8080 --server.enableCORS false --server.enableXsrfProtection false

# La sintaxis de escalado para Standard que SÍ permite cero instancias
automatic_scaling:
  min_instances: 0
  max_instances: 2

instance_class: F2
