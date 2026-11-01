FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app
COPY backend/requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements.txt
COPY backend/app /app/app
COPY scripts/obtener_modelo.py /app/scripts/obtener_modelo.py
# Se descarga del repositorio público del propio proyecto y se verifica el SHA de Git.
RUN python /app/scripts/obtener_modelo.py --dest /app/model/random_forest_vision360.joblib
# El modelo original se conserva; el candidato CPU validado se incluye con nombre independiente.
COPY backend/model/random_forest_vision360_cpu_optimizado.joblib /app/model/random_forest_vision360_cpu_optimizado.joblib
ENV MODEL_PATH=/app/model/random_forest_vision360_cpu_optimizado.joblib \
    MODEL_NAME="Random Forest CPU optimizado + HOG64"
RUN useradd -r -u 10001 vision && chown -R vision:vision /app
USER vision
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=25s --retries=3 CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/api/health',timeout=5).read()"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
