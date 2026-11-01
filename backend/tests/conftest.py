import os
import sys
from pathlib import Path

CARPETA_MODELOS = Path(__file__).resolve().parents[1] / 'model'
MODELO_LOCAL = CARPETA_MODELOS / 'random_forest_vision360_cpu_optimizado.joblib'
if not MODELO_LOCAL.is_file():
    MODELO_LOCAL = CARPETA_MODELOS / 'random_forest_vision360.joblib'
if MODELO_LOCAL.is_file():
    os.environ.setdefault('MODEL_PATH', str(MODELO_LOCAL))
    if 'cpu_optimizado' in MODELO_LOCAL.name:
        os.environ.setdefault('MODEL_NAME', 'Random Forest CPU optimizado + HOG64')

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
