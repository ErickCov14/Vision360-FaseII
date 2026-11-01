# VALIDACION
from io import BytesIO
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
from app import inferencia

@pytest.mark.skipif(not inferencia.RUTA_MODELO.is_file(), reason='Modelo no disponible: ejecutar scripts/obtener_modelo.py')
def test_inferencia_real():
    imagen = np.tile(np.arange(128, dtype=np.uint8), (128, 1))
    datos = BytesIO(); Image.fromarray(imagen).save(datos, format='PNG')
    salida = inferencia.predecir_imagen(datos.getvalue())
    assert salida['clase_id'] in (0, 1, 2, 3)
    assert len(salida['probabilidades']) == 4
    assert np.isclose(sum(v['probabilidad'] for v in salida['probabilidades']), 1, atol=1e-5)

@pytest.mark.skipif(not inferencia.RUTA_MODELO.is_file(), reason='Modelo no disponible: ejecutar scripts/obtener_modelo.py')
def test_metadata_modelo_real():
    modelo = inferencia.cargar_modelo()
    estimador = modelo.steps[-1][1] if hasattr(modelo, 'steps') else modelo
    assert estimador.__class__.__name__ == 'RandomForestClassifier'
    assert modelo.n_features_in_ == 324
    assert [int(c) for c in modelo.classes_] == [0, 1, 2, 3]
    assert estimador.max_depth == 20

def test_modelo_original_se_conserva():
    ruta = inferencia.RUTA_MODELO.parent / 'random_forest_vision360.joblib'
    assert ruta.is_file()
    assert ruta.stat().st_size == 10_132_306
