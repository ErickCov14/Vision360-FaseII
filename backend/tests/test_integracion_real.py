# VALIDACION
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageOps
from skimage.feature import hog

from app import inferencia
from app.main import app

ROOT = Path(__file__).resolve().parents[2]
EJEMPLOS = {
    '01_grieta.png': 0,
    '02_porosidad.png': 1,
    '03_falta_penetracion.png': 2,
    '04_sin_defecto.png': 3,
}


def referencia_actividad5(contenido: bytes) -> np.ndarray:
    with Image.open(BytesIO(contenido)) as imagen:
        imagen = ImageOps.exif_transpose(imagen)
        imagen = imagen.convert('L')
        imagen = imagen.resize((64, 64), Image.Resampling.LANCZOS)
        matriz = np.asarray(imagen, dtype=np.uint8)
    return hog(
        matriz / 255.0,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
    ).reshape(1, -1)


class ModeloCapturador:
    def __init__(self, modelo):
        self.modelo = modelo
        self.classes_ = modelo.classes_
        self.ultimo_x = None

    def predict(self, x):
        self.ultimo_x = np.array(x, copy=True)
        return self.modelo.predict(x)

    def predict_proba(self, x):
        self.ultimo_x = np.array(x, copy=True)
        return self.modelo.predict_proba(x)


@pytest.mark.skipif(not inferencia.RUTA_MODELO.is_file(), reason='Modelo real no disponible')
@pytest.mark.parametrize(('archivo', 'clase_real'), EJEMPLOS.items())
def test_directo_y_api_coinciden(monkeypatch, archivo, clase_real):
    ruta = ROOT / 'ejemplos' / archivo
    contenido = ruta.read_bytes()
    modelo_real = inferencia.cargar_modelo()
    capturador = ModeloCapturador(modelo_real)
    monkeypatch.setattr(inferencia, 'cargar_modelo', lambda: capturador)
    monkeypatch.setattr('app.main.cargar_modelo', lambda: capturador)

    x_referencia = referencia_actividad5(contenido)
    pred_directa = int(modelo_real.predict(x_referencia)[0])
    proba_directa = modelo_real.predict_proba(x_referencia)[0]
    clases = [int(c) for c in modelo_real.classes_]

    with TestClient(app) as client:
        respuesta = client.post('/api/predict', files={'file': (archivo, contenido, 'image/png')})
    assert respuesta.status_code == 200
    datos = respuesta.json()

    assert clase_real in (0, 1, 2, 3)
    assert capturador.ultimo_x is not None
    np.testing.assert_allclose(capturador.ultimo_x, x_referencia, rtol=0, atol=1e-12)
    assert datos['clase_id'] == pred_directa
    assert datos['clase'] == inferencia.ETIQUETAS[pred_directa]
    proba_api = {p['id']: p['probabilidad'] for p in datos['probabilidades']}
    for clase, valor in zip(clases, proba_directa):
        assert proba_api[clase] == pytest.approx(float(valor), abs=5e-7)
    assert datos['clase_id'] == clases[int(np.argmax(proba_directa))]


@pytest.mark.skipif(not inferencia.RUTA_MODELO.is_file(), reason='Modelo real no disponible')
def test_se_ejecuta_una_imagen_real_de_cada_categoria():
    salidas = []
    for archivo, clase_real in EJEMPLOS.items():
        salida = inferencia.predecir_imagen((ROOT / 'ejemplos' / archivo).read_bytes())
        assert salida['clase_id'] in inferencia.ETIQUETAS
        assert len(salida['probabilidades']) == 4
        salidas.append({'clase_real': clase_real, 'predicha': salida['clase_id']})
    assert len(salidas) == 4
