from io import BytesIO
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from app import inferencia
from app.main import app

# VALIDACION

class ModeloFicticio:
    classes_ = np.array([0, 1, 2, 3])
    def predict_proba(self, X):
        assert X.shape == (1, 324)
        return np.array([[0.1, 0.2, 0.3, 0.4]])
    def predict(self, X):
        assert X.shape == (1, 324)
        return np.array([3])

@pytest.fixture
def client(monkeypatch):
    inferencia.cargar_modelo.cache_clear()
    monkeypatch.setattr(inferencia, 'cargar_modelo', lambda: ModeloFicticio())
    monkeypatch.setattr('app.main.cargar_modelo', lambda: ModeloFicticio())
    with TestClient(app) as c:
        yield c

def png_prueba():
    data = np.tile(np.arange(128, dtype=np.uint8), (128, 1))
    f = BytesIO(); Image.fromarray(data).save(f, format='PNG'); return f.getvalue()

def test_health(client):
    r = client.get('/api/health')
    assert r.status_code == 200 and r.json()['modelo_disponible'] is True

def test_metadata(client):
    r = client.get('/api/model')
    assert r.status_code == 200 and len(r.json()['clases']) == 4
    assert r.json()['caracteristicas'] == 324

def test_upload_ok(client):
    r = client.post('/api/predict', files={'file': ('foto.png', png_prueba(), 'image/png')})
    assert r.status_code == 200
    data = r.json()
    assert data['clase_id'] == 3
    assert len(data['probabilidades']) == 4
    assert abs(sum(p['probabilidad'] for p in data['probabilidades']) - 1) < 1e-5
    assert data['estado_analisis'] == 'ANÁLISIS COMPLETADO'
    assert data['ambigua'] is False
    assert data['margen_probabilidades'] == pytest.approx(0.1)

def test_corrupta(client):
    r = client.post('/api/predict', files={'file': ('foto.png', b'no-es-una-imagen', 'image/png')})
    assert r.status_code == 422

def test_vacia(client):
    r = client.post('/api/predict', files={'file': ('foto.png', b'', 'image/png')})
    assert r.status_code == 422

def test_mime_incorrecto(client):
    r = client.post('/api/predict', files={'file': ('x.txt', b'abc', 'text/plain')})
    assert r.status_code == 415

def test_demasiado_grande(client):
    r = client.post('/api/predict', files={'file': ('foto.png', b'x' * (inferencia.LIMITE_BYTES + 1), 'image/png')})
    assert r.status_code == 413

def test_sin_archivo(client):
    assert client.post('/api/predict').status_code == 422

def test_documentacion(client):
    assert client.get('/api/openapi.json').status_code == 200

def test_criterio_ambiguedad_documentado(client):
    criterio = client.get('/api/model').json()['criterio_ambiguedad']
    assert criterio['umbral'] == inferencia.UMBRAL_MARGEN_AMBIGUO
    assert 'sin cambiar la clase' in criterio['efecto']
