# API
from __future__ import annotations
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from app.inferencia import (
    ETIQUETAS, ImagenInvalida, LIMITE_BYTES, NOMBRE_MODELO, RUTA_MODELO,
    UMBRAL_MARGEN_AMBIGUO,
    cargar_modelo, predecir_imagen,
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    cargar_modelo()
    yield

app = FastAPI(
    title='Vision 360 | Weld AI - API', version='1.2.0',
    description='Clasificación educativa de radiografías públicas RIAWELC. No autoriza decisiones industriales.',
    docs_url='/api/docs', redoc_url=None,
    openapi_url='/api/openapi.json', lifespan=lifespan,
)

@app.get('/api/health', tags=['Sistema'])
def health():
    cargar_modelo()
    return {'estado': 'ok', 'modelo_disponible': True, 'version': '1.2.0'}

@app.get('/api/model', tags=['Sistema'])
def info_modelo():
    modelo = cargar_modelo()
    return {
        'nombre': NOMBRE_MODELO,
        'dataset': 'RIAWELC',
        'clases': [{'id': i, 'nombre': v} for i, v in ETIQUETAS.items()],
        'caracteristicas': 324,
        'tamano_entrada': [64, 64],
        'modelo_cargado': RUTA_MODELO.name,
        'clases_internas': [int(c) for c in modelo.classes_],
        'criterio_ambiguedad': {
            'metrica': 'margen entre las dos probabilidades mayores',
            'operador': '<',
            'umbral': UMBRAL_MARGEN_AMBIGUO,
            'efecto': 'Informa RESULTADO NO CONCLUYENTE sin cambiar la clase predicha.',
        },
        'uso': 'Académico. Radiografías públicas, no imagen óptica Vision 360.',
    }

@app.post('/api/predict', tags=['Predicción'])
async def predict(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=422, detail='Debes seleccionar un archivo de imagen.')
    if file.content_type and not (file.content_type.startswith('image/') or file.content_type == 'application/octet-stream'):
        raise HTTPException(status_code=415, detail='Solo se aceptan archivos de imagen.')
    try:
        contenido = await file.read(LIMITE_BYTES + 1)
        if len(contenido) > LIMITE_BYTES:
            raise HTTPException(status_code=413, detail='La imagen supera el límite de 8 MB.')
        return predecir_imagen(contenido)
    except ImagenInvalida as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        await file.close()

@app.exception_handler(FileNotFoundError)
async def sin_modelo(request, exc):
    return JSONResponse(status_code=503, content={'detail': str(exc)})
