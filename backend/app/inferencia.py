# PROCESAMIENTO
from __future__ import annotations
from functools import lru_cache
from io import BytesIO
import os
from pathlib import Path
import joblib
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from skimage.feature import hog

RUTA_MODELO = Path(os.getenv('MODEL_PATH', '/app/model/random_forest_vision360.joblib'))
NOMBRE_MODELO = os.getenv('MODEL_NAME', 'Random Forest + HOG (Actividad 5)')
ETIQUETAS = {0: 'Grieta', 1: 'Porosidad', 2: 'Falta de penetración', 3: 'Sin defecto'}
FORMATOS = {'PNG', 'JPEG', 'BMP', 'TIFF'}
LIMITE_BYTES = 8 * 1024 * 1024
LIMITE_PIXELES = 18_000_000
UMBRAL_MARGEN_AMBIGUO = 0.10
Image.MAX_IMAGE_PIXELS = LIMITE_PIXELES

class ImagenInvalida(ValueError):
    pass

def preparar_imagen(contenido: bytes) -> np.ndarray:
    if not contenido:
        raise ImagenInvalida('El archivo está vacío.')
    if len(contenido) > LIMITE_BYTES:
        raise ImagenInvalida('La imagen supera el límite de 8 MB.')
    try:
        with Image.open(BytesIO(contenido)) as imagen:
            if imagen.format not in FORMATOS:
                raise ImagenInvalida('Formato no admitido. Utiliza PNG, JPG, BMP o TIFF.')
            if imagen.width * imagen.height > LIMITE_PIXELES:
                raise ImagenInvalida('La imagen supera el límite de 18 megapíxeles.')
            imagen = ImageOps.exif_transpose(imagen)
            imagen = imagen.convert('L')
            imagen = imagen.resize((64, 64), Image.Resampling.LANCZOS)
            matriz = np.asarray(imagen, dtype=np.uint8)
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        if isinstance(exc, ImagenInvalida):
            raise
        raise ImagenInvalida('El archivo no es una imagen válida o está dañado.') from exc
    return matriz

def extraer_caracteristicas(imagen: np.ndarray) -> np.ndarray:
    vector = hog(
        imagen / 255.0,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
    )
    if vector.shape != (324,) or not np.isfinite(vector).all():
        raise ValueError('Error al construir las 324 características HOG.')
    return vector.reshape(1, -1)

@lru_cache(maxsize=1)
def cargar_modelo():
    if not RUTA_MODELO.is_file():
        raise FileNotFoundError('Modelo no encontrado. Ejecuta scripts/obtener_modelo.py.')
    modelo = joblib.load(RUTA_MODELO)
    if not hasattr(modelo, 'predict_proba') or not hasattr(modelo, 'classes_'):
        raise RuntimeError('El archivo no contiene un clasificador compatible.')
    clases = [int(c) for c in modelo.classes_]
    if sorted(clases) != [0, 1, 2, 3]:
        raise RuntimeError('El modelo no presenta las cuatro clases esperadas.')
    if int(getattr(modelo, 'n_features_in_', -1)) != 324:
        raise RuntimeError('El modelo activo no acepta las 324 características HOG esperadas.')
    return modelo

def predecir_imagen(contenido: bytes) -> dict:
    x = extraer_caracteristicas(preparar_imagen(contenido))
    modelo = cargar_modelo()
    probabilidades = np.asarray(modelo.predict_proba(x)[0], dtype=float)
    clases = [int(v) for v in modelo.classes_]
    elegido = clases[int(np.argmax(probabilidades))]
    prediccion_modelo = int(modelo.predict(x)[0])
    if prediccion_modelo != elegido:
        raise RuntimeError('predict y el máximo de predict_proba no coinciden.')
    probabilidades_ordenadas = np.sort(probabilidades)[::-1]
    margen = float(probabilidades_ordenadas[0] - probabilidades_ordenadas[1])
    ambigua = margen < UMBRAL_MARGEN_AMBIGUO
    vector = [
        {'id': int(clase), 'clase': ETIQUETAS[int(clase)], 'probabilidad': round(float(valor), 6)}
        for clase, valor in zip(clases, probabilidades)
    ]
    vector.sort(key=lambda item: item['id'])
    return {
        'clase_id': elegido,
        'clase': ETIQUETAS[elegido],
        'confianza_estimada': round(float(np.max(probabilidades)), 6),
        'margen_probabilidades': round(margen, 6),
        'ambigua': ambigua,
        'estado_analisis': 'RESULTADO NO CONCLUYENTE' if ambigua else 'ANÁLISIS COMPLETADO',
        'probabilidades': vector,
        'resultado': 'Sin defecto según el modelo' if elegido == 3 else 'Posible defecto: requiere revisión',
        'advertencia': 'Demostración académica con radiografías RIAWELC. No es un dictamen industrial. Las probabilidades no están calibradas.',
        'modelo': NOMBRE_MODELO,
    }
