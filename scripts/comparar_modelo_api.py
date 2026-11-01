#!/usr/bin/env python3
# VALIDACION
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import joblib
import numpy as np
from PIL import Image, ImageOps
from skimage.feature import hog

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'backend'))

EJEMPLOS = {
    '01_grieta.png': 0,
    '02_porosidad.png': 1,
    '03_falta_penetracion.png': 2,
    '04_sin_defecto.png': 3,
}


def vector_original(contenido: bytes) -> np.ndarray:
    with Image.open(BytesIO(contenido)) as imagen:
        imagen = ImageOps.exif_transpose(imagen).convert('L')
        matriz = np.asarray(imagen.resize((64, 64), Image.Resampling.LANCZOS), dtype=np.uint8)
    return hog(
        matriz / 255.0, orientations=9, pixels_per_cell=(16, 16),
        cells_per_block=(2, 2), block_norm='L2-Hys',
    ).reshape(1, -1)


def llamar_api(url: str, nombre: str, contenido: bytes) -> dict:
    boundary = 'vision360-comparacion-boundary'
    cuerpo = (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{nombre}"\r\nContent-Type: image/png\r\n\r\n'.encode()
              + contenido + f'\r\n--{boundary}--\r\n'.encode())
    request = urllib.request.Request(
        url.rstrip('/') + '/api/predict', data=cuerpo, method='POST',
        headers={'Content-Type': f'multipart/form-data; boundary={boundary}'},
    )
    with urllib.request.urlopen(request, timeout=30) as respuesta:
        return json.load(respuesta)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://localhost:8080')
    modelo_optimizado = ROOT / 'backend/model/random_forest_vision360_cpu_optimizado.joblib'
    parser.add_argument('--model', type=Path, default=modelo_optimizado if modelo_optimizado.is_file() else ROOT / 'backend/model/random_forest_vision360.joblib')
    parser.add_argument('--examples-dir', type=Path, default=ROOT / 'ejemplos')
    parser.add_argument('--save', type=Path, default=ROOT / 'evidencias/comparacion_modelo_api.json')
    args = parser.parse_args()
    os.environ['MODEL_PATH'] = str(args.model.resolve())
    from app import inferencia

    modelo = joblib.load(args.model)
    clases = [int(c) for c in modelo.classes_]
    resultados = []
    for nombre, clase_real in EJEMPLOS.items():
        contenido = (args.examples_dir / nombre).read_bytes()
        referencia = vector_original(contenido)
        backend = inferencia.extraer_caracteristicas(inferencia.preparar_imagen(contenido))
        np.testing.assert_allclose(backend, referencia, rtol=0, atol=1e-12)
        pred_directa = int(modelo.predict(referencia)[0])
        proba_directa = np.asarray(modelo.predict_proba(referencia)[0], dtype=float)
        respuesta_api = llamar_api(args.url, nombre, contenido)
        proba_api = {int(p['id']): float(p['probabilidad']) for p in respuesta_api['probabilidades']}
        diferencias = [abs(float(valor) - proba_api[clase]) for clase, valor in zip(clases, proba_directa)]
        if respuesta_api['clase_id'] != pred_directa or max(diferencias) > 5e-7:
            raise RuntimeError(f'La API no coincide con el modelo directo para {nombre}')
        resultados.append({
            'archivo': nombre,
            'clase_real_id': clase_real,
            'vector_caracteristicas': int(referencia.shape[1]),
            'max_diferencia_vector_original_backend': float(np.max(np.abs(referencia - backend))),
            'prediccion_directa_id': pred_directa,
            'prediccion_api_id': int(respuesta_api['clase_id']),
            'nombre_api': respuesta_api['clase'],
            'probabilidades_directas_por_clase_modelo': {
                str(clase): float(valor) for clase, valor in zip(clases, proba_directa)
            },
            'probabilidades_api_por_clase_modelo': {str(k): v for k, v in proba_api.items()},
            'max_diferencia_probabilidades': max(diferencias),
            'estado_api': respuesta_api['estado_analisis'],
            'coincide': True,
        })
    reporte = {
        'fecha_utc': datetime.now(timezone.utc).isoformat(),
        'api': args.url,
        'tolerancia_vector': 1e-12,
        'tolerancia_probabilidades': 5e-7,
        'clases_modelo': clases,
        'imagenes_comparadas': len(resultados),
        'todas_coinciden': all(r['coincide'] for r in resultados),
        'resultados': resultados,
    }
    args.save.parent.mkdir(parents=True, exist_ok=True)
    args.save.write_text(json.dumps(reporte, indent=2, ensure_ascii=False), encoding='utf-8')
    print('COMPARACIÓN MODELO DIRECTO / API: APROBADA')
    print('Imágenes:', len(resultados))
    print('Evidencia:', args.save)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
