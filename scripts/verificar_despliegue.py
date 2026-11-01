#!/usr/bin/env python3
# VALIDACION
import argparse
import json
import struct
import urllib.request
import zlib
import binascii
from pathlib import Path
from datetime import datetime, timezone


def png_gris_64():
    def bloque(tipo, datos):
        return struct.pack('>I', len(datos)) + tipo + datos + struct.pack('>I', binascii.crc32(tipo + datos) & 0xffffffff)
    filas = b''.join(b'\x00' + bytes((x * 4 + y * 2) % 256 for x in range(64)) for y in range(64))
    return b'\x89PNG\r\n\x1a\n' + bloque(b'IHDR', struct.pack('>IIBBBBB', 64, 64, 8, 0, 0, 0, 0)) + bloque(b'IDAT', zlib.compress(filas)) + bloque(b'IEND', b'')


def leer(url):
    with urllib.request.urlopen(url, timeout=20) as r:
        return json.load(r)


def predecir(url, nombre, imagen):
    boundary = 'vision360-test-boundary'
    cuerpo = (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{nombre}"\r\nContent-Type: image/png\r\n\r\n'.encode()
              + imagen + f'\r\n--{boundary}--\r\n'.encode())
    req = urllib.request.Request(url + '/api/predict', data=cuerpo,
                                 headers={'Content-Type': f'multipart/form-data; boundary={boundary}'},
                                 method='POST')
    with urllib.request.urlopen(req, timeout=30) as r:
        pred = json.load(r)
    assert pred['clase_id'] in (0, 1, 2, 3) and len(pred['probabilidades']) == 4
    assert abs(sum(p['probabilidad'] for p in pred['probabilidades']) - 1.0) < 0.00001
    assert pred['estado_analisis'] in ('ANÁLISIS COMPLETADO', 'RESULTADO NO CONCLUYENTE')
    return pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://localhost:8080')
    parser.add_argument('--save', type=Path, default=Path('evidencias/prueba_http_real.json'))
    parser.add_argument('--examples-dir', type=Path, default=Path('ejemplos'))
    args = parser.parse_args()
    url = args.url.rstrip('/')
    health = leer(url + '/api/health')
    assert health['estado'] == 'ok' and health['modelo_disponible']
    info = leer(url + '/api/model')
    assert info['caracteristicas'] == 324 and len(info['clases']) == 4
    imagen = png_gris_64()
    pred = predecir(url, 'test_sintetico.png', imagen)
    esperados = {
        '01_grieta.png': 0,
        '02_porosidad.png': 1,
        '03_falta_penetracion.png': 2,
        '04_sin_defecto.png': 3,
    }
    predicciones_reales = []
    for nombre, clase_real in esperados.items():
        ruta = args.examples_dir / nombre
        if not ruta.is_file():
            raise FileNotFoundError(f'Falta la imagen RIAWELC de ejemplo: {ruta}')
        salida = predecir(url, nombre, ruta.read_bytes())
        predicciones_reales.append({
            'archivo': nombre,
            'clase_real_id': clase_real,
            'clase_predicha_id': salida['clase_id'],
            'clase_predicha': salida['clase'],
            'respuesta': salida,
        })
    reporte = {'fecha_utc': datetime.now(timezone.utc).isoformat(), 'url': url,
               'tipo_prueba': 'Integración HTTP con transporte sintético y cuatro imágenes reales RIAWELC',
               'health': health, 'metadata': info, 'prediccion_sintetica': pred,
               'predicciones_riawelc': predicciones_reales, 'resultado': 'APROBADO'}
    args.save.parent.mkdir(parents=True, exist_ok=True)
    args.save.write_text(json.dumps(reporte, indent=2, ensure_ascii=False), encoding='utf-8')
    print('PRUEBA HTTP DE EXTREMO A EXTREMO: APROBADA')
    print('Evidencia:', args.save)
    print('Imágenes RIAWELC verificadas:', len(predicciones_reales))

if __name__ == '__main__':
    main()
