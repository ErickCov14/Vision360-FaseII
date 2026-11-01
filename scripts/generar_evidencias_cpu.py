#!/usr/bin/env python3
# RESULTADOS
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'evidencias/experimentos_cpu'
CLASES = ['Grieta', 'Porosidad', 'Falta penetr.', 'Sin defecto']


def svg_matriz(ruta: Path, matriz: list[list[int]], titulo: str) -> None:
    x0, y0, celda, maximo = 280, 120, 92, max(max(f) for f in matriz)
    partes = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="820" height="620" viewBox="0 0 820 620">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#172033}.title{font-size:23px;font-weight:700}.label{font-size:14px}.value{font-size:22px;font-weight:700}</style>',
        f'<text x="410" y="42" text-anchor="middle" class="title">{titulo}</text>',
        '<text x="464" y="82" text-anchor="middle" class="label">Predicción</text>',
        '<text x="38" y="310" text-anchor="middle" class="label" transform="rotate(-90 38 310)">Clase real</text>',
    ]
    for i, nombre in enumerate(CLASES):
        partes.append(f'<text x="{x0+i*celda+celda/2}" y="105" text-anchor="middle" class="label">{nombre}</text>')
        partes.append(f'<text x="265" y="{y0+i*celda+celda/2+5}" text-anchor="end" class="label">{nombre}</text>')
    for i in range(4):
        for j in range(4):
            valor = int(matriz[i][j]); intensidad = valor / max(maximo, 1)
            color = f'rgb({int(238-120*intensidad)},{int(247-42*intensidad)},{int(255-20*intensidad)})'
            x, y = x0+j*celda, y0+i*celda
            partes.append(f'<rect x="{x}" y="{y}" width="{celda}" height="{celda}" fill="{color}" stroke="#fff" stroke-width="2"/>')
            partes.append(f'<text x="{x+celda/2}" y="{y+celda/2+8}" text-anchor="middle" class="value">{valor}</text>')
    partes.append('</svg>')
    ruta.write_text('\n'.join(partes), encoding='utf-8')


def csv_escribir(ruta: Path, filas: list[dict]) -> None:
    with ruta.open('w', newline='', encoding='utf-8-sig') as archivo:
        writer = csv.DictWriter(archivo, fieldnames=list(filas[0]))
        writer.writeheader(); writer.writerows(filas)


def main() -> None:
    matrices = BASE / 'matrices'; matrices.mkdir(exist_ok=True)
    modelos = json.loads((BASE / 'comparacion_modelos_validacion.json').read_text(encoding='utf-8'))
    filas_modelos = []
    for resultado in modelos['resultados']:
        met = resultado['validacion']; nombre = resultado['modelo']
        filas_modelos.append({
            'modelo': nombre, 'f1_macro_validacion': met['f1_macro'],
            'accuracy_validacion': met['accuracy'],
            'recall_grieta': met['por_clase']['0']['recall'],
            'recall_falta_penetracion': met['por_clase']['2']['recall'],
            'defectos_como_sin_defecto': met['defectos_clasificados_sin_defecto'],
            'busqueda_segundos': resultado['entrenamiento_busqueda_segundos'],
            'inferencia_ms_imagen': met['inferencia_ms_por_imagen_modelo'],
            'tamano_modelo_bytes': resultado['tamano_modelo_busqueda_bytes'],
        })
        svg_matriz(matrices / f'validacion_{nombre}.svg', met['matriz_confusion'], f'Validación agrupada — {nombre}')
    csv_escribir(BASE / 'comparacion_modelos_validacion.csv', filas_modelos)

    features = json.loads((BASE / 'comparacion_caracteristicas.json').read_text(encoding='utf-8'))
    filas_features = []
    for resultado in features['resultados']:
        met = resultado['validacion']
        filas_features.append({
            'configuracion': resultado['configuracion'], 'dimensiones': resultado['dimensiones'],
            'f1_macro_validacion': met['f1_macro'], 'accuracy_validacion': met['accuracy'],
            'recall_prioritario': met['recall_prioritario_promedio'],
            'defectos_como_sin_defecto': met['defectos_clasificados_sin_defecto'],
        })
    csv_escribir(BASE / 'comparacion_caracteristicas.csv', filas_features)

    final = json.loads((BASE / 'evaluacion_final_una_vez.json').read_text(encoding='utf-8'))
    svg_matriz(matrices / 'test_baseline_rf.svg', final['baseline_rf_agrupado']['metricas_test']['matriz_confusion'], 'Test agrupado — RF baseline')
    svg_matriz(matrices / 'test_candidato_rf_optimizado.svg', final['candidato']['metricas_test']['matriz_confusion'], 'Test agrupado — RF optimizado')
    print('Tablas y matrices generadas en', BASE)


if __name__ == '__main__':
    main()
