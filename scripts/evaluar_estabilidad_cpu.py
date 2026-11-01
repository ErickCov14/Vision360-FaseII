#!/usr/bin/env python3
# VALIDACION
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, recall_score
from sklearn.model_selection import StratifiedGroupKFold, cross_validate

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from optimizar_modelo_cpu import evaluar, guardar_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiments-dir', type=Path, default=ROOT / 'evidencias/experimentos_cpu')
    args = parser.parse_args()
    base = args.experiments_dir.resolve()
    with (base / 'inventario_agrupado.csv').open(encoding='utf-8-sig', newline='') as archivo:
        filas = list(csv.DictReader(archivo))
    y = np.array([int(f['etiqueta']) for f in filas])
    grupos = np.array([f['grupo'] for f in filas])
    particiones = np.array([f['particion_agrupada'] for f in filas])
    x = np.load(base / 'cache_caracteristicas/hog64.npy')
    train = np.flatnonzero(particiones == 'training')
    val = np.flatnonzero(particiones == 'validation')

    baseline = RandomForestClassifier(
        n_estimators=200, max_depth=20, min_samples_split=2, min_samples_leaf=1,
        max_features='sqrt', class_weight=None, random_state=42, n_jobs=2,
    )
    inicio = time.perf_counter()
    baseline.fit(x[train], y[train])
    baseline_val, _ = evaluar(baseline, x[val], y[val])
    baseline_val['entrenamiento_segundos'] = time.perf_counter() - inicio

    modelos = {'baseline_rf_hog64': baseline}
    for nombre in ('random_forest', 'extra_trees', 'svc_rbf', 'hist_gradient_boosting'):
        modelos[nombre] = joblib.load(base / 'busquedas' / f'{nombre}.joblib')
    scorers = {
        'f1_macro': 'f1_macro',
        'recall_grieta': make_scorer(recall_score, labels=[0], average='macro', zero_division=0),
        'recall_falta_penetracion': make_scorer(recall_score, labels=[2], average='macro', zero_division=0),
    }
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=142)
    resultados = []
    for nombre, modelo in modelos.items():
        print('Estabilidad:', nombre, flush=True)
        inicio = time.perf_counter()
        scores = cross_validate(
            modelo, x[train], y[train], groups=grupos[train], cv=cv,
            scoring=scorers, n_jobs=1, return_train_score=False,
        )
        fila = {'modelo': nombre, 'segundos': time.perf_counter() - inicio, 'folds': 5}
        for metrica in scorers:
            valores = scores[f'test_{metrica}']
            fila[metrica + '_folds'] = [float(v) for v in valores]
            fila[metrica + '_media'] = float(np.mean(valores))
            fila[metrica + '_desviacion'] = float(np.std(valores))
        resultados.append(fila)
        print(f"  F1={fila['f1_macro_media']:.4f} ± {fila['f1_macro_desviacion']:.4f}", flush=True)
    guardar_json(base / 'estabilidad_cv_agrupada.json', {
        'datos': 'solo training agrupado', 'metodo': 'StratifiedGroupKFold 5 folds',
        'baseline_validacion': baseline_val, 'resultados': resultados,
    })
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
