#!/usr/bin/env python3
# ENTRENAMIENTO
from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import json
import os
import platform
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from PIL import Image, ImageOps
from scipy.stats import loguniform
from skimage.feature import hog, local_binary_pattern
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
CLASES = {0: 'Grieta', 1: 'Porosidad', 2: 'Falta de penetración', 3: 'Sin defecto'}
CARPETAS = {'Difetto1': 0, 'Difetto2': 1, 'Difetto4': 2, 'NoDifetto': 3}
SEMILLA = 42


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def memoria_sistema() -> tuple[int, int]:
    if os.name == 'nt':
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong),
                ('ullTotalPhys', ctypes.c_ulonglong), ('ullAvailPhys', ctypes.c_ulonglong),
                ('ullTotalPageFile', ctypes.c_ulonglong), ('ullAvailPageFile', ctypes.c_ulonglong),
                ('ullTotalVirtual', ctypes.c_ulonglong), ('ullAvailVirtual', ctypes.c_ulonglong),
                ('ullAvailExtendedVirtual', ctypes.c_ulonglong),
            ]
        estado = MEMORYSTATUSEX()
        estado.dwLength = ctypes.sizeof(estado)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(estado))
        return int(estado.ullTotalPhys), int(estado.ullAvailPhys)
    paginas = os.sysconf('SC_PHYS_PAGES')
    disponibles = os.sysconf('SC_AVPHYS_PAGES')
    tamano = os.sysconf('SC_PAGE_SIZE')
    return paginas * tamano, disponibles * tamano


def memoria_proceso_pico() -> int | None:
    if os.name != 'nt':
        try:
            import resource
            return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        except Exception:
            return None
    class PMC(ctypes.Structure):
        _fields_ = [
            ('cb', ctypes.c_ulong), ('PageFaultCount', ctypes.c_ulong),
            ('PeakWorkingSetSize', ctypes.c_size_t), ('WorkingSetSize', ctypes.c_size_t),
            ('QuotaPeakPagedPoolUsage', ctypes.c_size_t), ('QuotaPagedPoolUsage', ctypes.c_size_t),
            ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t), ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
            ('PagefileUsage', ctypes.c_size_t), ('PeakPagefileUsage', ctypes.c_size_t),
        ]
    contador = PMC()
    contador.cb = ctypes.sizeof(contador)
    proceso = ctypes.windll.kernel32.GetCurrentProcess()
    if ctypes.windll.psapi.GetProcessMemoryInfo(proceso, ctypes.byref(contador), contador.cb):
        return int(contador.PeakWorkingSetSize)
    return None


def sha256(ruta: Path) -> str:
    digest = hashlib.sha256()
    with ruta.open('rb') as archivo:
        for bloque in iter(lambda: archivo.read(1024 * 1024), b''):
            digest.update(bloque)
    return digest.hexdigest()


def grupo_origen(nombre: str) -> str:
    return re.sub(r'_\[\d+\]\[\d+\]$', '', Path(nombre).stem)


def guardar_json(ruta: Path, datos: Any) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(datos, ensure_ascii=False, indent=2, default=str), encoding='utf-8')


def escribir_csv(ruta: Path, filas: list[dict], campos: list[str] | None = None) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    if not filas:
        return
    with ruta.open('w', newline='', encoding='utf-8-sig') as archivo:
        writer = csv.DictWriter(archivo, fieldnames=campos or list(filas[0]))
        writer.writeheader()
        writer.writerows(filas)


def inventariar(dataset: Path) -> list[dict]:
    filas = []
    for particion in ('training', 'validation', 'testing'):
        for carpeta, etiqueta in CARPETAS.items():
            for ruta in sorted((dataset / particion / carpeta).glob('*.png')):
                with Image.open(ruta) as imagen:
                    dimensiones = f'{imagen.width}x{imagen.height}'
                filas.append({
                    'ruta': str(ruta.resolve()), 'archivo': ruta.name,
                    'particion_original': particion, 'carpeta': carpeta,
                    'etiqueta': etiqueta, 'clase': CLASES[etiqueta],
                    'grupo_nombre': grupo_origen(ruta.name), 'sha256': sha256(ruta),
                    'dimensiones': dimensiones, 'bytes': ruta.stat().st_size,
                })
    if len(filas) != 2000:
        raise RuntimeError(f'Se esperaban 2,000 imágenes y se encontraron {len(filas)}.')
    return filas


def conectar_grupos(filas: list[dict]) -> dict:
    uf = UnionFind(len(filas))
    por_nombre: dict[str, int] = {}
    por_hash: dict[str, int] = {}
    for i, fila in enumerate(filas):
        nombre, digest = fila['grupo_nombre'], fila['sha256']
        if nombre in por_nombre:
            uf.union(i, por_nombre[nombre])
        else:
            por_nombre[nombre] = i
        if digest in por_hash:
            uf.union(i, por_hash[digest])
        else:
            por_hash[digest] = i
    raices = sorted({uf.find(i) for i in range(len(filas))})
    ids = {raiz: f'G{n:04d}' for n, raiz in enumerate(raices)}
    for i, fila in enumerate(filas):
        fila['grupo'] = ids[uf.find(i)]

    por_grupo: dict[str, list[dict]] = defaultdict(list)
    for fila in filas:
        por_grupo[fila['grupo']].append(fila)
    conflictos = []
    for grupo, items in por_grupo.items():
        etiquetas = sorted({int(x['etiqueta']) for x in items})
        if len(etiquetas) > 1:
            conflictos.append({'grupo': grupo, 'etiquetas': etiquetas, 'archivos': [x['archivo'] for x in items]})
    hashes_cruzados = sum(
        1 for digest in {f['sha256'] for f in filas}
        if len({f['particion_original'] for f in filas if f['sha256'] == digest}) > 1
    )
    nombres_cruzados = sum(
        1 for nombre in {f['grupo_nombre'] for f in filas}
        if len({f['particion_original'] for f in filas if f['grupo_nombre'] == nombre}) > 1
    )
    return {
        'imagenes': len(filas), 'grupos_conectados': len(por_grupo),
        'hashes_exactos_entre_particiones_originales': hashes_cruzados,
        'grupos_nombre_entre_particiones_originales': nombres_cruzados,
        'grupos_con_etiquetas_mixtas': conflictos,
        'criterio': 'componente conectado por nombre sin coordenadas o SHA-256 idéntico',
    }


def construir_particiones(filas: list[dict]) -> dict:
    y = np.array([int(f['etiqueta']) for f in filas])
    grupos = np.array([f['grupo'] for f in filas])
    dummy = np.zeros((len(filas), 1), dtype=np.uint8)
    exterior = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEMILLA)
    indices_desarrollo, indices_test = next(exterior.split(dummy, y, grupos))
    interior = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=SEMILLA + 1)
    train_local, val_local = next(
        interior.split(dummy[indices_desarrollo], y[indices_desarrollo], grupos[indices_desarrollo])
    )
    indices_train = indices_desarrollo[train_local]
    indices_val = indices_desarrollo[val_local]
    asignacion = np.empty(len(filas), dtype=object)
    asignacion[indices_train] = 'training'
    asignacion[indices_val] = 'validation'
    asignacion[indices_test] = 'testing'
    for i, fila in enumerate(filas):
        fila['particion_agrupada'] = str(asignacion[i])

    grupos_particiones: dict[str, set[str]] = defaultdict(set)
    hashes_particiones: dict[str, set[str]] = defaultdict(set)
    for fila in filas:
        grupos_particiones[fila['grupo']].add(fila['particion_agrupada'])
        hashes_particiones[fila['sha256']].add(fila['particion_agrupada'])
    solapamiento_grupos = {g: sorted(v) for g, v in grupos_particiones.items() if len(v) > 1}
    solapamiento_hashes = {h: sorted(v) for h, v in hashes_particiones.items() if len(v) > 1}
    if solapamiento_grupos or solapamiento_hashes:
        raise RuntimeError('La construcción agrupada dejó contaminación entre particiones.')

    resumen = {}
    for particion in ('training', 'validation', 'testing'):
        items = [f for f in filas if f['particion_agrupada'] == particion]
        resumen[particion] = {
            'imagenes': len(items), 'grupos': len({f['grupo'] for f in items}),
            'por_clase': {CLASES[c]: sum(int(f['etiqueta']) == c for f in items) for c in CLASES},
        }
    return {
        'semilla': SEMILLA,
        'metodo': 'StratifiedGroupKFold exterior 5 folds (fold 0 test); interior 4 folds (fold 0 validation)',
        'resumen': resumen,
        'grupos_compartidos': len(solapamiento_grupos),
        'hashes_compartidos': len(solapamiento_hashes),
        'test_reservado': True,
    }


def cargar_gris(ruta: str, tamano: int) -> np.ndarray:
    with Image.open(ruta) as imagen:
        imagen = ImageOps.exif_transpose(imagen).convert('L')
        return np.asarray(imagen.resize((tamano, tamano), Image.Resampling.LANCZOS), dtype=np.uint8)


def hog_vector(imagen: np.ndarray) -> np.ndarray:
    return hog(
        imagen / 255.0, orientations=9, pixels_per_cell=(16, 16),
        cells_per_block=(2, 2), block_norm='L2-Hys',
    ).astype(np.float32)


def lbp_vector(imagen: np.ndarray) -> np.ndarray:
    puntos, radio, cuadricula = 24, 3, 4
    codigos = local_binary_pattern(imagen, puntos, radio, method='uniform')
    bins = puntos + 2
    lado = imagen.shape[0] // cuadricula
    histogramas = []
    for fila in range(cuadricula):
        for columna in range(cuadricula):
            region = codigos[fila * lado:(fila + 1) * lado, columna * lado:(columna + 1) * lado]
            hist, _ = np.histogram(region, bins=np.arange(bins + 1), range=(0, bins))
            hist = hist.astype(np.float32)
            hist /= max(float(hist.sum()), 1.0)
            histogramas.append(hist)
    return np.concatenate(histogramas).astype(np.float32)


def obtener_caracteristicas(filas: list[dict], salida: Path) -> tuple[dict[str, np.ndarray], list[dict]]:
    cache = salida / 'cache_caracteristicas'
    cache.mkdir(parents=True, exist_ok=True)
    base: dict[str, np.ndarray] = {}
    definiciones = {
        'hog64': (64, hog_vector),
        'hog128': (128, hog_vector),
        'lbp128': (128, lbp_vector),
    }
    registro = []
    for nombre, (tamano, funcion) in definiciones.items():
        ruta_cache = cache / f'{nombre}.npy'
        inicio = time.perf_counter()
        if ruta_cache.is_file():
            matriz = np.load(ruta_cache, mmap_mode=None)
            origen = 'cache'
        else:
            vectores = []
            for n, fila in enumerate(filas, start=1):
                vectores.append(funcion(cargar_gris(fila['ruta'], tamano)))
                if n % 250 == 0:
                    print(f'  {nombre}: {n}/{len(filas)}', flush=True)
            matriz = np.vstack(vectores).astype(np.float32)
            np.save(ruta_cache, matriz, allow_pickle=False)
            origen = 'calculado'
        base[nombre] = matriz
        registro.append({
            'configuracion': nombre, 'resolucion': f'{tamano}x{tamano}',
            'dimensiones': int(matriz.shape[1]), 'dtype': str(matriz.dtype),
            'memoria_bytes': int(matriz.nbytes), 'memoria_mib': matriz.nbytes / 2**20,
            'segundos': time.perf_counter() - inicio, 'origen': origen,
        })
    combinada = np.hstack([base['hog128'], base['lbp128']]).astype(np.float32)
    base['hog_lbp128'] = combinada
    registro.append({
        'configuracion': 'hog_lbp128', 'resolucion': '128x128',
        'dimensiones': int(combinada.shape[1]), 'dtype': str(combinada.dtype),
        'memoria_bytes': int(combinada.nbytes), 'memoria_mib': combinada.nbytes / 2**20,
        'segundos': 0.0, 'origen': 'concatenación de cachés',
    })
    return base, registro


def metricas(y_real: np.ndarray, y_pred: np.ndarray) -> dict:
    precision, recall, f1, soporte = precision_recall_fscore_support(
        y_real, y_pred, labels=[0, 1, 2, 3], zero_division=0,
    )
    matriz = confusion_matrix(y_real, y_pred, labels=[0, 1, 2, 3])
    por_clase = {}
    for i, clase in enumerate((0, 1, 2, 3)):
        por_clase[str(clase)] = {
            'clase': CLASES[clase], 'precision': float(precision[i]),
            'recall': float(recall[i]), 'f1': float(f1[i]),
            'soporte': int(soporte[i]), 'falsos_negativos': int(soporte[i] - matriz[i, i]),
        }
    defectos_como_sin_defecto = int(sum(matriz[c, 3] for c in (0, 1, 2)))
    return {
        'accuracy': float(accuracy_score(y_real, y_pred)),
        'f1_macro': float(np.mean(f1)),
        'recall_defectos_promedio': float(np.mean(recall[:3])),
        'recall_prioritario_promedio': float(np.mean(recall[[0, 2]])),
        'defectos_clasificados_sin_defecto': defectos_como_sin_defecto,
        'por_clase': por_clase,
        'matriz_confusion': matriz.tolist(),
    }


def evaluar(modelo, x: np.ndarray, y: np.ndarray) -> tuple[dict, np.ndarray]:
    inicio = time.perf_counter()
    pred = modelo.predict(x)
    segundos = time.perf_counter() - inicio
    resultado = metricas(y, pred)
    resultado['inferencia_segundos_total'] = segundos
    resultado['inferencia_ms_por_imagen_modelo'] = 1000 * segundos / len(y)
    return resultado, pred


def indices_particion(filas: list[dict], nombre: str) -> np.ndarray:
    return np.array([i for i, fila in enumerate(filas) if fila['particion_agrupada'] == nombre], dtype=int)


def comparar_caracteristicas(
    matrices: dict[str, np.ndarray], filas: list[dict], salida: Path, hilos_arbol: int,
) -> tuple[str, list[dict]]:
    ruta = salida / 'comparacion_caracteristicas.json'
    if ruta.is_file():
        datos = json.loads(ruta.read_text(encoding='utf-8'))
        return datos['seleccionada'], datos['resultados']
    y = np.array([int(f['etiqueta']) for f in filas])
    train, val = indices_particion(filas, 'training'), indices_particion(filas, 'validation')
    resultados = []
    for nombre, x in matrices.items():
        modelo = ExtraTreesClassifier(
            n_estimators=250, max_depth=None, min_samples_leaf=1,
            max_features='sqrt', class_weight='balanced',
            random_state=SEMILLA, n_jobs=hilos_arbol,
        )
        inicio = time.perf_counter()
        modelo.fit(x[train], y[train])
        met, _ = evaluar(modelo, x[val], y[val])
        resultados.append({
            'configuracion': nombre, 'dimensiones': int(x.shape[1]),
            'entrenamiento_segundos': time.perf_counter() - inicio,
            'validacion': met,
        })
        print(f"Característica {nombre}: F1 macro={met['f1_macro']:.4f}", flush=True)
    mejor = max(
        resultados,
        key=lambda r: (
            r['validacion']['f1_macro'], r['validacion']['recall_prioritario_promedio'],
            -r['validacion']['defectos_clasificados_sin_defecto'],
        ),
    )['configuracion']
    guardar_json(ruta, {'metodo': 'ExtraTrees fijo en training; comparación exclusivamente en validation', 'seleccionada': mejor, 'resultados': resultados})
    return mejor, resultados


def modelos_busqueda(hilos_arbol: int, cache_svc_mb: int) -> dict:
    return {
        'random_forest': (
            RandomForestClassifier(random_state=SEMILLA, n_jobs=hilos_arbol),
            {
                'n_estimators': [200, 350, 500], 'max_depth': [None, 20, 30, 40],
                'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', 0.5],
                'class_weight': [None, 'balanced', 'balanced_subsample'],
            },
            10,
        ),
        'extra_trees': (
            ExtraTreesClassifier(random_state=SEMILLA, n_jobs=hilos_arbol),
            {
                'n_estimators': [250, 400, 600], 'max_depth': [None, 20, 30, 40],
                'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', 0.5],
                'class_weight': [None, 'balanced', 'balanced_subsample'],
            },
            10,
        ),
        'svc_rbf': (
            Pipeline([
                ('scaler', StandardScaler()),
                ('svc', SVC(kernel='rbf', probability=False, cache_size=cache_svc_mb, random_state=SEMILLA)),
            ]),
            {
                'svc__C': loguniform(0.1, 100), 'svc__gamma': loguniform(1e-5, 1e-1),
                'svc__class_weight': [None, 'balanced'],
            },
            10,
        ),
        'hist_gradient_boosting': (
            HistGradientBoostingClassifier(random_state=SEMILLA, early_stopping=True),
            {
                'learning_rate': [0.03, 0.06, 0.1], 'max_iter': [80, 120, 160],
                'max_leaf_nodes': [15, 31, 63], 'max_depth': [None, 10, 20],
                'min_samples_leaf': [10, 20, 30], 'l2_regularization': [0.0, 0.1, 1.0],
                'class_weight': [None, 'balanced'],
            },
            6,
        ),
    }


def buscar_modelos(
    x: np.ndarray, filas: list[dict], salida: Path, hilos_arbol: int, trabajos_cv: int,
    cache_svc_mb: int,
) -> tuple[str, dict[str, Any], list[dict]]:
    y = np.array([int(f['etiqueta']) for f in filas])
    grupos = np.array([f['grupo'] for f in filas])
    train, val = indices_particion(filas, 'training'), indices_particion(filas, 'validation')
    cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEMILLA)
    modelos: dict[str, Any] = {}
    resultados = []
    carpeta = salida / 'busquedas'
    carpeta.mkdir(parents=True, exist_ok=True)
    for nombre, (estimador, parametros, iteraciones) in modelos_busqueda(hilos_arbol, cache_svc_mb).items():
        ruta_modelo, ruta_resultado = carpeta / f'{nombre}.joblib', carpeta / f'{nombre}.json'
        if ruta_modelo.is_file() and ruta_resultado.is_file():
            modelo = joblib.load(ruta_modelo)
            resultado = json.loads(ruta_resultado.read_text(encoding='utf-8'))
            print(f'Reanudado: {nombre}', flush=True)
        else:
            print(f'Buscando: {nombre} ({iteraciones} configuraciones, CV agrupada de 3 folds)', flush=True)
            busqueda = RandomizedSearchCV(
                estimador, parametros, n_iter=iteraciones, scoring='f1_macro', refit=True,
                cv=cv, random_state=SEMILLA, n_jobs=trabajos_cv,
                pre_dispatch=trabajos_cv, return_train_score=False, verbose=1,
            )
            inicio = time.perf_counter()
            busqueda.fit(x[train], y[train], groups=grupos[train])
            duracion = time.perf_counter() - inicio
            modelo = busqueda.best_estimator_
            met, _ = evaluar(modelo, x[val], y[val])
            resultado = {
                'modelo': nombre, 'iteraciones': iteraciones, 'cv_folds': 3,
                'mejor_f1_macro_cv': float(busqueda.best_score_),
                'mejores_parametros': busqueda.best_params_,
                'entrenamiento_busqueda_segundos': duracion,
                'validacion': met,
                'memoria_proceso_pico_bytes': memoria_proceso_pico(),
            }
            joblib.dump(modelo, ruta_modelo)
            resultado['tamano_modelo_busqueda_bytes'] = ruta_modelo.stat().st_size
            guardar_json(ruta_resultado, resultado)
        modelos[nombre] = modelo
        resultados.append(resultado)
        print(f"  {nombre}: validación F1={resultado['validacion']['f1_macro']:.4f}", flush=True)
    seleccionado = max(
        resultados,
        key=lambda r: (
            r['validacion']['f1_macro'], r['validacion']['recall_prioritario_promedio'],
            -r['validacion']['defectos_clasificados_sin_defecto'],
            -r['validacion']['inferencia_ms_por_imagen_modelo'],
        ),
    )['modelo']
    guardar_json(salida / 'comparacion_modelos_validacion.json', {
        'protocolo': 'RandomizedSearchCV agrupada solo en training; selección entre modelos solo en validation',
        'seleccionado_antes_de_test': seleccionado, 'resultados': resultados,
    })
    return seleccionado, modelos, resultados


def evaluacion_final(
    seleccionado: str, modelos: dict[str, Any], x: np.ndarray, filas: list[dict],
    salida: Path, hilos_arbol: int, configuracion: str,
) -> dict:
    ruta_resultado = salida / 'evaluacion_final_una_vez.json'
    if ruta_resultado.is_file():
        return json.loads(ruta_resultado.read_text(encoding='utf-8'))
    y = np.array([int(f['etiqueta']) for f in filas])
    train, val, test = (
        indices_particion(filas, 'training'), indices_particion(filas, 'validation'),
        indices_particion(filas, 'testing'),
    )
    desarrollo = np.concatenate([train, val])
    baseline = RandomForestClassifier(
        n_estimators=200, max_depth=20, min_samples_split=2, min_samples_leaf=1,
        max_features='sqrt', class_weight=None, random_state=SEMILLA, n_jobs=hilos_arbol,
    )
    inicio = time.perf_counter()
    baseline.fit(x[desarrollo], y[desarrollo])
    tiempo_baseline = time.perf_counter() - inicio
    met_baseline, _ = evaluar(baseline, x[test], y[test])
    ruta_baseline = salida / 'modelos' / 'baseline_rf_agrupado.joblib'
    ruta_baseline.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(baseline, ruta_baseline)

    mejor_busqueda = modelos[seleccionado]
    inicio = time.perf_counter()
    calibracion = None
    if seleccionado == 'svc_rbf':
        candidato = clone(mejor_busqueda).fit(x[train], y[train])
        calibrado = CalibratedClassifierCV(candidato, method='sigmoid', cv='prefit')
        calibrado.fit(x[val], y[val])
        candidato = calibrado
        calibracion = 'sigmoid, estimador entrenado en training y calibrado en validation'
    else:
        candidato = clone(mejor_busqueda).fit(x[desarrollo], y[desarrollo])
    tiempo_candidato = time.perf_counter() - inicio
    met_candidato, _ = evaluar(candidato, x[test], y[test])
    ruta_candidato = salida / 'modelos' / f'candidato_{seleccionado}_{configuracion}.joblib'
    joblib.dump(candidato, ruta_candidato)

    mejora_valida = (
        met_candidato['f1_macro'] >= met_baseline['f1_macro'] + 0.01
        and met_candidato['recall_prioritario_promedio'] >= met_baseline['recall_prioritario_promedio']
        and met_candidato['defectos_clasificados_sin_defecto'] <= met_baseline['defectos_clasificados_sin_defecto']
    )
    bundle = {
        'formato': 'vision360_model_bundle_v1', 'modelo': candidato,
        'configuracion_caracteristicas': configuracion, 'classes_': [0, 1, 2, 3],
        'tamano_entrada': 64 if configuracion == 'hog64' else 128,
        'criterio_ambiguedad': 0.10,
        'metadata': {'modelo': seleccionado, 'semilla': SEMILLA, 'calibracion': calibracion},
    }
    ruta_bundle = salida / 'modelos' / f'bundle_candidato_{seleccionado}_{configuracion}.joblib'
    joblib.dump(bundle, ruta_bundle)
    resultado = {
        'test_abierto_una_vez_despues_de_seleccion': True,
        'modelo_seleccionado_en_validacion': seleccionado,
        'configuracion_caracteristicas': configuracion,
        'criterio_reemplazo': 'F1 macro +0.01, recall prioritario no menor y defectos como sin defecto no mayor, todo contra baseline agrupado',
        'cumple_criterio_reemplazo': mejora_valida,
        'baseline_rf_agrupado': {
            'entrenamiento_segundos': tiempo_baseline, 'metricas_test': met_baseline,
            'archivo': str(ruta_baseline), 'tamano_bytes': ruta_baseline.stat().st_size,
        },
        'candidato': {
            'entrenamiento_segundos': tiempo_candidato, 'calibracion': calibracion,
            'metricas_test': met_candidato, 'archivo': str(ruta_candidato),
            'bundle': str(ruta_bundle), 'tamano_bytes': ruta_candidato.stat().st_size,
        },
    }
    guardar_json(ruta_resultado, resultado)
    return resultado


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-root', required=True, type=Path, help='muestra_original con training/validation/testing')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'evidencias/experimentos_cpu')
    args = parser.parse_args()
    salida = args.output_dir.resolve()
    salida.mkdir(parents=True, exist_ok=True)
    total_ram, libre_ram = memoria_sistema()
    nucleos_logicos = os.cpu_count() or 1
    trabajos_cv = 1 if libre_ram < 4 * 2**30 else min(2, max(1, nucleos_logicos // 4))
    hilos_arbol = min(2, max(1, nucleos_logicos // 2))
    cache_svc_mb = 256 if libre_ram < 2 * 2**30 else 512
    recursos = {
        'fecha_inicio_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'cpu_logicos': nucleos_logicos, 'ram_total_bytes': total_ram,
        'ram_disponible_inicio_bytes': libre_ram, 'trabajos_cv': trabajos_cv,
        'hilos_internos_arboles': hilos_arbol, 'cache_svc_mb': cache_svc_mb,
        'gpu_utilizada': False, 'cuda_utilizado': False,
        'python': platform.python_version(), 'plataforma': platform.platform(),
    }
    guardar_json(salida / 'recursos_equipo.json', recursos)
    print(json.dumps(recursos, indent=2), flush=True)

    ruta_inventario = salida / 'inventario_agrupado.csv'
    ruta_auditoria = salida / 'auditoria_dataset.json'
    if ruta_inventario.is_file() and ruta_auditoria.is_file():
        with ruta_inventario.open(encoding='utf-8-sig', newline='') as archivo:
            filas = list(csv.DictReader(archivo))
        for fila in filas:
            fila['etiqueta'] = int(fila['etiqueta'])
            fila['bytes'] = int(fila['bytes'])
        auditoria = json.loads(ruta_auditoria.read_text(encoding='utf-8'))
        print('Inventario y particiones reanudados desde evidencia.', flush=True)
    else:
        filas = inventariar(args.dataset_root.resolve())
        auditoria_grupos = conectar_grupos(filas)
        particiones = construir_particiones(filas)
        auditoria = {'dataset': auditoria_grupos, 'particiones': particiones}
        escribir_csv(ruta_inventario, filas)
        guardar_json(ruta_auditoria, auditoria)
    print(json.dumps(auditoria, ensure_ascii=False, indent=2), flush=True)

    matrices, registro_features = obtener_caracteristicas(filas, salida)
    guardar_json(salida / 'configuraciones_caracteristicas.json', registro_features)
    seleccion_feature, _ = comparar_caracteristicas(matrices, filas, salida, hilos_arbol)
    print(f'Características seleccionadas antes de búsquedas: {seleccion_feature}', flush=True)
    seleccionado, modelos, _ = buscar_modelos(
        matrices[seleccion_feature], filas, salida, hilos_arbol, trabajos_cv, cache_svc_mb,
    )
    print(f'Modelo seleccionado en validation antes de abrir test: {seleccionado}', flush=True)
    final = evaluacion_final(
        seleccionado, modelos, matrices[seleccion_feature], filas, salida,
        hilos_arbol, seleccion_feature,
    )
    recursos['fecha_fin_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    recursos['memoria_proceso_pico_bytes'] = memoria_proceso_pico()
    guardar_json(salida / 'recursos_equipo.json', recursos)
    guardar_json(salida / 'RESUMEN_EJECUCION.json', {
        'caracteristicas_seleccionadas': seleccion_feature,
        'modelo_seleccionado_validacion': seleccionado,
        'evaluacion_final': final,
        'nota_mlflow': 'No se instaló MLflow para evitar dependencias y consumo adicionales; los experimentos se registraron en JSON, CSV y Joblib reanudables.',
    })
    print(json.dumps(final, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
