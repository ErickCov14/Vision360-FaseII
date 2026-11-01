#!/usr/bin/env python3
# VALIDACION
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import PIL
import sklearn
import skimage
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

ETIQUETAS = {
    0: "Grieta",
    1: "Porosidad",
    2: "Falta de penetración",
    3: "Sin defecto",
}
CARPETAS = {
    "Difetto1": 0,
    "Difetto2": 1,
    "Difetto4": 2,
    "NoDifetto": 3,
}
IDS = [0, 1, 2, 3]


def escribir_csv(ruta: Path, filas: list[dict], campos: list[str]) -> None:
    with ruta.open("w", newline="", encoding="utf-8-sig") as archivo:
        writer = csv.DictWriter(archivo, fieldnames=campos)
        writer.writeheader()
        writer.writerows(filas)


def hash_archivo(ruta: Path) -> str:
    digest = hashlib.sha256()
    with ruta.open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(1024 * 1024), b""):
            digest.update(bloque)
    return digest.hexdigest()


def grupo_radiografia(nombre: str) -> str:
    return re.sub(r"_\[\d+\]\[\d+\]$", "", Path(nombre).stem)


def auditar_particiones(testing: Path) -> dict:
    base = testing.parent
    hashes: dict[str, list[tuple[str, str]]] = defaultdict(list)
    grupos: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    conteos: dict[str, int] = {}
    for particion in ("training", "validation", "testing"):
        ruta_particion = base / particion
        imagenes = sorted(ruta_particion.glob("*/*.png")) if ruta_particion.is_dir() else []
        conteos[particion] = len(imagenes)
        for imagen in imagenes:
            hashes[hash_archivo(imagen)].append((particion, str(imagen.relative_to(base))))
            grupos[grupo_radiografia(imagen.name)][particion] += 1

    duplicados = []
    for digest, ubicaciones in hashes.items():
        particiones = sorted({p for p, _ in ubicaciones})
        if len(particiones) > 1:
            duplicados.append(
                {"sha256": digest, "particiones": particiones, "archivos": [r for _, r in ubicaciones]}
            )
    grupos_cruzados = []
    for grupo, por_particion in grupos.items():
        if len(por_particion) > 1:
            grupos_cruzados.append({"grupo": grupo, "conteos": dict(sorted(por_particion.items()))})
    return {
        "conteos_imagenes": conteos,
        "duplicados_exactos_entre_particiones": duplicados,
        "cantidad_duplicados_exactos": len(duplicados),
        "grupos_relacionados_entre_particiones": sorted(grupos_cruzados, key=lambda x: x["grupo"]),
        "cantidad_grupos_relacionados": len(grupos_cruzados),
        "criterio_grupo": "nombre sin sufijo de coordenadas _[fila][col]",
    }


def svg_matriz(ruta: Path, matriz: np.ndarray) -> None:
    ancho, alto = 820, 620
    x0, y0, celda = 280, 120, 92
    maximo = max(int(matriz.max()), 1)
    partes = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{ancho}" height="{alto}" viewBox="0 0 {ancho} {alto}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#172033}.title{font-size:25px;font-weight:700}.label{font-size:14px}.value{font-size:22px;font-weight:700}</style>',
        '<text x="410" y="42" text-anchor="middle" class="title">Matriz de confusión — RIAWELC testing (n=400)</text>',
        '<text x="464" y="82" text-anchor="middle" class="label">Predicción</text>',
        '<text x="38" y="310" text-anchor="middle" class="label" transform="rotate(-90 38 310)">Clase real</text>',
    ]
    nombres = ["Grieta", "Porosidad", "Falta penetr.", "Sin defecto"]
    for i, nombre in enumerate(nombres):
        x = x0 + i * celda + celda / 2
        y = y0 + i * celda + celda / 2
        partes.append(f'<text x="{x}" y="105" text-anchor="middle" class="label">{nombre}</text>')
        partes.append(f'<text x="265" y="{y + 5}" text-anchor="end" class="label">{nombre}</text>')
    for fila in range(4):
        for columna in range(4):
            valor = int(matriz[fila, columna])
            intensidad = valor / maximo
            color = f"rgb({int(238 - 120 * intensidad)},{int(247 - 42 * intensidad)},{int(255 - 20 * intensidad)})"
            x, y = x0 + columna * celda, y0 + fila * celda
            partes.append(f'<rect x="{x}" y="{y}" width="{celda}" height="{celda}" fill="{color}" stroke="#ffffff" stroke-width="2"/>')
            partes.append(f'<text x="{x + celda / 2}" y="{y + celda / 2 + 8}" text-anchor="middle" class="value">{valor}</text>')
    partes.append('</svg>')
    ruta.write_text("\n".join(partes), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path, help="Carpeta testing con las cuatro subcarpetas")
    parser.add_argument("--model", type=Path, default=ROOT / "backend/model/random_forest_vision360.joblib")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "evidencias/evaluacion_riawelc")
    args = parser.parse_args()
    testing = args.dataset_root.resolve()
    salida = args.output_dir.resolve()
    salida.mkdir(parents=True, exist_ok=True)

    faltantes = [carpeta for carpeta in CARPETAS if not (testing / carpeta).is_dir()]
    if faltantes:
        raise SystemExit(f"Faltan carpetas de clase en {testing}: {', '.join(faltantes)}")
    if not args.model.is_file():
        raise SystemExit(f"No existe el modelo: {args.model}")

    os.environ["MODEL_PATH"] = str(args.model.resolve())
    from app import inferencia

    inferencia.cargar_modelo.cache_clear()
    modelo = inferencia.cargar_modelo()
    clases = [int(valor) for valor in modelo.classes_]
    filas: list[dict] = []
    y_real: list[int] = []
    y_pred: list[int] = []
    probabilidades: list[np.ndarray] = []

    for carpeta, clase_real in CARPETAS.items():
        imagenes = sorted((testing / carpeta).glob("*.png"))
        if len(imagenes) != 100:
            print(f"ADVERTENCIA: {carpeta} contiene {len(imagenes)} imágenes; se esperaban 100.", file=sys.stderr)
        for imagen in imagenes:
            contenido = imagen.read_bytes()
            x = inferencia.extraer_caracteristicas(inferencia.preparar_imagen(contenido))
            proba = np.asarray(modelo.predict_proba(x)[0], dtype=float)
            pred_modelo = int(modelo.predict(x)[0])
            pred_argmax = clases[int(np.argmax(proba))]
            if pred_modelo != pred_argmax:
                raise RuntimeError(f"predict y argmax(predict_proba) no coinciden para {imagen}")
            por_id = {clase: float(valor) for clase, valor in zip(clases, proba)}
            ordenadas = sorted(por_id.values(), reverse=True)
            margen = ordenadas[0] - ordenadas[1]
            fila = {
                "archivo": imagen.name,
                "carpeta": carpeta,
                "clase_real_id": clase_real,
                "clase_real": ETIQUETAS[clase_real],
                "clase_predicha_id": pred_modelo,
                "clase_predicha": ETIQUETAS[pred_modelo],
                "acierto": pred_modelo == clase_real,
                "prob_grieta": por_id[0],
                "prob_porosidad": por_id[1],
                "prob_falta_penetracion": por_id[2],
                "prob_sin_defecto": por_id[3],
                "probabilidad_maxima": ordenadas[0],
                "margen_top2": margen,
                "ambiguo_margen_menor_0_10": margen < 0.10,
            }
            filas.append(fila)
            y_real.append(clase_real)
            y_pred.append(pred_modelo)
            probabilidades.append(np.array([por_id[i] for i in IDS]))

    matriz = confusion_matrix(y_real, y_pred, labels=IDS)
    reporte = classification_report(
        y_real, y_pred, labels=IDS, target_names=[ETIQUETAS[i] for i in IDS],
        output_dict=True, zero_division=0,
    )
    predicciones = Counter(y_pred)
    total = len(y_real)
    aciertos = int(np.trace(matriz))
    probs = np.vstack(probabilidades)
    metricas_clase = []
    for clase in IDS:
        datos = reporte[ETIQUETAS[clase]]
        metricas_clase.append({
            "clase_id": clase,
            "clase": ETIQUETAS[clase],
            "precision": datos["precision"],
            "recall": datos["recall"],
            "f1": datos["f1-score"],
            "soporte": int(datos["support"]),
            "predicciones": predicciones[clase],
        })

    distribucion_probabilidades = []
    for clase in IDS:
        valores = probs[:, clase]
        distribucion_probabilidades.append({
            "clase_id": clase,
            "clase": ETIQUETAS[clase],
            "media": float(np.mean(valores)),
            "desviacion": float(np.std(valores)),
            "minimo": float(np.min(valores)),
            "q25": float(np.quantile(valores, 0.25)),
            "mediana": float(np.median(valores)),
            "q75": float(np.quantile(valores, 0.75)),
            "maximo": float(np.max(valores)),
        })

    modelo_base = modelo.steps[-1][1] if hasattr(modelo, "steps") else modelo
    metadata = {
        "fecha_utc": datetime.now(timezone.utc).isoformat(),
        "modelo_ruta": str(args.model.resolve()),
        "modelo_sha256": hash_archivo(args.model),
        "tipo_objeto": f"{type(modelo).__module__}.{type(modelo).__name__}",
        "tipo_estimador_final": f"{type(modelo_base).__module__}.{type(modelo_base).__name__}",
        "clases_modelo": clases,
        "n_features_in": int(modelo.n_features_in_),
        "parametros_estimador": modelo_base.get_params(),
        "versiones": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "scikit_image": skimage.__version__,
            "pillow": PIL.__version__,
            "joblib": joblib.__version__,
        },
    }
    resumen = {
        "imagenes_evaluadas": total,
        "aciertos": aciertos,
        "errores": total - aciertos,
        "accuracy": float(accuracy_score(y_real, y_pred)),
        "f1_macro": float(f1_score(y_real, y_pred, average="macro", zero_division=0)),
        "porcentaje_predicho_porosidad": 100.0 * predicciones[1] / total,
        "predicciones_por_categoria": {ETIQUETAS[i]: predicciones[i] for i in IDS},
        "ambiguas_margen_top2_menor_0_10": sum(bool(f["ambiguo_margen_menor_0_10"]) for f in filas),
        "criterio_ambiguedad": "diferencia entre las dos probabilidades mayores < 0.10",
        "matriz_confusion_filas_real_columnas_prediccion": matriz.tolist(),
        "metricas_por_categoria": metricas_clase,
        "distribucion_probabilidades": distribucion_probabilidades,
    }
    auditoria = auditar_particiones(testing)

    campos_pred = list(filas[0].keys())
    escribir_csv(salida / "predicciones_400.csv", filas, campos_pred)
    escribir_csv(salida / "metricas_por_categoria.csv", metricas_clase, list(metricas_clase[0].keys()))
    escribir_csv(
        salida / "distribucion_probabilidades.csv",
        distribucion_probabilidades,
        list(distribucion_probabilidades[0].keys()),
    )
    matriz_filas = []
    for i, real in enumerate(IDS):
        matriz_filas.append({"clase_real": ETIQUETAS[real], **{ETIQUETAS[p]: int(matriz[i, p]) for p in IDS}})
    escribir_csv(salida / "matriz_confusion.csv", matriz_filas, ["clase_real"] + [ETIQUETAS[i] for i in IDS])
    (salida / "resumen_metricas.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8")
    (salida / "metadata_modelo_y_entorno.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (salida / "auditoria_particiones.json").write_text(json.dumps(auditoria, ensure_ascii=False, indent=2), encoding="utf-8")
    svg_matriz(salida / "matriz_confusion.svg", matriz)

    ejemplos = salida / "ejemplos"
    (ejemplos / "aciertos").mkdir(parents=True, exist_ok=True)
    (ejemplos / "errores").mkdir(parents=True, exist_ok=True)
    por_real: dict[int, list[dict]] = defaultdict(list)
    errores_por_real: dict[int, list[dict]] = defaultdict(list)
    for fila in filas:
        (por_real if fila["acierto"] else errores_por_real)[int(fila["clase_real_id"])].append(fila)
    indice_ejemplos = []
    for clase in IDS:
        for tipo, candidatos in (("acierto", por_real[clase]), ("error", errores_por_real[clase])):
            if not candidatos:
                continue
            elegido = sorted(candidatos, key=lambda f: float(f["probabilidad_maxima"]), reverse=True)[0]
            origen = testing / str(elegido["carpeta"]) / str(elegido["archivo"])
            destino_dir = ejemplos / ("aciertos" if tipo == "acierto" else "errores")
            destino = destino_dir / f"real_{clase}_pred_{elegido['clase_predicha_id']}_{origen.name}"
            shutil.copy2(origen, destino)
            indice_ejemplos.append({"tipo": tipo, **elegido, "copia": str(destino.relative_to(salida))})
    escribir_csv(salida / "ejemplos_seleccionados.csv", indice_ejemplos, list(indice_ejemplos[0].keys()))

    print(json.dumps(resumen, ensure_ascii=False, indent=2))
    print(f"Evidencias guardadas en: {salida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
