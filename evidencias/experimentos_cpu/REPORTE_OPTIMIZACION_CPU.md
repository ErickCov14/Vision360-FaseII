# Vision 360 — optimización CPU con particiones agrupadas

Ejecución real del 31 de octubre de 2026. No se utilizó GPU, CUDA, NVIDIA ni datos de prueba para seleccionar características, modelos o hiperparámetros.

## Recursos y protocolo

- CPU: Intel Core i5-1035G1, 4 núcleos/8 hilos.
- RAM: 11.81 GiB; disponible al inicio: 1.31 GiB.
- Ajuste conservador: 1 candidato a la vez, 2 hilos internos para árboles, caché SVC de 256 MB y matrices `float32`.
- Dataset: 2,000 recortes RIAWELC de 227×227.
- Agrupación: componente conectado por nombre de radiografía sin coordenadas o SHA-256 idéntico.
- Grupos: 395. Se confirmaron 30 hashes idénticos y 248 nombres de radiografía cruzados en las particiones antiguas.
- Nueva partición: training 1,224 imágenes/240 grupos; validation 428/79; testing 348/76.
- Verificación nueva: 0 grupos y 0 hashes compartidos entre particiones.
- Selección: `StratifiedGroupKFold` únicamente en training; comparación en validation; test abierto una sola vez después de fijar el candidato.

## Características

| Configuración | Dimensiones | Memoria (2,000) | F1 macro validation |
|---|---:|---:|---:|
| HOG 64×64 | 324 | 2.47 MiB | **0.6914** |
| HOG 128×128 | 1,764 | 13.46 MiB | 0.5732 |
| LBP 128×128 | 416 | 3.17 MiB | 0.6679 |
| HOG+LBP 128×128 | 2,180 | 16.63 MiB | 0.6734 |

HOG64 fue seleccionado antes de las búsquedas. Aumentar resolución o combinar LBP no mejoró el resultado en este protocolo.

## Comparación en validación agrupada

| Modelo | F1 macro | Recall grieta | Recall falta penetración | Defectos→sin defecto | Búsqueda CPU |
|---|---:|---:|---:|---:|---:|
| Random Forest optimizado | **0.7009** | 0.5104 | 0.7400 | **22** | 299.7 s |
| Extra Trees | 0.6824 | 0.5208 | **0.7500** | 31 | 81.6 s |
| SVM RBF | 0.6643 | **0.5938** | 0.6400 | 26 | 4.8 s |
| HistGradientBoosting | 0.6656 | 0.5625 | 0.6600 | 25 | 138.3 s |

Baseline RF fijo en la misma validación: F1 macro 0.6905, recall grieta 0.5417, recall falta de penetración 0.7400 y 25 defectos clasificados como sin defecto.

Parámetros seleccionados para RF: 500 árboles, profundidad 20, `min_samples_leaf=2`, `min_samples_split=2`, `max_features=log2` y `class_weight=balanced_subsample`.

La CV agrupada de cinco folds sobre training mostró F1 medio/desviación: baseline 0.5904±0.0247, RF optimizado 0.5932±0.0153, Extra Trees 0.5882±0.0284, SVM 0.6132±0.0333 e HistGradientBoosting 0.5985±0.0280. SVM obtuvo mayor media interna, pero no sostuvo esa ventaja en validation y fue más variable.

## Confirmación final agrupada

| Métrica | RF baseline | RF optimizado | Cambio |
|---|---:|---:|---:|
| Accuracy | 0.6437 | **0.6638** | +0.0201 |
| F1 macro | 0.6468 | **0.6670** | +0.0201 |
| Recall defectos promedio | 0.5953 | **0.6147** | +0.0194 |
| Recall grieta | **0.6105** | 0.6000 | −0.0105 |
| Recall falta penetración | 0.5976 | **0.6220** | +0.0244 |
| Defectos→sin defecto | 29 | **27** | −2 |
| Inferencia del modelo | **0.160 ms/img** | 0.342 ms/img | +0.182 ms |
| Tamaño | **10.37 MB** | 25.09 MB | +14.72 MB |

El candidato mejora F1 macro, recall agregado de defectos y errores peligrosos, a cambio de una pérdida de una grieta detectada y mayor tamaño/latencia. Cumplió el gate definido antes del test: F1 +0.01, recall prioritario promedio no menor y defectos enviados a “sin defecto” no mayores.

## Integración y validación

- Modelo nuevo: `backend/model/random_forest_vision360_cpu_optimizado.joblib`.
- Modelo original conservado: `backend/model/random_forest_vision360.joblib`.
- Docker contiene ambos y activa el optimizado mediante `MODEL_PATH`.
- API conserva IDs, nombres, probabilidades y criterio de ambigüedad.
- Comparación Python/HTTP con cuatro imágenes: aprobada.
- Pytest: 18 aprobadas, 0 fallidas, 0 omitidas.
- Backend: healthy; frontend: HTTP disponible en `localhost:8080`.

## Limitaciones

Las 2,000 imágenes ya habían sido consultadas durante el proyecto; el nuevo test es metodológicamente más independiente por radiografía, pero no es una prueba externa completamente ciega. Varias radiografías contienen recortes con etiquetas distintas. La mejora no debe compararse directamente con el 0.6676 histórico porque los protocolos son diferentes. El modelo solo ha sido validado con radiografías públicas RIAWELC, no con imágenes ópticas industriales de Vision 360.

MLflow no se instaló para evitar dependencias y presión adicional de RAM; todos los experimentos quedaron registrados en JSON, CSV, Joblib y logs reanudables.
