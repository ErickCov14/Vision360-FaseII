# VISION 360 · FASE II | CLASIFICACIÓN ACADÉMICA DE SOLDADURAS

Aplicación web contenerizada para clasificar imágenes radiográficas del dataset público **RIAWELC** en cuatro categorías: grieta, porosidad, falta de penetración y sin defecto. Implementación para Gestión de Proyectos de Inteligencia Artificial, Fase II.

> **Alcance:** Vision 360 es el proyecto industrial principal, pero sus imágenes y parámetros son confidenciales. Esta entrega demuestra el despliegue de un modelo independiente previamente entrenado con radiografías públicas en la Actividad 5. No se valida el desempeño con panorámicas ópticas de la cámara 360, y la etiqueta «Sin defecto» no debe utilizarse para liberar piezas.

## DIAGNÓSTICO Y CORRECCIÓN DE LA INFERENCIA

La auditoría del 31 de octubre de 2026 comparó este backend con el commit `8d5e9ec6150446b539ebf07cd3e8ab1de3c64eed` de la Actividad 5 y evaluó el artefacto final sobre las 400 imágenes originales de `testing` (100 por clase).

No se encontró una diferencia entre el preprocesamiento de entrenamiento y el de inferencia: ambos aplican corrección EXIF, escala de grises, LANCZOS a 64 × 64, `uint8`, división entre 255 y HOG con los mismos parámetros. El artefacto es un `Pipeline` con `RandomForestClassifier`, acepta 324 características, contiene las clases `[0, 1, 2, 3]` y `predict` coincide con el máximo de `predict_proba`.

La repetición observada de «Porosidad» se reprodujo con las imágenes manuales: los ejemplos de grieta, porosidad y falta de penetración seleccionados previamente producen la clase 1; dos de ellos tienen márgenes muy pequeños. Sin embargo, en las 400 imágenes el modelo no queda fijo en esa clase: genera 71 predicciones de grieta, 112 de porosidad, 107 de falta de penetración y 110 de sin defecto. La causa confirmada es una combinación de errores propios del clasificador y una selección manual de ejemplos poco representativa, no un intercambio de etiquetas ni una alteración de probabilidades.

Se incorporó el estado **RESULTADO NO CONCLUYENTE** cuando la diferencia entre las dos probabilidades mayores es menor a `0.10`. Este indicador no cambia la clase predicha ni recalibra las probabilidades. En la prueba completa, 142 de 400 resultados cumplen ese criterio.

## MODELO CPU OPTIMIZADO

Se creó un protocolo nuevo agrupado por radiografía y hash exacto para eliminar contaminación entre particiones. La nueva división contiene 1,224 imágenes de entrenamiento, 428 de validación y 348 de test, con cero grupos y cero hashes compartidos.

Se compararon HOG64, HOG128, LBP128 y HOG+LBP, además de Random Forest, Extra Trees, SVM RBF e HistGradientBoosting. HOG64 y un Random Forest de 500 árboles fueron seleccionados exclusivamente mediante training/validation. Frente a un Random Forest baseline reentrenado bajo el mismo protocolo, el candidato elevó F1 macro agrupado de 0.6468 a 0.6670 y redujo defectos clasificados como «sin defecto» de 29 a 27. El recall de falta de penetración aumentó de 0.5976 a 0.6220; el de grieta bajó de 0.6105 a 0.6000.

El modelo optimizado se ejecuta solo con CPU y está activo en Docker como `random_forest_vision360_cpu_optimizado.joblib`. El modelo original se conserva con su nombre anterior. Véase `evidencias/experimentos_cpu/REPORTE_OPTIMIZACION_CPU.md` para el protocolo, parámetros, matrices y limitaciones.

### RESULTADOS FINALES DEL MODELO OPTIMIZADO

Se realizó una nueva evaluación utilizando particiones agrupadas por radiografía para reducir la contaminación entre los conjuntos de datos.

Se identificaron 395 grupos y se utilizaron 2,000 imágenes distribuidas en 1,224 para entrenamiento, 428 para validación y 348 para prueba.

No se encontraron grupos ni hashes compartidos entre particiones.

| Métrica | Random Forest baseline | Random Forest optimizado |
|---|---:|---:|
| Accuracy | 0.6437 | 0.6638 |
| F1 macro | 0.6468 | 0.6670 |
| Recall grieta | 0.6105 | 0.6000 |
| Recall falta de penetración | 0.5976 | 0.6220 |
| Defectos clasificados como sin defecto | 29 | 27 |

El modelo seleccionado utiliza 500 árboles, profundidad máxima de 20, min_samples_leaf de 2, max_features log2 y class_weight balanced_subsample.

La optimización mejoró el desempeño general y redujo los defectos clasificados como piezas sin defecto. Sin embargo, el recall de grietas disminuyó ligeramente.

Las métricas históricas obtenidas con 400 imágenes pertenecen a un protocolo anterior y no deben compararse directamente con estos resultados.

### VALIDACION DEL DESPLIEGUE

- 18 pruebas automatizadas aprobadas.
- Ninguna prueba fallida ni omitida.
- Backend FastAPI funcionando correctamente.
- Frontend React disponible en localhost:8080.
- Contenedores Docker operativos.
- Modelo optimizado cargado correctamente.
- Consistencia entre inferencia directa y API HTTP.
- Ejecución mediante CPU, sin requerir GPU.
## REQUISITOS

- Docker Desktop con Compose v2 y conexión a Internet **durante la primera construcción**.
- Al menos 4 GB RAM disponible y 3 GB libres en disco recomendados.
- Windows 10/11, macOS o Linux con Docker Desktop o Docker Engine + Compose.
- Puerto 8080 libre (o modificar `compose.yaml`).

## INICIO RÁPIDO (WINDOWS, MAC, LINUX)

```bash
docker compose up --build -d
```

Abrir **http://localhost:8080** y cargar una imagen de soldadura (PNG, JPEG, BMP o TIFF, hasta 8 MB). La documentación interactiva está en **http://localhost:8080/api/docs**.

Para obtener cuatro **imágenes de ejemplo públicas** con las categorías de la Actividad 5, ejecuta `python scripts/obtener_ejemplos.py` y luego carga un PNG de la carpeta `ejemplos/`. Este paso es opcional y requiere Internet.

Para comprobar la ejecución:

```bash
docker compose ps
docker compose logs --tail=100 backend
docker compose logs --tail=100 frontend
```

Después de iniciarlos, ejecutar la prueba HTTP de extremo a extremo (utiliza imagen sintética solo para comunicación):

```bash
python scripts/verificar_despliegue.py
```

La evidencia se guarda en `evidencias/prueba_http_real.json`. Esta prueba requiere que el contenedor real esté funcionando.

Para detener:

```bash
docker compose down
```

Windows también incluye `COMANDOS_WINDOWS.bat`. El archivo no instala Docker.

## ¿DE DÓNDE SALE EL MODELO?

El artefacto **original** `random_forest_vision360.joblib` (10,132,306 bytes) procede de nuestro repositorio de la Actividad 5:

https://github.com/ErickCov14/Actividad5-Vision360/tree/main/resultados

El repositorio contiene el modelo entrenado, el notebook y el preprocesamiento. Para evitar duplicar un archivo binario grande, este proyecto incluye `scripts/obtener_modelo.py`, que **descarga el archivo original en la construcción del backend**, verifica su longitud y su SHA-1 *de blob Git* y solo entonces lo carga. Si la descarga está bloqueada, el build termina con error; no se sustituye silenciosamente por un modelo inventado.

La validación de Git comprueba integridad frente al commit revisado, no equivale a una auditoría de seguridad criptográfica del artefacto. Los archivos Joblib solo deben cargarse de fuentes confiables.

## ARQUITECTURA

```text
Navegador React (Vite compilado) 
          |
          | HTTP :8080
          v
Nginx [frontend] -- /api/* --> FastAPI [backend]
                                   |
                                   v
                            PIL + HOG (324)
                                   |
                                   v
                          RandomForest / Joblib
```

El frontend envía un archivo multipart/form-data a `POST /api/predict`. El backend corrige orientación EXIF, convierte a escala de grises, redimensiona a 64x64 con LANCZOS, divide intensidades entre 255 y extrae **324 características HOG** con 9 orientaciones, celda 16x16, bloque 2x2 y norma L2-Hys. Estas operaciones están alineadas con `fuentes/datos_prep.py` y `fuentes/train.py` de la Actividad 5.

### API

| Método | Ruta | Respuesta |
|---|---|---|
| GET | `/api/health` | Estado del servicio y modelo |
| GET | `/api/model` | Clases y parámetros de entrada |
| POST | `/api/predict` | Clasificación, probabilidades estimadas y advertencia |
| GET | `/api/docs` | Interfaz Swagger/OpenAPI |

Ejemplo desde Windows PowerShell:

```powershell
curl.exe -X POST http://localhost:8080/api/predict -F "file=@C:\ruta\radiografia.png"
```

Devolverá `clase`, `clase_id`, `confianza_estimada`, `probabilidades`, `margen_probabilidades`, `ambigua`, `estado_analisis`, `resultado`, `modelo` y `advertencia`. No hay entrenamiento al iniciar el contenedor ni se almacenan imágenes subidas.

## EJECUCIÓN LOCAL SIN DOCKER (OPCIONAL)

Requiere Python 3.11 y Node.js 22, además de conexión para descargar dependencias y modelo:

```bash
python -m venv .venv
# Activar el entorno virtual según el sistema
pip install -r backend/requirements-test.txt
python scripts/obtener_modelo.py
```

En Linux/macOS:

```bash
export MODEL_PATH="$(pwd)/backend/model/random_forest_vision360_cpu_optimizado.joblib"
PYTHONPATH=backend uvicorn app.main:app --host 0.0.0.0 --port 8000
```

En PowerShell:

```powershell
$env:MODEL_PATH=(Resolve-Path backend/model/random_forest_vision360_cpu_optimizado.joblib).Path
$env:PYTHONPATH="backend"
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

En otra terminal:

```bash
cd frontend
npm install
npm run dev
```

Abrir http://localhost:5173. Vite redirige `/api` al puerto 8000.

## PRUEBAS

```bash
# Con entorno Python local y dependencias instaladas:
PYTHONPATH=backend pytest backend/tests -v
```

En PowerShell: `$env:PYTHONPATH="backend"; pytest backend/tests -v`.

`test_api.py` emplea un modelo determinista exclusivamente para los contratos HTTP y los casos de error. `test_modelo_real.py` inspecciona el artefacto verdadero. `test_integracion_real.py` procesa cuatro imágenes públicas, una por carpeta de origen, y compara el vector HOG, la predicción y las probabilidades entre la implementación de referencia y FastAPI. La ejecución documentada produjo **18 pruebas aprobadas, 0 omitidas y 0 fallidas**.

Para repetir la evaluación de 400 imágenes, obtener el repositorio público original y ejecutar:

```powershell
python scripts/evaluar_riawelc.py --dataset-root "C:\ruta\Actividad5-Vision360\datos\datos_ini\muestra_original\testing"
```

El evaluador no entrena ni ajusta el modelo. Genera CSV/JSON, matriz de confusión SVG, ejemplos de aciertos/errores, distribución de probabilidades, versiones y auditoría de particiones en `evidencias/evaluacion_riawelc/`.

Con Docker activo, la equivalencia contra HTTP real se repite con:

```powershell
python scripts/verificar_despliegue.py
python scripts/comparar_modelo_api.py
```

## REPRODUCIBILIDAD Y LIMITACIONES

- Modelo activo: Random Forest optimizado con CPU y HOG64. El modelo original de la Actividad 5 se conserva como referencia.
- Pipeline previamente guardado como Joblib. Mantener scikit-learn 1.6.1, scikit-image 0.25.2 y dependencias fijadas.
- Las métricas históricas de 0.6534 F1 macro en validación y 0.6627 F1 macro en prueba correspondieron al modelo **anterior al reentrenamiento final** de 1,600 imágenes. No son métricas verificadas del artefacto final.
- No hay garantías de calibración de probabilidades. El margen de ambigüedad es una regla de presentación documentada, no un umbral de aceptación industrial.
- La auditoría por SHA-256 encontró 30 grupos de archivos exactamente duplicados entre particiones y 248 grupos de recortes relacionados (mismo nombre base sin coordenadas) presentes en más de una partición. Por ello, la independencia del conjunto de prueba no está garantizada y sus métricas pueden ser optimistas.
- Las cuatro etiquetas se toman de las carpetas públicas de RIAWELC; no hubo revisión independiente por un especialista en soldadura.
- El origen de la imagen se verifica por el contenido con PIL; límite de 8 MB y 18 MP para reducir riesgos.
- El servicio es académico, sin autenticación, sin TLS y sin persistencia. No exponerse directamente a Internet. Para la nube: autenticación, HTTPS, secretos, límites y monitoreo.

## ESTRUCTURA

```text
backend/app             FastAPI y procesamiento
backend/tests           Pytest, pruebas de contratos y artefacto real
backend/Dockerfile      Servicio Python con modelo validado
frontend/src            React + CSS
frontend/Dockerfile     Vite (build) y Nginx (runtime)
frontend/default.conf.template     Proxy de API configurable por API_UPSTREAM
scripts/obtener_modelo.py  Descarga y validación del artefacto
scripts/obtener_ejemplos.py  Descarga opcional de cuatro radiografías públicas
scripts/verificar_despliegue.py  Verificación HTTP end-to-end
scripts/comparar_modelo_api.py  Comparación directa contra la API real
scripts/evaluar_riawelc.py  Evaluación reproducible de las 400 imágenes
compose.yaml            Orquestación local
manuales/               Manual de despliegue y validación (Word)
evidencias/              Evidencias de ejecución de pruebas disponibles
```

## DESPLIEGUE EN NUBE (PROPUESTA, NO EJECUTADO)

Azure Container Apps: almacenar imágenes Docker en un registro privado (ACR), desplegar frontend/backend, configurar `API_UPSTREAM=vision360-api`, comunicación interna y HTTPS para el frontend, ajustar límites de carga, configurar revisión y pruebas. Véase `manuales/01_Manual_Despliegue_Vision360.docx` para los pasos y diferencias entre validación local y propuesta de Azure.

## FUENTES

- Totino, B., Spagnolo, F., y Perri, S. (2023). RIAWELC. https://doi.org/10.53375/ijecer.2023.320
- Repositorio del conjunto: https://github.com/stefyste/RIAWELC
- Código original de entrenamiento: https://github.com/ErickCov14/Actividad5-Vision360
- Docker: https://docs.docker.com/compose/
- FastAPI: https://fastapi.tiangolo.com/
