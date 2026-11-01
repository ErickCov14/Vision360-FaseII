# Vision 360 Fase II — reporte de diagnóstico y validación

Fecha de ejecución: 31 de octubre de 2026. Todas las cifras de este documento provienen de los artefactos de `evidencias/` generados durante esta ejecución.

## A. Diagnóstico

Se comparó el código de inferencia con `fuentes/datos_prep.py` y `fuentes/train.py` del commit `8d5e9ec6150446b539ebf07cd3e8ab1de3c64eed` de Actividad 5. El vector producido por ambos procedimientos coincide exactamente (`máxima diferencia = 0.0`) en las cuatro imágenes reales comparadas.

Problemas confirmados:

- La selección manual anterior contiene dos errores del modelo: la imagen etiquetada como grieta y la etiquetada como falta de penetración se predicen como porosidad. La imagen de porosidad sí se predice como porosidad y la imagen sin defecto como sin defecto.
- Dos de esos resultados erróneos son ambiguos: los márgenes entre las dos probabilidades principales son 0.0750 y 0.0316.
- El clasificador tiene desempeño limitado: 268/400 aciertos, con recall especialmente bajo para grieta (0.53).
- El muestreo original no garantiza independencia: se detectaron 30 hashes con contenido idéntico entre particiones y 248 grupos de recortes relacionados presentes en más de una partición.

Hipótesis descartadas mediante pruebas:

- No hay diferencias en EXIF, conversión a gris, redimensionamiento, tipo `uint8`, normalización o HOG.
- `model.classes_` es `[0, 1, 2, 3]` y cada probabilidad se relaciona con ese ID real.
- El modelo es un Pipeline con Random Forest, acepta 324 características y se carga con las versiones fijadas.
- `predict`, `argmax(predict_proba)`, la integración FastAPI y el endpoint HTTP real coinciden.
- No existe una salida fija de porosidad: solo 112/400 (28 %) fueron asignadas a esa clase.

No se puede determinar con esta evidencia si las etiquetas públicas son clínicamente correctas ni cuánto influyen los recortes relacionados en la estimación de desempeño. Eso requiere reconstruir particiones agrupadas por radiografía y, preferentemente, revisión experta.

## B. Archivos modificados o creados

| Ruta | Acción | Motivo |
|---|---|---|
| `backend/app/inferencia.py` | Modificar | Verificar `predict` contra `argmax`, calcular margen y estado no concluyente sin alterar la clase. |
| `backend/app/main.py` | Modificar | Documentar por API el criterio de ambigüedad y versión 1.1.0. |
| `frontend/src/main.jsx` | Modificar | Mostrar clase, estado, margen, probabilidades, modelo y advertencia de no calibración. |
| `frontend/src/style.css` | Modificar | Estilos mínimos para estado y advertencia de ambigüedad. |
| `backend/tests/conftest.py` | Modificar | Detectar automáticamente el artefacto local real. |
| `backend/tests/test_api.py` | Modificar | Validar los nuevos campos y el criterio documentado. |
| `backend/tests/test_modelo_real.py` | Modificar | Verificar tipo, clases, dimensiones y parámetros del artefacto. |
| `backend/tests/test_integracion_real.py` | Crear | Comparar procedimiento original y FastAPI con una imagen por categoría. |
| `scripts/evaluar_riawelc.py` | Crear | Evaluación completa reproducible y generación de evidencias. |
| `scripts/comparar_modelo_api.py` | Crear | Comparación directa frente al HTTP real de Docker. |
| `scripts/verificar_despliegue.py` | Modificar | Añadir cuatro imágenes reales al smoke test HTTP. |
| `README.md` | Modificar | Documentar diagnóstico, resultados, límites y reproducción. |
| `evidencias/evaluacion_riawelc/*` | Crear | Métricas, predicciones, matriz, ejemplos, versiones y auditoría. |
| `evidencias/respaldo_previo/*` | Crear | Copia recuperable previa; la carpeta original no tenía `.git`. |

No se eliminó ningún archivo ni se sustituyó el modelo.

## C. Validación RIAWELC

- Imágenes: 400 (100 por clase).
- Aciertos/errores: 268/132.
- Accuracy: 0.6700.
- F1 macro: 0.6675669554.
- Predicciones: grieta 71; porosidad 112; falta de penetración 107; sin defecto 110.
- Porosidad: 28.0 %.

| Clase | Precision | Recall | F1 |
|---|---:|---:|---:|
| Grieta | 0.7465 | 0.5300 | 0.6199 |
| Porosidad | 0.5893 | 0.6600 | 0.6226 |
| Falta de penetración | 0.5888 | 0.6300 | 0.6087 |
| Sin defecto | 0.7818 | 0.8600 | 0.8190 |

Matriz de confusión, filas reales y columnas predichas en orden grieta, porosidad, falta de penetración y sin defecto:

```text
[[53, 18, 27,  2],
 [ 9, 66, 14, 11],
 [ 9, 17, 63, 11],
 [ 0, 11,  3, 86]]
```

## D. Comparación modelo / API

Cuatro imágenes públicas fueron procesadas mediante referencia independiente, backend y HTTP real. En las cuatro:

- vector de 324 características: máxima diferencia 0.0;
- predicción e ID: iguales;
- nombre: asociado al ID correcto;
- máxima diferencia de probabilidades: menor a `5e-7`, explicada por redondeo a seis decimales en JSON.

No existe una medición anterior comparable de la aplicación que demuestre mejora de accuracy. El 0.6700 medido corresponde al artefacto final; el 0.6675 histórico del README original pertenecía al modelo anterior al reentrenamiento y no debe interpretarse como comparación antes/después.

## E. Docker

Se ejecutó `docker compose build --no-cache`, seguido de recreación forzada. Backend: healthy. Frontend: puerto 8080. Se verificaron con HTTP real `GET /`, `GET /api/health`, `GET /api/model` y cinco solicitudes `POST /api/predict` (una sintética de transporte y cuatro RIAWELC), todas con código 200.

## F. Resultado final y limitaciones

La aplicación funciona y la inferencia coincide con el modelo original. Se corrigió la presentación de resultados ambiguos y se añadió validación reproducible; no se forzaron clases ni probabilidades. El problema restante es el desempeño del modelo y la falta de independencia demostrada entre particiones. Un reentrenamiento futuro debe reagrupar por radiografía de origen, separar un test verdaderamente independiente y usar solo entrenamiento/validación para selección. El sistema sigue siendo una demostración académica con radiografías públicas, no una validación para imágenes ópticas reales de Vision 360.
