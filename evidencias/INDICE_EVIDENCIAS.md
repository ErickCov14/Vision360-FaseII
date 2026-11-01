# Índice de evidencias actualizadas

Ejecución: 31 de octubre de 2026.

| Evidencia | Contenido |
|---|---|
| `REPORTE_DIAGNOSTICO.md` | Diagnóstico, archivos cambiados, resultados, Docker y limitaciones. |
| `resultados_pytest_actualizados.xml` | Reporte JUnit: 17 aprobadas, 0 fallidas, 0 omitidas. |
| `pruebas_locales_actualizadas.txt` | Salida legible de Pytest. |
| `comparacion_modelo_api.json` | Cuatro comparaciones modelo directo / backend / HTTP real. |
| `prueba_http_docker_actualizada.json` | Salud, metadatos y cinco predicciones por Docker. |
| `estado_docker_actualizado.txt` | Versión, contenedores y registros de ejecución. |
| `evaluacion_riawelc/resumen_metricas.json` | Métricas globales y por clase sobre 400 imágenes. |
| `evaluacion_riawelc/predicciones_400.csv` | Predicción, probabilidades, margen y acierto por imagen. |
| `evaluacion_riawelc/metricas_por_categoria.csv` | Precision, recall y F1 por categoría. |
| `evaluacion_riawelc/matriz_confusion.csv` | Matriz numérica. |
| `evaluacion_riawelc/matriz_confusion.svg` | Matriz visual editable. |
| `evaluacion_riawelc/matriz_confusion.png` | Matriz visual lista para el reporte académico. |
| `evaluacion_riawelc/distribucion_probabilidades.csv` | Media, dispersión y cuantiles por clase. |
| `evaluacion_riawelc/ejemplos_seleccionados.csv` | Índice de aciertos y errores copiados. |
| `evaluacion_riawelc/ejemplos/` | Imágenes reales de aciertos y errores por clase. |
| `evaluacion_riawelc/metadata_modelo_y_entorno.json` | Hash, clases, parámetros y versiones. |
| `evaluacion_riawelc/auditoria_particiones.json` | Duplicados exactos y grupos relacionados. |
| `respaldo_previo/` | Copias de los archivos existentes antes de modificarlos. |

Los archivos sin el sufijo `actualizado` que ya existían en esta carpeta corresponden a ejecuciones anteriores y no se usaron para las conclusiones del reporte nuevo.

## Optimización CPU agrupada

La carpeta `experimentos_cpu/` contiene el reporte de optimización, inventario y particiones agrupadas, cachés de características, búsquedas reanudables, modelos independientes, comparación CSV, matrices SVG, estabilidad por folds, Pytest y pruebas HTTP del modelo optimizado.
