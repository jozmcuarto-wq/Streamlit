# Plan y checklist de la guía del profesor

Proyecto elegido: clasificación supervisada sobre el dataset público UCI Wine Quality (vino tinto), usando `quality >= 7` como clase alta. Este checklist sigue el flujo de la guía, en el mismo orden.

| # | Tarea de la guía | Estado | Evidencia / pendiente |
|---:|---|---|---|
| 1 | Seleccionar o cargar dataset | ✅ Hecho | `data/winequality-red.csv`, UCI, 1.599 filas. |
| 2 | Explorar y definir variables | ✅ Hecho | 11 medidas fisicoquímicas; etiqueta positiva definida como `quality >= 7`. |
| 3 | Crear `train.py` | ✅ Hecho | `train.py` valida el CSV y define el pipeline reproducible. |
| 4 | Entrenar y evaluar | ✅ Hecho | Matriz de confusión, reporte, F1/recall/ROC-AUC y línea base mayoritaria. |
| 5 | Guardar `model.joblib` | ✅ Hecho | Pipeline y metadatos generados en `model.joblib`. |
| 6 | Crear `app.py` | ✅ Hecho | Interfaz Streamlit con entradas numéricas e interpretación. |
| 7 | Probar Streamlit en Colab | 🟡 Verificado localmente | Streamlit arrancó en local y respondió HTTP 200; Colab/túnel temporal no se configuró. |
| 8 | Crear requirements, Dockerfile y .dockerignore | ✅ Hecho | Archivos presentes con versiones fijadas. |
| 9 | Guardar proyecto en GitHub | ⏳ Pendiente externo | La estudiante no tiene cuenta; una persona de la pareja debe crearla y subir el ZIP por la web. No se necesita Git local. |
| 10 | Crear VM en GCP | ⏳ Pendiente externo | Requiere proyecto/cuenta Google Cloud y revisar facturación. |
| 11 | Instalar Docker en la VM | ⏳ Pendiente | Se ejecuta al tener la VM; comandos en README. |
| 12 | Clonar desde GitHub | ⏳ Pendiente | Requiere publicar el repo primero. |
| 13 | Construir imagen Docker | ✅ Hecho localmente | Imagen `ml-streamlit-vino` construida con `python:3.12-slim` y dependencias fijadas. |
| 14 | Ejecutar contenedor | ✅ Hecho localmente | Contenedor activo en 8501; logs muestran Streamlit iniciado y HTTP local respondió 200. |
| 15 | Abrir firewall | ⏳ Pendiente externo | Requiere permisos en el proyecto GCP; TCP 8501 y etiqueta `streamlit-server`. |
| 16 | Probar IP pública | ⏳ Pendiente externo | Requiere VM activa, contenedor y regla de firewall. |
| 17 | Prueba cruzada con otra pareja | ⏳ Pendiente humano | Requiere otra pareja y repositorio publicado; registrar en `EVIDENCIA_PRUEBA_CRUZADA.md`. |

## Revisión del dataset

- **Qué representa cada fila:** una muestra de vino tinto Vinho Verde.
- **Objetivo:** clasificación binaria; calidad alta si `quality >= 7`, habitual si es menor.
- **Datos usados:** las 11 mediciones fisicoquímicas descritas por UCI.
- **Faltantes:** el CSV incluido no tiene valores vacíos en las columnas del modelo; se deja imputación por mediana en el pipeline para que el preprocesamiento quede dentro del artefacto.
- **Tipos:** las entradas y el objetivo son numéricos.
- **Fuga de información:** `quality` se excluye explícitamente de `X`; no se usan identificadores.

## Estado de validación local

- [x] Dataset cargado y esquema validado por `train.py`.
- [x] Entrenamiento y métricas de clasificación ejecutados: F1 alta 0,531; recall alta 0,791; ROC-AUC 0,882; baseline F1 0,000.
- [x] `model.joblib` se genera y puede recargarse para inferencia.
- [x] App Streamlit arrancó en local y respondió HTTP 200.
- [x] App probada con valores medianos, una fila real y valores mínimos/máximos.
- [x] Imagen Docker construida; contenedor activo, logs limpios y modelo cargado dentro de Python 3.12.
- [ ] Despliegue GCP, acceso por IP externa y control de recursos.
- [ ] Otra pareja reproduce el repo y registra evidencia cruzada.
- [ ] Evidencia firmada por otra pareja.

