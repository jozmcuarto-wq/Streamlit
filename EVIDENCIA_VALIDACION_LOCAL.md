# Evidencia de validación local

Ejecución desde `ml_streamlit_project` con las dependencias listadas en `requirements.txt`.

## Entrenamiento

Comando: `python train.py`

- Dataset: 1.599 filas; 1.279 entrenamiento y 320 prueba.
- Métricas en prueba: accuracy 0,813; precisión de la clase alta 0,400; recall de la clase alta 0,791; F1 de la clase alta 0,531; ROC-AUC 0,882.
- Línea base mayoritaria: accuracy 0,866; F1 de clase alta 0,000.
- Matriz de confusión (habitual/alta): `[[226, 51], [9, 34]]`.
- Artefacto generado: `model.joblib`.

## Inferencia de ejemplo

Primera fila de `data/winequality-red.csv`: calidad registrada 5; el modelo clasifica como **Calidad habitual** y devuelve `P(alta)=0,0369`.

## Interfaz

`AppTest` cargó la app sin excepciones y ejecutó la predicción del botón. Se probaron valores medianos (P(alta) 20,6%), primera fila del dataset (3,7%) y valores mínimos (17,3%) y máximos (39,6%) de las columnas. En todos los casos respondió con una clase y probabilidad. GCP y la prueba desde otra pareja siguen pendientes según el checklist.

## Docker local

- Docker Desktop instalado para el usuario; Docker CLI 29.9.3 y motor 29.9.3 responden.
- `docker build -t ml-streamlit-vino .`: imagen construida correctamente con `python:3.12-slim`.
- `docker run -d --name ml-streamlit-vino --restart unless-stopped -p 8501:8501 ml-streamlit-vino`: contenedor `ml-streamlit-vino` activo.
- Los logs muestran Streamlit escuchando en el puerto 8501; `http://127.0.0.1:8501` respondió HTTP 200.
- Desde el contenedor, Python 3.12 cargó `model.joblib`, predijo la clase 0 y calculó P(alta)=0,2061 para las entradas medianas.
- Este contenedor corre en el computador local; el build y el despliegue en la VM de GCP siguen pendientes.
