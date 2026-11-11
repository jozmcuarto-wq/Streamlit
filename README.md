# Estimación de calidad de vino tinto - Jorge Zainea_Daniel Gamba

Proyecto académico de Machine Learning con la ruta **supervisada**. Entrena un clasificador binario con un pipeline reproducible, muestra una predicción en Streamlit y permite ejecutar la aplicación con Docker.

## Objetivo y unidad de observación

Predecir si una muestra de vino tinto Vinho Verde pertenece a la categoría de **alta calidad sensorial** a partir de sus mediciones fisicoquímicas. Una fila representa una muestra; el dataset incluido contiene 1.599 muestras y 11 variables de entrada.

Los datos son de [Wine Quality, UCI Machine Learning Repository](https://doi.org/10.24432/C56S3T), descrito por Cortez et al. (2009). La licencia es CC BY 4.0; se conserva la atribución a sus autores. El archivo `data/winequality-red.csv` está incluido para que el entrenamiento no dependa de una descarga durante la ejecución.

## Ruta, variables y preparación

- **Ruta:** supervisada, clasificación binaria.
- **Objetivo modelado:** alta calidad (`quality >= 7`). La clase negativa agrupa puntuaciones menores que 7.
- **Entradas:** `fixed acidity`, `volatile acidity`, `citric acid`, `residual sugar`, `chlorides`, `free sulfur dioxide`, `total sulfur dioxide`, `density`, `pH`, `sulphates` y `alcohol`.
- **Preparación:** lectura del CSV separado por punto y coma; validación de columnas; creación de la etiqueta binaria con umbral explícito 7; separación estratificada 80/20 con `random_state=42`; imputación por mediana y estandarización dentro del pipeline.
- No se incluyen identificadores ni se hace ninguna transformación de `quality` en las entradas.

## Modelo y evaluación

Se usa `LogisticRegression` con `class_weight="balanced"`. El umbral 7 define como positiva la categoría de muestras con puntuación sensorial alta; es una decisión de este ejercicio, no una categoría original del dataset. El peso de clases ayuda a tratar el desbalance entre las dos categorías.

Al ejecutar `train.py` se imprimen matriz de confusión, reporte de clasificación, accuracy, precisión, recall, F1 y ROC-AUC, junto con una línea base que siempre predice la clase mayoritaria. Como la categoría alta es minoritaria, accuracy por sí sola puede ocultar errores; se prioriza el F1 y recall de la clase alta. Las métricas usan solo el conjunto de prueba reservado. La app presenta la clase predicha y la probabilidad estimada de la clase alta; no son evaluaciones oficiales de vino.

Resultado de esta ejecución con `random_state=42`: F1 de alta **0,531**, recall **0,791**, precisión **0,400**, ROC-AUC **0,882** y accuracy **0,813**. La línea base de clase mayoritaria logra accuracy **0,866** y F1 de alta **0,000**. La matriz de confusión (filas reales, columnas predichas; habitual/alta) es `[[226, 51], [9, 34]]`: el modelo recupera 34 de 43 casos altos, con 51 falsas alarmas. Por eso el resultado aporta detección de la clase minoritaria, pero requiere interpretar el costo de falsos positivos.

Las métricas finales se imprimen al ejecutar `python train.py` y quedan guardadas en `model.joblib`; el conjunto de prueba contiene 320 filas. Se comparan con la línea base para mostrar qué aporta el modelo a la predicción de muestras de alta calidad.

## Contenido

```text
ml_streamlit_project/
├── app.py
├── train.py
├── model.joblib                 # se genera al ejecutar train.py
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── README.md
└── data/
    └── winequality-red.csv
```

## Ejecutar localmente

Se recomienda Python 3.12, igual que la imagen Docker. Desde esta carpeta:

```bash
python -m pip install -r requirements.txt
python train.py
streamlit run app.py
```

Abre `http://localhost:8501`. La app carga `model.joblib`; no vuelve a entrenar. Los campos aparecen inicialmente con las medianas del dataset. Cambia los valores y presiona **Estimar calidad**. Si usas otras mediciones, la app avisa cuando alguna queda por fuera del rango observado.

Para entrenar desde otra ruta o guardar el modelo en otra ubicación:

```bash
python train.py --data data/winequality-red.csv --output model.joblib
```

## Ejemplo de entrada y salida

La primera fila del dataset tiene estas entradas: acidez fija 7,4; acidez volátil 0,7; ácido cítrico 0; azúcar residual 1,9; cloruros 0,076; dióxido de azufre libre 11; dióxido de azufre total 34; densidad 0,9978; pH 3,51; sulfatos 0,56; alcohol 9,4. Su calidad registrada es 5. El modelo predice **Calidad habitual** y estima probabilidad **0,0369** para la clase alta. Para reproducirlo con el artefacto:

```bash
python -c "import joblib,pandas as pd; a=joblib.load('model.joblib'); d=pd.read_csv('data/winequality-red.csv',sep=';'); x=d[a['features']].iloc[[0]]; c=int(a['model'].predict(x)[0]); p=float(a['model'].predict_proba(x)[0][list(a['model'].classes_).index(1)]); print(x.to_dict(orient='records')[0]); print(a['class_labels'][c], 'P(alta)=', round(p,4))"
```

La salida es una clase y la probabilidad estimada de pertenecer a la clase alta. Los valores pueden cambiar si se modifica el dataset, las dependencias o los parámetros; el ejemplo documenta la forma de entrada y salida, no garantiza el resultado de nuevas muestras.

## Docker

Desde esta carpeta, con Docker instalado:

```bash
docker build -t ml-streamlit-vino .
docker run --rm -p 8501:8501 ml-streamlit-vino
```

Abre `http://localhost:8501`. Para dejar el contenedor ejecutándose en segundo plano:

```bash
docker run -d --name ml-streamlit-vino --restart unless-stopped -p 8501:8501 ml-streamlit-vino
docker logs ml-streamlit-vino
```

En una instalación por usuario de Docker Desktop en Windows, si PowerShell no reconoce `docker`, añade la carpeta del CLI al `PATH` de esa sesión:

```powershell
$env:PATH = "$env:LOCALAPPDATA\Programs\DockerDesktop\resources\bin;$env:PATH"
```

La aplicación local probada para este proyecto queda disponible en `http://localhost:8501`. Para detener el contenedor: `docker stop ml-streamlit-vino`.

El `Dockerfile` instala dependencias, copia el modelo ya entrenado y ejecuta Streamlit escuchando en `0.0.0.0:8501`. En un equipo nuevo ejecuta primero `python train.py`, o conserva el `model.joblib` incluido.

## Despliegue alternativo en Streamlit Community Cloud

Usa esta opción solo si el profesor acepta una plataforma distinta de una VM de Google Cloud. Streamlit Community Cloud publica la app en una URL web, pero no crea una VM de GCP ni proporciona la IP pública de una VM.

El repositorio de GitHub debe incluir `app.py`, `model.joblib` y `requirements.txt` en la raíz. Conserva también `data/winequality-red.csv` y el resto de los archivos del proyecto para mantenerlo reproducible. No es necesario volver a entrenar el modelo ni construir la imagen Docker para este despliegue.

1. Sube el proyecto a GitHub y confirma que Streamlit Community Cloud tenga acceso al repositorio.
2. Entra a [Streamlit Community Cloud](https://share.streamlit.io/) e inicia sesión con GitHub.
3. Selecciona **Create app** y elige el repositorio, la rama `main` y el archivo `app.py`.
4. Pulsa **Deploy**. Streamlit instala las dependencias de `requirements.txt` y publica la app.
5. Abre la URL generada y pulsa **Estimar calidad** para comprobar una predicción.

**Repositorio:** `<[URL_DEL_REPOSITORIO](https://github.com/jozmcuarto-wq/Streamlit)>`  
**Aplicación publicada:** `<https://app-2ezgqvael9neqtfndvtgsq.streamlit.app/>`

La app se ejecuta en Streamlit Community Cloud, por lo que no es necesario mantener encendido el portátil. Si no recibe visitas durante 12 horas puede dormirse; cualquier visitante puede abrir el enlace para despertarla. Consulta la [documentación de administración y límites](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app).

Antes de entregar, comparte el enlace de la app con otra pareja y registra la prueba en `EVIDENCIA_PRUEBA_CRUZADA.md`.

## Reproducibilidad y autores

La semilla, la división de datos y las dependencias están fijadas. Para regenerar el artefacto ejecuta `python train.py`; para servirlo, ejecuta Streamlit o Docker. Antes de entregar, una pareja diferente debe reproducir los pasos usando solo este README y registrar el resultado en `EVIDENCIA_PRUEBA_CRUZADA.md`.

**Autores:** completar con los nombres de la pareja.

## Preguntas para la discusión final

- **Notebook frente a `train.py`:** el notebook ayuda a explorar; el script reúne el flujo final en pasos repetibles que se pueden ejecutar sin depender del orden manual de celdas.
- **Pipeline:** mantiene la imputación y el modelo juntos para que se apliquen igual durante entrenamiento y predicción.
- **Transformaciones distintas:** cambian la escala o el significado de las entradas y pueden degradar o invalidar las predicciones.
- **Artefacto frente a imagen:** `model.joblib` guarda el pipeline entrenado y sus metadatos; la imagen Docker contiene el entorno y los archivos necesarios para ejecutar la app.
- **Imagen frente a contenedor:** la imagen es la plantilla construida; el contenedor es un proceso iniciado desde esa plantilla.
- **Docker en la VM:** la guía permite desarrollar desde Colab y hacer visible la construcción/ejecución en una máquina Linux.
- **Mapeo de puertos y firewall:** `-p` conecta el puerto de la VM con el del contenedor; la regla de firewall permite que el tráfico externo llegue a la VM.
- **Ruta supervisada frente a clustering:** cambia el objetivo y la evaluación; la interfaz, Docker y el despliegue conservan la misma arquitectura general.
- **Mejoras productivas:** autenticación, HTTPS, restricción de red, monitoreo, versionado del modelo y gestión de costos.
