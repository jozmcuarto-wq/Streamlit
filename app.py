"""Interfaz Streamlit para clasificar muestras según su puntuación de calidad."""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "model.joblib"

st.set_page_config(page_title="Calidad del vino | ML", page_icon="🍷", layout="wide")


@st.cache_resource
def load_artifact():
    return joblib.load(MODEL_PATH)


st.title("Estimador educativo de calidad del vino")
st.write(
    "Ingresa las mediciones fisicoquímicas de una muestra para estimar su puntuación "
    "sensorial. El modelo se entrenó con muestras de vino tinto Vinho Verde."
)
st.caption(
    "La predicción es una estimación estadística para fines académicos; no representa "
    "una evaluación oficial ni una recomendación de consumo."
)

if not MODEL_PATH.exists():
    st.error("No existe model.joblib. Ejecuta primero `python train.py` desde esta carpeta.")
    st.stop()

artifact = load_artifact()
model = artifact["model"]
features = artifact["features"]
labels = artifact.get("feature_labels", {})
metrics = artifact.get("metrics", {})
ranges = artifact.get("input_ranges", {})

with st.expander("Sobre el conjunto de datos y el modelo"):
    st.write(
        f"Dataset: {artifact.get('dataset', 'UCI Wine Quality')}. "
        f"Filas: {artifact.get('dataset_rows', '—')}. "
        f"La clase positiva es `quality >= {artifact.get('quality_threshold', 7)}`. "
        "Se usa regresión logística con imputación por mediana y estandarización, "
        "evaluada en una partición de prueba estratificada del 20%."
    )
    if metrics:
        st.write(
            f"En la partición de prueba: F1 de alta {metrics['f1_high']:.3f}, "
            f"recall de alta {metrics['recall_high']:.3f}, "
            f"ROC-AUC {metrics['roc_auc']:.3f}. "
            f"La línea base mayoritaria obtuvo accuracy {metrics['baseline_accuracy']:.3f}."
        )
        st.caption(
            "La matriz de confusión usa filas como clase real y columnas como clase predicha, "
            "en el orden habitual/alta. Se prioriza F1 y recall de alta porque esa clase "
            "es minoritaria."
        )
    importances = artifact.get("feature_importance", {})
    if importances:
        top = list(importances.items())[:5]
        st.write("Variables con mayor magnitud de coeficiente estandarizado:")
        st.dataframe(
            pd.DataFrame(
                [{"Variable": labels.get(name, name), "Magnitud": round(value, 3)} for name, value in top]
            ),
            hide_index=True,
            width="stretch",
        )
        st.caption("Esta importancia describe asociaciones aprendidas por el modelo, no causalidad.")

st.subheader("Características de la muestra")
st.write("Los valores iniciales corresponden a las medianas observadas en el dataset.")
values = {}
columns = st.columns(3)
for index, feature in enumerate(features):
    feature_range = ranges.get(feature, {})
    default = float(feature_range.get("median", 0.0))
    with columns[index % len(columns)]:
        values[feature] = st.number_input(
            labels.get(feature, feature),
            value=default,
            format="%.4f",
            key=f"input_{feature}",
        )

if st.button("Estimar calidad", type="primary"):
    X_new = pd.DataFrame([{feature: values[feature] for feature in features}], columns=features)
    prediction = int(model.predict(X_new)[0])
    class_labels = artifact.get("class_labels", {})
    class_label = class_labels.get(prediction, str(prediction))
    high_probability = float(model.predict_proba(X_new)[0][list(model.classes_).index(1)])
    st.success(f"Clase predicha: **{class_label}**")
    st.metric("Probabilidad estimada de alta calidad", f"{high_probability:.1%}")
    st.caption(
        f"La clase alta corresponde a una puntuación sensorial de "
        f"{artifact.get('quality_threshold', 7)} o más."
    )

    outside = [
        feature
        for feature, value in values.items()
        if feature in ranges and not ranges[feature]["min"] <= value <= ranges[feature]["max"]
    ]
    if outside:
        names = ", ".join(labels.get(feature, feature) for feature in outside)
        st.warning(
            f"Estas entradas están fuera del rango observado ({names}); "
            "la predicción puede ser menos confiable."
        )

st.divider()
st.caption("Datos: UCI Machine Learning Repository — Wine Quality (Cortez et al., 2009).")
