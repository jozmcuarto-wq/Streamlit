"""Entrena y evalúa un modelo reproducible para estimar la calidad del vino."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = PROJECT_DIR / "data" / "winequality-red.csv"
DEFAULT_MODEL_PATH = PROJECT_DIR / "model.joblib"
TARGET = "quality"
HIGH_QUALITY_THRESHOLD = 7
FEATURES = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]
FEATURE_LABELS = {
    "fixed acidity": "Acidez fija (g/dm³)",
    "volatile acidity": "Acidez volátil (g/dm³)",
    "citric acid": "Ácido cítrico (g/dm³)",
    "residual sugar": "Azúcar residual (g/dm³)",
    "chlorides": "Cloruros (g/dm³)",
    "free sulfur dioxide": "Dióxido de azufre libre (mg/dm³)",
    "total sulfur dioxide": "Dióxido de azufre total (mg/dm³)",
    "density": "Densidad (g/cm³)",
    "pH": "pH",
    "sulphates": "Sulfatos (g/dm³)",
    "alcohol": "Alcohol (% vol.)",
}


def load_data(data_path: Path) -> pd.DataFrame:
    """Carga el CSV de UCI, que utiliza punto y coma como separador."""
    if not data_path.exists():
        raise FileNotFoundError(
            f"No se encontró el dataset: {data_path}. "
            "Consulte README.md para volver a descargarlo."
        )

    df = pd.read_csv(data_path, sep=";")
    required_columns = FEATURES + [TARGET]
    missing = sorted(set(required_columns) - set(df.columns))
    if missing:
        raise ValueError(f"Faltan columnas requeridas en el CSV: {missing}")

    df = df[required_columns].apply(pd.to_numeric, errors="coerce")
    if df[TARGET].isna().any():
        raise ValueError("La columna objetivo 'quality' contiene datos vacíos o no numéricos.")
    return df


def train_model(df: pd.DataFrame) -> tuple[dict, dict]:
    """Separa holdout, ajusta el pipeline y devuelve el artefacto y métricas."""
    X = df[FEATURES].copy()
    # La calidad de UCI es ordinal y desbalanceada. Se define como positiva
    # la clase de puntuación alta (7 o más) antes de separar los datos.
    y = (df[TARGET] >= HIGH_QUALITY_THRESHOLD).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, list(model.classes_).index(1)]
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)
    baseline_predictions = baseline.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision_high": float(precision_score(y_test, predictions, pos_label=1, zero_division=0)),
        "recall_high": float(recall_score(y_test, predictions, pos_label=1, zero_division=0)),
        "f1_high": float(f1_score(y_test, predictions, pos_label=1, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "baseline_accuracy": float(accuracy_score(y_test, baseline_predictions)),
        "baseline_f1_high": float(f1_score(y_test, baseline_predictions, pos_label=1, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["habitual", "alta"],
            output_dict=True,
            zero_division=0,
        ),
        "test_size": int(len(y_test)),
        "train_size": int(len(y_train)),
        "random_state": 42,
    }

    classifier = model.named_steps["classifier"]
    coefficients = np.abs(classifier.coef_[0])
    feature_importance = {
        feature: float(score)
        for feature, score in sorted(
            zip(FEATURES, coefficients),
            key=lambda item: item[1],
            reverse=True,
        )
    }
    artifact = {
        "model": model,
        "features": FEATURES,
        "feature_labels": FEATURE_LABELS,
        "numeric_features": FEATURES,
        "categorical_features": [],
        "target": TARGET,
        "task": "classification",
        "classification_target": "high_quality",
        "quality_threshold": HIGH_QUALITY_THRESHOLD,
        "class_labels": {
            0: f"Calidad habitual (puntuación menor que {HIGH_QUALITY_THRESHOLD})",
            1: f"Alta calidad sensorial (puntuación de {HIGH_QUALITY_THRESHOLD} o más)",
        },
        "dataset": "UCI Wine Quality (red vinho verde)",
        "dataset_rows": int(len(df)),
        "class_counts": {
            0: int((y == 0).sum()),
            1: int((y == 1).sum()),
        },
        "dataset_source": "https://doi.org/10.24432/C56S3T",
        "metrics": metrics,
        "feature_importance": feature_importance,
        "input_ranges": {
            feature: {
                "min": float(df[feature].min()),
                "max": float(df[feature].max()),
                "median": float(df[feature].median()),
            }
            for feature in FEATURES
        },
        "source_quality_range": [int(df[TARGET].min()), int(df[TARGET].max())],
    }
    return artifact, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Ruta al CSV winequality-red.csv (por defecto: data/winequality-red.csv).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Ruta de salida para model.joblib (por defecto: model.joblib).",
    )
    args = parser.parse_args()

    df = load_data(args.data)
    artifact, metrics = train_model(df)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.output)

    print(f"Filas: {len(df)} | Entrenamiento: {metrics['train_size']} | Prueba: {metrics['test_size']}")
    print(f"Accuracy: {metrics['accuracy']:.3f} | Baseline mayoritaria: {metrics['baseline_accuracy']:.3f}")
    print(f"Precisión clase alta: {metrics['precision_high']:.3f}")
    print(f"Recall clase alta: {metrics['recall_high']:.3f}")
    print(f"F1 clase alta: {metrics['f1_high']:.3f} | Baseline: {metrics['baseline_f1_high']:.3f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.3f}")
    print("Matriz de confusión [habitual, alta]:")
    print(np.array(metrics["confusion_matrix"]))
    print("Reporte de clasificación en el conjunto de prueba:")
    print(pd.DataFrame(metrics["classification_report"]).T.round(3).to_string())
    print(f"Modelo guardado en: {args.output.resolve()}")


if __name__ == "__main__":
    main()
