"""Script para entrenar el modelo de análisis de sentimientos."""

import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import numpy as np


def load_data():
    """Carga los datos desde el archivo CSV."""
    dataframe = pd.read_csv(
        "files/input/sentences.csv.zip",
        index_col=False,
        compression="zip",
    )
    
    data = dataframe.phrase
    target = dataframe.target
    
    return data, target


def create_pipeline():
    """Crea el pipeline de procesamiento y clasificación."""
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
        )),
        ('classifier', LogisticRegression(
            max_iter=1000,
            C=10.0,
            solver='lbfgs',
            random_state=42,
            class_weight='balanced',
        ))
    ])
    
    return pipeline


def train_and_save_model():
    """Entrena el modelo y lo guarda."""
    # Cargar datos
    print("Cargando datos...")
    data, target = load_data()
    print(f"Datos cargados: {len(data)} muestras")
    print(f"Distribución de clases:\n{target.value_counts()}")
    
    # Crear pipeline
    print("\nCreando pipeline...")
    estimator = create_pipeline()
    
    # Validación cruzada para verificar el rendimiento
    print("\nEvaluando con validación cruzada...")
    scores = cross_val_score(estimator, data, target, cv=5, scoring='accuracy')
    print(f"Accuracy por fold: {scores}")
    print(f"Accuracy promedio: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
    
    # Entrenar con todos los datos
    print("\nEntrenando modelo con todos los datos...")
    estimator.fit(data, target)
    
    # Verificar accuracy en el conjunto completo
    from sklearn.metrics import accuracy_score
    predictions = estimator.predict(data)
    accuracy = accuracy_score(target, predictions)
    print(f"Accuracy en conjunto completo: {accuracy:.4f}")
    
    # Guardar el modelo
    print("\nGuardando modelo...")
    with open("homework/estimator.pickle", "wb") as file:
        pickle.dump(estimator, file)
    
    print("¡Modelo guardado exitosamente en homework/estimator.pickle!")
    print(f"Accuracy final: {accuracy:.4f}")
    
    if accuracy > 0.9545:
        print("✓ El modelo cumple con el requisito de accuracy > 0.9545")
    else:
        print("✗ ADVERTENCIA: El modelo NO cumple con el requisito de accuracy > 0.9545")
    
    return estimator


if __name__ == "__main__":
    train_and_save_model()