 # -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 08:24:08 2026

@author: HP 840 G6
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report,
                             roc_auc_score)
import joblib
import warnings
warnings.filterwarnings('ignore')


def crear_modelo_regresion_logistica():
    print("CARGANDO Y PREPROCESANDO DATOS...")

    # Cargar el dataset
    df = pd.read_csv('insurance_with_levels.csv')

    # Excluir la columna level
    df = df.drop('level', axis=1)

    print(f"Dataset cargado: {df.shape[0]} registros, {df.shape[1]} columnas")
    print(f"Columnas: {list(df.columns)}")

    # Crear variable objetivo binaria a partir de 'charges'
    # 1 = alto riesgo (charges > mediana), 0 = bajo riesgo
    mediana = df['charges'].median()
    df['alto_riesgo'] = (df['charges'] > mediana).astype(int)
    print(f"\nVariable objetivo creada: 'alto_riesgo' (umbral = ${mediana:.2f})")
    print(f"Distribución: {df['alto_riesgo'].value_counts().to_dict()}")

    # Eliminar la columna 'charges' original para evitar fuga de información
    df = df.drop('charges', axis=1)

    # Preprocesamiento de variables categóricas
    label_encoders = {}
    categorical_columns = ['sex', 'smoker', 'region']

    for col in categorical_columns:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
            print(f"Encoded {col}: {list(le.classes_)}")

    # Separar características (X) y variable objetivo (y)
    X = df.drop('alto_riesgo', axis=1)
    y = df['alto_riesgo']

    print(f"\nCaracterísticas (X): {X.shape}")
    print(f"Variable objetivo (y): {y.shape}")

    # Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Estandarizar las características
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\nENTRENANDO MODELO DE REGRESIÓN LOGÍSTICA...")

    # Crear y entrenar el modelo de regresión logística
    modelo = LogisticRegression(max_iter=1000, random_state=42)
    modelo.fit(X_train_scaled, y_train)

    # Evaluar el modelo
    y_pred = modelo.predict(X_test_scaled)
    y_proba = modelo.predict_proba(X_test_scaled)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    print("\nRESULTADOS DEL MODELO:")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")
    print(f"\nMatriz de confusión:\n{cm}")
    print(f"\nReporte de clasificación:\n{classification_report(y_test, y_pred)}")

    # Guardar el modelo y preprocesadores
    joblib.dump(modelo, 'modelo_regresion_logistica.pkl')
    joblib.dump(scaler, 'scaler_regresion_logistica.pkl')
    joblib.dump(label_encoders, 'label_encoders_regresion_logistica.pkl')
    joblib.dump(mediana, 'umbral_regresion_logistica.pkl')

    print("\nMODELO GUARDADO EXITOSAMENTE:")
    print("✓ modelo_regresion_logistica.pkl - Modelo de regresión logística entrenado")
    print("✓ scaler_regresion_logistica.pkl - Estandarizador de características")
    print("✓ label_encoders_regresion_logistica.pkl - Encoders para variables categóricas")
    print("✓ umbral_regresion_logistica.pkl - Umbral usado para crear la variable objetivo")

    return modelo, scaler, label_encoders


if __name__ == "__main__":
    crear_modelo_regresion_logistica()