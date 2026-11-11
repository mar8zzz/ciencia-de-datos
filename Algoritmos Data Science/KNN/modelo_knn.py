# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 07:46:04 2025

@author: Profesor D
"""


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import warnings
warnings.filterwarnings('ignore')

def crear_modelo_knn():
    print("CARGANDO Y PREPROCESANDO DATOS...")
    
    # Cargar el dataset con la columna level
    df = pd.read_csv('insurance_with_levels.csv')
    
    # Excluir la columna charges como solicitado
    df = df.drop('charges', axis=1)
    
    print(f"Dataset cargado: {df.shape[0]} registros, {df.shape[1]} columnas")
    print(f"Columnas: {list(df.columns)}")
    
    # Preprocesamiento de variables categóricas
    label_encoders = {}
    categorical_columns = ['sex', 'smoker', 'region', 'level']
    
    for col in categorical_columns:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
            print(f"Encoded {col}: {list(le.classes_)}")
    
    # Separar características (X) y variable objetivo (y)
    X = df.drop('level', axis=1)
    y = df['level']
    
    print(f"\nCaracterísticas (X): {X.shape}")
    print(f"Variable objetivo (y): {y.shape}")
    print(f"Distribución de niveles: {np.unique(y, return_counts=True)}")
    
    # Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Estandarizar las características (importante para K-NN)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\nENTRENANDO MODELO K-NN...")
    
    # Crear y entrenar el modelo K-NN
    knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
    knn.fit(X_train_scaled, y_train)
    
    # Evaluar el modelo
    y_pred = knn.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nRESULTADOS DEL MODELO:")
    print(f"Precisión (Accuracy): {accuracy:.4f}")
    print(f"\nMatriz de Confusión:")
    print(confusion_matrix(y_test, y_pred))
    print(f"\nReporte de Clasificación:")
    print(classification_report(y_test, y_pred, 
                              target_names=label_encoders['level'].classes_))
    
    # Guardar el modelo y preprocesadores
    joblib.dump(knn, 'modelo_knn.pkl')
    joblib.dump(scaler, 'scaler_knn.pkl')
    joblib.dump(label_encoders, 'label_encoders_knn.pkl')
    
    print(f"\nMODELO GUARDADO EXITOSAMENTE:")
    print("✓ modelo_knn.pkl - Modelo K-NN entrenado")
    print("✓ scaler.pkl - Estandarizador de características")
    print("✓ label_encoders.pkl - Encoders para variables categóricas")
    
    return knn, scaler, label_encoders

if __name__ == "__main__":
    crear_modelo_knn()