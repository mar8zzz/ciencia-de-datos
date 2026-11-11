# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 07:46:04 2025

@author: Profesor D
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

def crear_modelo_arbol_decision():
    print("CARGANDO Y PREPROCESANDO DATOS...")
    
    # Cargar el dataset
    df = pd.read_csv('insurance_with_levels.csv')
    
    # Excluir la columna level
    df = df.drop('level', axis=1)
    
    print(f"Dataset cargado: {df.shape[0]} registros, {df.shape[1]} columnas")
    print(f"Columnas: {list(df.columns)}")
    
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
    X = df.drop('charges', axis=1)
    y = df['charges']
    
    print(f"\nCaracterísticas (X): {X.shape}")
    print(f"Variable objetivo (y): {y.shape}")
    
    # Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Estandarizar las características
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("\nENTRENANDO MODELO DE ÁRBOL DE DECISIÓN...")
    
    # Crear y entrenar el modelo de árbol de decisión
    modelo = DecisionTreeRegressor()
    modelo.fit(X_train_scaled, y_train)
    
    
    # Evaluar el modelo
    y_pred = modelo.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("\nRESULTADOS DEL MODELO:")
    print("Error Cuadrático Medio (MSE): {mse:.4f}")
    print("R^2 Score: {r2:.4f}")
    
    # Guardar el modelo y preprocesadores
    joblib.dump(modelo, 'modelo_arbol_decision.pkl')
    joblib.dump(scaler, 'scaler_arbol_decision.pkl')
    joblib.dump(label_encoders, 'label_encoders_arbol_decision.pkl')
    
    print(f"\nMODELO GUARDADO EXITOSAMENTE:")
    print("✓ modelo_arbol_decision.pkl - Modelo de árbol de decisión entrenado")
    print("✓ scaler_arbol_decision.pkl - Estandarizador de características")
    print("✓ label_encoders_arbol_decision.pkl - Encoders para variables categóricas")
    
    return modelo, scaler, label_encoders

if __name__ == "__main__":
    crear_modelo_arbol_decision()
