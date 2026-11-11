# -*- coding: utf-8 -*-
"""
Created on Wed Oct 29 07:56:20 2025

@author: Profesor D
"""


import pandas as pd
import joblib

print("=== PREDICCIÓN CON MODELO DE ÁRBOL DE DECISIÓN ===")

# Cargar modelo y preprocesadores
try:
    modelo = joblib.load('modelo_arbol_decision.pkl')
    scaler = joblib.load('scaler_arbol_decision.pkl')
    label_encoders = joblib.load('label_encoders_arbol_decision.pkl')
    print("✓ Modelo cargado exitosamente")
except:
    print("❌ Error: Ejecuta primero 'modelo_arbol_decision.py'")
    exit()

# Un solo caso de ejemplo
caso = {
    'age': 35,
    'sex': 'male',
    'bmi': 28.5,
    'children': 2,
    'smoker': 'no',
    'region': 'southeast'
}

print(f"\nDatos del caso:")
for key, value in caso.items():
    print(f"  {key}: {value}")

# Convertir a DataFrame y preprocesar
df_caso = pd.DataFrame([caso])

# Aplicar encoding a variables categóricas
for col in ['sex', 'smoker', 'region']:
    df_caso[col] = label_encoders[col].transform(df_caso[col])

# Estandarizar y predecir
X_scaled = scaler.transform(df_caso)
prediccion = modelo.predict(X_scaled)[0]

print(f"\nResultado: Carga estimada de ${prediccion:.2f}")