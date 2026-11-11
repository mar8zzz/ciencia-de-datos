# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 07:46:04 2025

@author: Profesor D
"""


import pandas as pd
import joblib

print("=== PREDICCIÓN CON MODELO K-NN ===")

# Cargar modelo y preprocesadores
try:
    knn = joblib.load('modelo_knn.pkl')
    scaler = joblib.load('scaler_knn.pkl')
    label_encoders = joblib.load('label_encoders_knn.pkl')
    print("✓ Modelo cargado exitosamente")
except:
    print("❌ Error: Ejecuta primero 'modelo_knn.py'")
    exit()

# Un solo caso de ejemplo
caso = {
    'age': 28,
    'sex': 'male',
    'bmi': 28.5,
    'children': 2,
    'smoker': 'yes',
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
prediccion = knn.predict(X_scaled)[0]
probabilidades = knn.predict_proba(X_scaled)[0]

# Convertir resultado
nivel_predicho = label_encoders['level'].inverse_transform([prediccion])[0]

print(f"\nResultado: Nivel {nivel_predicho.upper()}")
print("\nProbabilidades:")
for i, prob in enumerate(probabilidades):
    nivel = label_encoders['level'].classes_[i]
    print(f"  {nivel}: {prob:.1%}")