# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 07:53:17 2025

@author: Profesor D
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 07:46:04 2025

@author: Profesor D
"""

import pandas as pd
import joblib

print("=== PREDICCIÓN CON MODELO DE REGRESIÓN LINEAL ===")

# Cargar modelo y preprocesadores
try:
    modelo = joblib.load('modelo_regresion_lineal.pkl')
    scaler = joblib.load('scaler_regresion_lineal.pkl')
    print("✓ Modelo cargado exitosamente")
except:
    print("❌ Error: Ejecuta primero 'modelo_regresion_lineal.py'")
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
label_encoders = {}
for col in ['sex', 'smoker', 'region']:
    le = joblib.load(f'label_encoders_regresion_lineal.pkl')[col]
    df_caso[col] = le.transform(df_caso[col])

# Estandarizar y predecir
X_scaled = scaler.transform(df_caso)
prediccion = modelo.predict(X_scaled)[0]

print(f"\nResultado: Carga estimada de ${prediccion:.2f}")