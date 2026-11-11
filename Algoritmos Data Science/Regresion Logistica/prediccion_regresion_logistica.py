# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 07:46:24 2026

@author: HP 840 G6
"""

import pandas as pd
import joblib

print("=== PREDICCIÓN CON MODELO DE REGRESIÓN LOGÍSTICA ===")

# Cargar modelo y preprocesadores
try:
    modelo = joblib.load('modelo_regresion_logistica.pkl')
    scaler = joblib.load('scaler_regresion_logistica.pkl')
    label_encoders = joblib.load('label_encoders_regresion_logistica.pkl')
    umbral = joblib.load('umbral_regresion_logistica.pkl')
    print("✓ Modelo cargado exitosamente")
    print(f"✓ Umbral de decisión (mediana de charges): ${umbral:.2f}")
except Exception as e:
    print(f"❌ Error: Ejecuta primero 'modelo_regresion_logistica.py' -> {e}")
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
    le = label_encoders[col]
    df_caso[col] = le.transform(df_caso[col])

# Estandarizar y predecir
X_scaled = scaler.transform(df_caso)
prediccion = modelo.predict(X_scaled)[0]
probabilidad = modelo.predict_proba(X_scaled)[0][1]

# Interpretar resultado
if prediccion == 1:
    clase = "ALTO RIESGO"
else:
    clase = "BAJO RIESGO"

print(f"\nResultado:")
print(f"  Clase predicha      : {clase}")
print(f"  Probabilidad alto riesgo : {probabilidad:.4f} ({probabilidad*100:.2f}%)")
print(f"  Probabilidad bajo riesgo : {1 - probabilidad:.4f} ({(1-probabilidad)*100:.2f}%)")