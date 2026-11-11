import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, r2_score
from sklearn.datasets import make_classification, make_regression
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="Práctica de Modelado", layout="wide")
st.title("Práctica 3: Modelado, Joblib y Datos Sintéticos")

# 1. Análisis del conjunto de datos completo
st.header("1. Análisis del Conjunto de Datos Original")
df = pd.read_csv('insurance_with_levels.csv')
st.write(f"**Dimensiones del dataset:** {df.shape[0]} filas y {df.shape[1]} columnas.")
st.write("**Estadísticas descriptivas:**")
st.dataframe(df.describe())

# Asignar variables objetivo (ajusta 'age' o la última columna según tu CSV)
target_clas = df.columns[-1] 
target_reg = 'age' if 'age' in df.columns else df.columns[0]

# Preprocesamiento y Guardado de LabelEncoder
le = LabelEncoder()
cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    df[col] = le.fit_transform(df[col])
joblib.dump(le, 'label_encoder_activo.pkl') # Guardado (Activo)

# 2. Implementación de los 4 Modelos de Clasificación
st.header("2. Modelos de Clasificación (Datos Reales)")
X_clas = df.drop(columns=[target_clas])
y_clas = df[target_clas]
Xc_train, Xc_test, yc_train, yc_test = train_test_split(X_clas, y_clas, test_size=0.3, random_state=42)

modelos_clasificacion = {
    "Logística": Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(max_iter=1000))]),
    "KNN": Pipeline([('scaler', StandardScaler()), ('clf', KNeighborsClassifier(n_neighbors=5))]),
    "SVM": Pipeline([('scaler', StandardScaler()), ('clf', SVC())]),
    "Naive Bayes": Pipeline([('scaler', StandardScaler()), ('clf', GaussianNB())])
}

col1, col2 = st.columns(2)
for idx, (nombre, pipeline) in enumerate(modelos_clasificacion.items()):
    pipeline.fit(Xc_train, yc_train)
    preds = pipeline.predict(Xc_test)
    acc = accuracy_score(yc_test, preds)
    joblib.dump(pipeline, f'pipeline_clas_{nombre.lower()}.pkl') # Guardado (Activo)
    
    with col1 if idx % 2 == 0 else col2:
        st.success(f"**{nombre} Accuracy:** {acc:.2f}")

# 3. Implementación de los 3 Modelos de Regresión
st.header("3. Modelos de Regresión (Datos Reales)")
X_reg = df.drop(columns=[target_reg])
y_reg = df[target_reg]
Xr_train, Xr_test, yr_train, yr_test = train_test_split(X_reg, y_reg, test_size=0.3, random_state=42)

modelos_regresion = {
    "Regresión Lineal": Pipeline([('scaler', StandardScaler()), ('reg', LinearRegression())]),
    "Árbol de Decisión": Pipeline([('scaler', StandardScaler()), ('reg', DecisionTreeRegressor(random_state=42))]),
    "Random Forest": Pipeline([('scaler', StandardScaler()), ('reg', RandomForestRegressor(random_state=42))])
}

col3, col4, col5 = st.columns(3)
for idx, (nombre, pipeline) in enumerate(modelos_regresion.items()):
    pipeline.fit(Xr_train, yr_train)
    preds = pipeline.predict(Xr_test)
    r2 = r2_score(yr_test, preds)
    joblib.dump(pipeline, f'pipeline_reg_{nombre.replace(" ", "_").lower()}.pkl') # Guardado (Activo)
    
    with [col3, col4, col5][idx]:
        st.info(f"**{nombre} R²:** {r2:.2f}")

# 4. Carga de Activos y Datos Sintéticos
st.header("4. Carga de Activos (Prueba) y Datos Sintéticos")
st.write("Cargando el pipeline de KNN y Random Forest guardados previamente...")
knn_cargado = joblib.load('pipeline_clas_knn.pkl')
rf_cargado = joblib.load('pipeline_reg_random_forest.pkl')
st.success("Activos cargados correctamente para pruebas de inferencia.")

st.subheader("Pruebas obligatorias con Datos Sintéticos")
tab_clas, tab_reg = st.tabs(["Prueba Sintética Clasificación", "Prueba Sintética Regresión"])

with tab_clas:
    Xs_clas, ys_clas = make_classification(n_samples=200, n_features=10, random_state=42)
    Xsc_train, Xsc_test, ysc_train, ysc_test = train_test_split(Xs_clas, ys_clas, test_size=0.3)
    for nombre, pipeline in modelos_clasificacion.items():
        pipeline.fit(Xsc_train, ysc_train)
        st.write(f"**{nombre}** - Rendimiento sintético (Accuracy): {accuracy_score(ysc_test, pipeline.predict(Xsc_test)):.2f}")

with tab_reg:
    Xs_reg, ys_reg = make_regression(n_samples=200, n_features=10, noise=0.1, random_state=42)
    Xsr_train, Xsr_test, ysr_train, ysr_test = train_test_split(Xs_reg, ys_reg, test_size=0.3)
    for nombre, pipeline in modelos_regresion.items():
        pipeline.fit(Xsr_train, ysr_train)
        st.write(f"**{nombre}** - Rendimiento sintético (R²): {r2_score(ysr_test, pipeline.predict(Xsr_test)):.2f}")