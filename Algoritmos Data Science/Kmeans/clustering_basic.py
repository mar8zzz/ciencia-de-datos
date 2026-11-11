# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 07:46:04 2025

@author: Profesor D
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Cargar datos
df = pd.read_csv('insurance.csv')

print("ANÁLISIS EXPLORATORIO DE DATOS - SEGUROS")
print("=" * 50)

# 1. Estadísticas básicas de 'charges'
print("\n1. ESTADÍSTICAS DE CHARGES:")
print(f"Media: ${df['charges'].mean():.2f}")
print(f"Mediana: ${df['charges'].median():.2f}")
print(f"Rango: ${df['charges'].min():.2f} - ${df['charges'].max():.2f}")

# 2. Cuartiles
Q1 = df['charges'].quantile(0.25)
Q2 = df['charges'].quantile(0.50)
Q3 = df['charges'].quantile(0.75)

print(f"\n2. CUARTILES:")
print(f"Q1 (25%): ${Q1:.2f}")
print(f"Q2 (Mediana - 50%): ${Q2:.2f}")
print(f"Q3 (75%): ${Q3:.2f}")

# 3. Clustering para definir niveles
X = df[['charges']].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Aplicar K-means con 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)
df['cluster'] = clusters

# Asignar niveles basados en los centroides
centroids = kmeans.cluster_centers_
centroids_original = scaler.inverse_transform(centroids)
centroid_mapping = {i: centroid[0] for i, centroid in enumerate(centroids_original)}

# Ordenar clusters de menor a mayor centroide
sorted_clusters = sorted(centroid_mapping, key=centroid_mapping.get)
level_mapping = {sorted_clusters[0]: 'low', 
                sorted_clusters[1]: 'medium', 
                sorted_clusters[2]: 'high'}

df['level'] = df['cluster'].map(level_mapping)

# 4. Mostrar distribución por niveles
print(f"\n3. DISTRIBUCIÓN POR NIVELES:")
for level in ['low', 'medium', 'high']:
    count = len(df[df['level'] == level])
    avg_charge = df[df['level'] == level]['charges'].mean()
    print(f"- {level.upper()}: {count} personas - Cargo promedio: ${avg_charge:.2f}")

# Visualización de los clusters
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
colors = {'low': 'green', 'medium': 'orange', 'high': 'red'}
for level in df['level'].unique():
    level_data = df[df['level'] == level]
    plt.hist(level_data['charges'], bins=30, alpha=0.6, 
             label=f'{level.capitalize()} (n={len(level_data)})', 
             color=colors[level])
plt.xlabel('Charges ($)')
plt.ylabel('Frecuencia')
plt.title('Distribución por Niveles', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
df_boxplot = df.copy()
df_boxplot['level'] = pd.Categorical(df_boxplot['level'], categories=['low', 'medium', 'high'])
sns.boxplot(x='level', y='charges', data=df_boxplot, palette=colors)
plt.title('Boxplot por Niveles', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 3)
sns.violinplot(x='level', y='charges', data=df_boxplot, palette=colors)
plt.title('Violin Plot por Niveles', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Guardar nuevo dataset
df_final = df.drop('cluster', axis=1)
df_final.to_csv('insurance_with_levels.csv', index=False)
print(f"\n4. NUEVO DATASET GUARDADO: 'insurance_with_levels.csv'")
print(f"   - Total de registros: {len(df_final)}")
print(f"   - Nueva columna 'level' añadida")