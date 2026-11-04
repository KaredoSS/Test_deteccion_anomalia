import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd
from sklearn.decomposition import PCA

# Configuración de la página web
st.set_page_config(page_title="Detección de Intrusiones - PCA & SVD", layout="centered")

st.title("🛡️ Sistema de Detección de Intrusiones en Servidores")
st.markdown(
    "Modelo híbrido basado en **PCA y SVD** para identificar accesos no autorizados (inspirado en brechas de seguridad de la industria de videojuegos).")

# Barra lateral para controles interactivos
st.sidebar.header("Parámetros de Simulación")
n_muestras = st.sidebar.slider("Número de registros de logs:", min_value=100, max_value=1000, value=600, step=50)


# 1. Generación de datos
def generar_datos_servidor(n):
    M = np.zeros((n, 4))
    for i in range(n):
        if i < int(n * 0.95):
            M[i, 0] = np.random.poisson(0.2)
            M[i, 1] = np.random.normal(45, 10)
            M[i, 2] = np.random.normal(120, 25)
            M[i, 3] = np.random.choice([0, 1], p=[0.85, 0.15])
        else:
            M[i, 0] = np.random.poisson(6)
            M[i, 1] = np.random.normal(550, 40)
            M[i, 2] = np.random.normal(10, 3)
            M[i, 3] = 1.0
    return M


# 2. Filtrado SVD
def filtrar_ruido_svd(A):
    U, s, Vt = svd(A, full_matrices=False)
    k = 2
    Sigma_red = np.zeros((k, k))
    for i in range(k):
        Sigma_red[i, i] = s[i]
    return np.dot(U[:, :k], np.dot(Sigma_red, Vt[:k, :]))


# 3. Detección de anomalías
def detectar_anomalias(X_pca):
    filas, columnas = X_pca.shape
    centroide = np.zeros(columnas)
    for i in range(filas):
        for j in range(columnas):
            centroide[j] += X_pca[i, j]
    for j in range(columnas):
        centroide[j] = centroide[j] / filas

    distancias = np.zeros(filas)
    for i in range(filas):
        suma = 0
        for j in range(columnas):
            suma += (X_pca[i, j] - centroide[j]) ** 2
        distancias[i] = np.sqrt(suma)

    umbral = np.mean(distancias) + 2.5 * np.std(distancias)

    es_anomalia = np.zeros(filas, dtype=bool)
    for i in range(filas):
        if distancias[i] > umbral:
            es_anomalia[i] = True
    return es_anomalia


# Botón para ejecutar el modelo
if st.button("Ejecutar Análisis de Servidores"):
    with st.spinner("Procesando matrices y aplicando SVD/PCA..."):
        np.random.seed(42)
        matriz_logs = generar_datos_servidor(n_muestras)
        matriz_limpia = filtrar_ruido_svd(matriz_logs)

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(matriz_limpia)

        anomalias = detectar_anomalias(X_pca)

        # Renderizar la gráfica en Streamlit
        fig, ax = plt.subplots(figsize=(8, 5))
        filas_pca, _ = X_pca.shape

        for i in range(filas_pca):
            if anomalias[i]:
                ax.scatter(X_pca[i, 0], X_pca[i, 1], c='crimson', marker='X', s=70,
                           label='Brecha Detectada' if i == filas_pca - 1 else "")
            else:
                ax.scatter(X_pca[i, 0], X_pca[i, 1], c='forestgreen', alpha=0.6, s=25,
                           label='Tráfico Legítimo' if i == 0 else "")

        ax.set_title("Detección Espacial de Intrusiones (PCA-SVD)")
        ax.set_xlabel("Componente Principal 1")
        ax.set_ylabel("Componente Principal 2")
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend()

        st.pyplot(fig)
        st.success("¡Análisis completado con éxito! Se han separado los clústeres de tráfico normal y las anomalías.")