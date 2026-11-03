import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd
from sklearn.decomposition import PCA


def generar_datos_servidor(n_muestras):
    # Definimos la matriz para los logs
    M = np.zeros((n_muestras, 4))

    for i in range(n_muestras):
        # tráfico normal y anomalías
        if i < int(n_muestras * 0.95):
            M[i, 0] = np.random.poisson(0.2)  # Intentos fallidos
            M[i, 1] = np.random.normal(45, 10)  # Volumen de transferencia (MB)
            M[i, 2] = np.random.normal(120, 25)  # Duración de sesión (min)
            M[i, 3] = np.random.choice([0, 1], p=[0.85, 0.15])  # Acceso a código
        else:
            M[i, 0] = np.random.poisson(6)  # Intentos fallidos masivamente
            M[i, 1] = np.random.normal(550, 40)  # Descarga masiva anómala
            M[i, 2] = np.random.normal(10, 3)  # Sesión corta
            M[i, 3] = 1.0  # Acceso crítico forzado

    return M


def filtrar_ruido_svd(A):
    filas, columnas = A.shape
    # Aplicamos descomposición en valores singulares
    U, s, Vt = svd(A, full_matrices=False)

    # Truncamiento de la matriz para eliminar ruido
    k = 2
    Sigma_red = np.zeros((k, k))
    for i in range(k):
        Sigma_red[i, i] = s[i]

    # Reconstrucción matricial filtrada
    A_filtrada = np.dot(U[:, :k], np.dot(Sigma_red, Vt[:k, :]))
    return A_filtrada


def detectar_anomalias_con_bucles(X_pca):
    filas, columnas = X_pca.shape

    # Calculamos el centroide de forma manual mediante bucles
    centroide = np.zeros(columnas)
    for i in range(filas):
        for j in range(columnas):
            centroide[j] += X_pca[i, j]

    for j in range(columnas):
        centroide[j] = centroide[j] / filas

    # Calculamos la distancia euclidiana de cada punto al centroide
    distancias = np.zeros(filas)
    for i in range(filas):
        suma_cuadrados = 0
        for j in range(columnas):
            suma_cuadrados += (X_pca[i, j] - centroide[j]) ** 2
        distancias[i] = np.sqrt(suma_cuadrados)

    # Establecemos un umbral estadístico manual
    media_d = np.mean(distancias)
    std_d = np.std(distancias)
    umbral = media_d + 2.5 * std_d

    # Evaluamos qué registros superan el umbral
    es_anomalia = np.zeros(filas, dtype=bool)
    for i in range(filas):
        if distancias[i] > umbral:
            es_anomalia[i] = True
        else:
            es_anomalia[i] = False

    return es_anomalia


# Flujo del programa
np.random.seed(42)

# Generamos los datos simulados de los servidores
matriz_logs = generar_datos_servidor(600)

# Aplicamos SVD para limpiar el ruido de la red
matriz_limpia = filtrar_ruido_svd(matriz_logs)

# Reducción dimensional con PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(matriz_limpia)

# Detección de brechas usando bucles iterativos
anomalias = detectar_anomalias_con_bucles(X_pca)

# Visualización de resultados
plt.figure(figsize=(9, 6))
filas_pca, _ = X_pca.shape

for i in range(filas_pca):
    if anomalias[i]:
        plt.scatter(X_pca[i, 0], X_pca[i, 1], c='crimson', marker='X', s=80,
                    label='Brecha Detectada' if i == 595 else "")
    else:
        plt.scatter(X_pca[i, 0], X_pca[i, 1], c='forestgreen', alpha=0.6, s=30,
                    label='Tráfico Legítimo' if i == 0 else "")

plt.title('Detección de Intrusiones en Servidores (Modelo PCA-SVD)')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()