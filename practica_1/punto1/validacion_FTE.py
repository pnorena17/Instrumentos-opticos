import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# --- 1. Parámetros de la Simulación ---
long_onda = 633e-9    # 633 nm
k = (2 * np.pi) / long_onda
z = 0.01               # Distancia de propagación (ej: 10 cm)

# --- Parámetros de la Rejilla (Grid) ---
N = 2048              # Aumentamos N para mejor resolución y evitar aliasing
dx = 8e-6             # Tamaño de píxel (8 um)
L = dx * N
df = 1 / L

z_max = N*(dx**2)/long_onda
print(z_max)
assert z <= z_max, "No cumple el criterio de z para FTE"

# --- Coordenadas Espaciales y de Frecuencia ---
x = (np.arange(N) - N // 2) * dx
X, Y = np.meshgrid(x, x)

fx = (np.arange(N) - N // 2) * df
Fx, Fy = np.meshgrid(fx, fx)

# --- 2. Campo Inicial: Haz Gaussiano ---
w0 = 0.5e-3  # Radio de la cintura del haz (0.5 mm)
U_0 = np.exp(-(X**2 + Y**2) / w0**2)

# --- 3. Propagación Numérica (Tu algoritmo de Espectro Angular) ---
print("Iniciando propagación numérica...")
A_0 = np.fft.fftshift(np.fft.fft2(U_0))

# Argumento para la raíz cuadrada en kz
arg_kz = (1. / long_onda)**2 - Fx**2 - Fy**2

# Calculamos kz para ondas propagantes y evanescentes
tmp = np.sqrt(np.abs(arg_kz))
kz = np.where(arg_kz >= 0, tmp, 1j * tmp)

# Función de transferencia H(fx, fy)
H = np.exp(1j * z * 2 * np.pi * kz)

# Propagación en el dominio de la frecuencia
A = A_0 * H
A_ishift = np.fft.ifftshift(A)

# Regreso al dominio espacial
U_simulado = np.fft.ifft2(A_ishift)

# Calculamos la intensidad 2D simulada
intensidad_simulada_2D = np.abs(U_simulado)**2
print("Propagación numérica completada.")

# --- 4. Solución Analítica para Comparación ---
def intensidad_gaussiana_analitica(r, z, w0, lambda_):
    """
    Calcula la intensidad de un haz gaussiano propagado una distancia z.
    """
    # 1. Rango de Rayleigh
    z_R = np.pi * w0**2 / lambda_
    # 2. Radio del haz en z
    w_z = w0 * np.sqrt(1 + (z / z_R)**2)
    # 3. Intensidad (asumiendo I0 = 1 en z=0)
    I_rz = (w0 / w_z)**2 * np.exp(-2 * r**2 / w_z**2)
    return I_rz

# Calculamos el perfil 1D analítico usando el mismo eje de coordenadas `x`
perfil_analitico = intensidad_gaussiana_analitica(x, z, w0, long_onda)

# --- 5. Extracción de Perfiles y Normalización ---
center_index = N // 2
# Extraemos el perfil central de la simulación 2D
perfil_simulado = intensidad_simulada_2D[center_index, :]

# Normalizamos ambos perfiles para que su máximo sea 1 y la comparación sea justa
perfil_simulado_norm = perfil_simulado / np.max(perfil_simulado)
perfil_analitico_norm = perfil_analitico / np.max(perfil_analitico)

# --- 6. Graficar Solo los Perfiles de Comparación ---

correlation_coefficient, p_value = pearsonr(
    perfil_simulado_norm, 
    perfil_analitico_norm
)

print("\n--- Validación de Correlación ---")
print(f"Coeficiente de Correlación de Pearson (ρ): {correlation_coefficient:.5f}")

# Creamos una única figura y un único eje
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

# Dibujamos los perfiles normalizados
ax.plot(x * 1e3, perfil_analitico_norm, 'r-', linewidth=3, label='Solución Analítica Exacta')
ax.plot(x * 1e3, perfil_simulado_norm, 'b--', linewidth=2, label='Perfil de Simulación AS')

# Personalización del gráfico
ax.set_title(f"Validación de Perfiles a z = {z*100} cm", fontsize=16)
ax.set_xlabel("Posición radial (mm)", fontsize=12)
ax.set_ylabel("Intensidad Normalizada", fontsize=12)
ax.grid(True)
ax.legend(fontsize=10)

# Opcional: ajustar el zoom en la zona de interés
# Se utiliza el radio inicial del haz (w0) para definir el límite del zoom
zoom_limit = 3 * w0 * 1e3 # Un poco más amplio que el límite anterior
ax.set_xlim(-zoom_limit, zoom_limit)

fig.tight_layout()
plt.show()