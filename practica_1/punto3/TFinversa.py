import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# --- LECTURA DE IMAGEN ---
ruta = r"C:\Users\pauli\OneDrive\Documents\Universidad\Instrumentos-opticos\practica_1\punto3\Resultados\pez2_mean.tif"
img = Image.open(ruta).convert("L") 
arr = np.array(img)/255.0 

M_size = np.shape(arr)
if M_size[0] != M_size[1]:
    M = max(M_size[0],M_size[1])
    imagen = np.zeros((M,M)) 
    imagen[int((M-M_size[0])/2):int((M-M_size[0])/2)+M_size[0], int((M-M_size[1])/2):int((M-M_size[1])/2)+M_size[1]] = arr
else:
    M = M_size[0]
    imagen = arr

campo_detector = np.sqrt(imagen).astype(complex)

# --- PARÁMETROS FÍSICOS ---
long_de_onda = 633e-9  # (m)
N = 2048
dx = 5.2e-6            # tamaño pixel detector (m)

l = 5.8e-3
dx_0 = l/M
L = N*dx_0

# Espacios de cálculo
n = np.linspace(-N/2, N/2 - 1, N) * dx
m = np.linspace(-N/2, N/2 - 1, N) * dx
X, Y = np.meshgrid(n, m)

matriz_con_relleno = np.zeros((N, N), dtype=complex)
min_indice = (N-M)//2
matriz_con_relleno[min_indice:min_indice+M, min_indice:min_indice+M] = campo_detector

n_0 = (np.arange(N) - N/2) * dx_0
m_0 = (np.arange(N) - N/2) * dx_0
N_0, M_0 = np.meshgrid(n_0, m_0)

k = 2*np.pi/long_de_onda

# --- RANGO DE Z ---
z1, z2 = 0.267, 0.275  # metros (ejemplo: 5 cm a 15 cm)
zs = np.linspace(z1, z2, 8)

fig, axes = plt.subplots(2, 4, figsize=(15, 6))
axes = axes.ravel()

for idx, z in enumerate(zs):
    fase_cuadratica_salida = (np.exp(-1j*k*z) * (1j*long_de_onda*z)) * np.exp(-1j*(k/(2*z)) * (X**2 + Y**2))

    campo_difraccion = matriz_con_relleno * fase_cuadratica_salida
    campo_sin_FFT = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(campo_difraccion)))

    fase_cuadratica_entrada = np.exp(-1j * (k/(2*z)) * (N_0**2 + M_0**2))
    campo_en_apertura = campo_sin_FFT * fase_cuadratica_entrada

    intensidad_norm = np.abs(campo_en_apertura)**2
    intensidad_norm /= np.max(intensidad_norm)

    ax = axes[idx]
    ax.imshow(intensidad_norm, cmap='gray', extent=[-L/2, L/2, -L/2, L/2])
    ax.set_title(f"z={z*100:.3f} cm", fontsize=8)
    ax.axis("off")

plt.tight_layout()
plt.show()

