
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

#Primero leemos la imagen en la ruta y la convierte en una matriz MxM
ruta=r"C:\Users\pauli\OneDrive\Documents\Universidad\Instrumentos-opticos\practica_1\punto3\Resultados\pez0_mean.tif"

img = Image.open(ruta).convert("L") #la convertimos a blanco y negro, Objeto Image (4000x3000)
arr = np.array(img)/255.0 #la normalizamos [0,1]

M_size = np.shape((arr))

if M_size[0] != M_size[1]:
    M = max(M_size[0],M_size[1])
    imagen = np.zeros((M,M)) 
    imagen[int((M-M_size[0])/2) : int((M-M_size[0])/2) + M_size[0], int((M-M_size[1])/2) : int((M-M_size[1])/2) + M_size[1]] = arr
else:
    M = M_size[0]
    imagen = arr

campo_detector = np.sqrt(imagen).astype(complex)

#Definimos las variables con las que vamos a trabajar (Detector DCC1545M 1280 x 1024, Pixel size 5.2 µm, Square)
##Variables ya establecidas 
long_de_onda = 633e-9  # Longitud de onda en metros (633 nm)
N = 2048    #Resolución de pixeles 
dx = 5.2e-6 #(en metros) Pixel size del detector (5.2 um)

##Valores para ajustar
z = 0.110                           # distancia de la fibra al detector (3 cm)

##Variables de la abertura, ya están establecidad
l = 5.8e-3    #(en metros) 5.8 mm
dx_0 = l/M
L = N*dx_0

#Verificaciones antes de iniciar el cálculo
z_min = M*(dx_0**2)/long_de_onda #(en metros) Distancia mínima de la pantalla para que podamos usar la Transformada de Fresnel
print(z_min)
assert z > z_min, "No cumple el criterio de z para TF"

##################  MÉTODO POR TRANSFORMADA DE FRESNEL    ######################################################

#Creamos el espacio físico de la pantalla centrado en (N/2,N/2)
n = np.linspace(-N/2, (N/2) - 1, N) * dx
m = np.linspace(-N/2, (N/2) - 1, N) * dx
X, Y = np.meshgrid(n, m)  #Cambio de variable para evitar confusiones con N y M

#Hacemos la operación para rellenar de 0 la matriz NxN por fuera de la MxM
matriz_con_relleno = np.zeros((N, N), dtype=complex)
min_indice = (N-M)//2
matriz_con_relleno[min_indice : min_indice + M, min_indice : min_indice + M] = campo_detector

k = 2*np.pi/long_de_onda

fase_cuadratica_salida = (np.exp(-1j*k*z) * (1j*long_de_onda*z)) * np.exp( -1j*(k/(2*z)) * ((X)**2 +(Y)**2)) 

campo_difraccion = matriz_con_relleno * fase_cuadratica_salida        #Este es U[n,m,z]

#Ahora, debemos realizar la transformada de Fourier inversa de 2 dimensiones
campo_sin_FFT = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(campo_difraccion))) #/ dx**2

#Creamos el espacio físico de la abertura centrada en (N/2,N/2)
n_0 = (np.arange(N) - N/2) * dx_0
m_0 = (np.arange(N) - N/2) * dx_0
N_0, M_0 = np.meshgrid(n_0, m_0)

#Calculamos la matriz de fase cuadrática

fase_cuadratica_entrada = np.exp(-1j * (k / (2*z)) * ((N_0)**2 + ((M_0)**2)))

#Multiplicación campo de entrada por la fase de entrada
campo_en_apertura = campo_sin_FFT * fase_cuadratica_entrada    #Este es U'[n_0,m_0,0]

#Calculamos la intensidad para graficar el patrón
intensidad = abs(campo_en_apertura)**2
max_intensidad = np.max(intensidad)     #Encontramos la intensidad máxima para normalizar

intensidad_norm = intensidad/max_intensidad

#Graficamos
fig, ax = plt.subplots(1,2,figsize=(10,5))

extent = [-L/2, L/2, -L/2, L/2]
im0 = ax[0].imshow(intensidad_norm, cmap='gray', extent=extent)
ax[0].set_title("Plano de Difracción")
ax[0].set_xlabel("x en plano de difracción (m)")
ax[0].set_ylabel("y en plano de difracción (m)")
ax[0].set_facecolor('black') 
ax[0].set_aspect('equal')

extent = [m.min(), m.max(), n.min(), n.max()]
im = ax[1].imshow(np.abs(matriz_con_relleno), extent=extent, cmap="gray")
ax[1].set_title("Patrón de Difracción")
ax[1].set_xlabel("x en plano de observación (m)")
ax[1].set_ylabel("y en plano de observación (m)")

plt.colorbar(im, ax=ax[1], label="Intensidad normalizada")
# Mostrar gráficos
plt.tight_layout()
plt.show()


