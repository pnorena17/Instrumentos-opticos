
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


#Primero leemos la imagen en la ruta y la convierte en una matriz MxM
ruta=r"C:\Users\david\OneDrive\Desktop\Universidad\Semestre 11\Instrumentos Opticos\Transm_E06.png"

img = Image.open(ruta).convert("L") #la convertimos a blanco y negro
arr = np.array(img)/255.0 #la normalizamos [0,1]
umbral = 0.5
M_size = np.shape((arr))
if M_size[0] =! M_size[1]:
    M = min(M_size[0],M_size[1]])
    
else:
    M = M_size[0]
transmitancia = (arr > umbral).astype(complex)#comparamos los valores de pixel con el umbral(nos da una matriz booleana)


#Definimos las variables con las que vamos a trabajar
##Variables ya establecidas 
long_de_onda = 633e-9  # Longitud de onda en metros (633 nm)
N = 1080    #Resolución mínima de pixeles del detector DFM 37UX290-ML
dx = 2.9e-6 #(en metros) Pixel size del detector (2.9 um)

##Variables modificables
z = 0.2     #(en metros) Distancia entre pantalla y abertura

##Variables de la abertura, ya están establecidad
l = 5.8e-3    #(en metros) 5.8 mm
dx_0 = l/M
L = N*dx_0

#Verificaciones antes de iniciar el cálculo
z_min = M*(dx_0**2)/long_de_onda #(en metros) Distancia mínima de la pantalla para que podamos usar la Transformada de Fresnel
assert z > z_min, "No cumple el criterio de z para TF"

##################  MÉTODO POR TRANSFORMADA DE FRESNEL    ######################################################

#Creamos el espacio físico de la abertura centrada en (M/2,M/2)
n_0 = (np.arange(M) - M/2) * dx_0
m_0 = (np.arange(M) - M/2) * dx_0
N_0, M_0 = np.meshgrid(n_0, m_0)

#Creamos la matriz MxM para la abertura
iluminacion = np.ones((M,M), dtype=complex)   

#Creamos la matriz MxM para la abertura
campo_entrada = iluminacion*transmitancia                #Esta es U[n_0,m_0,0]

#Calculamos la matriz de fase cuadrática
k = 2*np.pi/long_de_onda
fase_cuadratica_entrada = np.exp(1j * (k / 2*z) * ((N_0)**2 + ((M_0)**2)))

#Multiplicación campo de entrada por la fase de entrada
campo_en_apertura = campo_entrada * fase_cuadratica_entrada    #Este es U'[n_0,m_0,0]

#Hacemos la operación para rellenar de 0 la matriz NxN por fuera de la MxM
matriz_con_relleno = np.zeros((N, N), dtype=complex)
min_indice = (N-M)//2
matriz_con_relleno[min_indice : min_indice + M, min_indice : min_indice + M] = campo_en_apertura

#Ahora, debemos realizar la transformada de Fourier de 2 dimensiones
difraccion_fft = np.fft.fft2(matriz_con_relleno)
centrar_fft = dx**2 * np.fft.fftshift(difraccion_fft)          #Esta es U"[n,m,z]

#Creamos el espacio físico de la pantalla centrado en (N/2,N/2)
n = np.linspace(-N/2, (N/2) - 1, N) * dx
m = np.linspace(-N/2, (N/2) - 1, N) * dx
X, Y = np.meshgrid(n, m)  #Cambio de variable para evitar confusiones con N y M

fase_cuadratica_salida = (np.exp(1j*k*z) / (1j*long_de_onda*z)) * np.exp( 1j*(k/(2*z)) * ((X)**2 +(Y)**2)) 
campo_difraccion = centrar_fft * fase_cuadratica_salida        #Este es U[n,m,z]

#Calculamos la intensidad para graficar el patrón
intensidad = abs(campo_difraccion)**2
max_intensidad = np.max(intensidad)     #Encontramos la intensidad máxima para normalizar

#Buscamos una intensidad que se vea bien el patrón
if max_intensidad > 0:
    intensidad_log = np.log1p(intensidad / max_intensidad * 100)
    intensidad_norm = intensidad_log / np.max(intensidad_log)
else:
    intensidad_norm = intensidad

#Aplicamos escala logarítmica (para visualizar detalles en zonas de baja intensidad)
intensidad_log = np.log10(intensidad/max_intensidad + 1e-6)   #Se suma 1 a la intensidad para evitar log(0), que es -infinito


#Graficamos
fig, ax = plt.subplots(1,2,figsize=(10,5))

extent = [-L/2, L/2, -L/2, L/2]
im0 = ax[0].imshow(np.abs(matriz_con_relleno), cmap='gray', extent=extent)
ax[0].set_title("Plano de Difracción")
ax[0].set_xlabel("x en plano de difracción (m)")
ax[0].set_ylabel("y en plano de difracción (m)")
ax[0].set_facecolor('black') 
ax[0].set_aspect('equal')

extent = [m.min(), m.max(), n.min(), n.max()]
im = ax[1].imshow(intensidad_log, extent=extent, cmap="gray")
ax[1].set_title("Patrón de Difracción")
ax[1].set_xlabel("x en plano de observación (m)")
ax[1].set_ylabel("y en plano de observación (m)")

plt.colorbar(im, ax=ax[1], label="Intensidad normalizada")
# Mostrar gráficos
plt.tight_layout()
plt.show()


