import numpy as np
import matplotlib.pyplot as plt

#En primer lugar, definimos la abertura y longitud de onda
long_de_onda = 633*(10**(-9)) #(en metros) Usamos la longitud de onda del  rojo: 650 nm
#Por decir, usaremos de abertura un cuadrado de lado l
l = 1*(10**(-3)) #(en metros) Usamos dimesión máxima: 1 mm
z_max = (l/2)**2/(long_de_onda) #(en metros) Distancia máxima de la pantalla, para que cumpla criterio de frenel
#z = z_max - 0.050 #(En metros) La disminuimos 5 cm para evitar criticidad.
z=1
#PODRÍAMOS REVISAR QUE z SEA POSITIVO, POR SI ALGO

##################  MÉTODO POR TRANSFORMADA DE FRESNEL    ######################################################33

# Ahora, crearemos las variables que necesitamos
N_f = (l/2)**2/(long_de_onda*z) #Numero de Fresnel, deber ser mayor a 1 para que cumpla la aproximación
M = 2**8 #Criterio aliasing: M > 4N_f


assert M > 4*N_f, "No cumple ele criterio de Aliasing"

Q = 5 #Debe ser mayor a q y depende de el orden de interpolacion
N = int(Q*M) #Numero de muestras totales ? (CREO QUE TIENE QUE SER UNA POTENCIA DE 2)

#Campo de entrada
dx_entrada = l/M #(en metros) Dividimos la dimensión máxima de la abertura entre el numero de muestras
L = N * dx_entrada

#Creamos la matriz MxM centrada en (M/2,M/2)
   #Esta es U[n_0,m_0,0]

k = (np.arange(M) - M/2)* dx_entrada
p = (np.arange(M) - M/2) * dx_entrada
K, P = np.meshgrid(k, p)

lineas_mm = 10
p = 1/(lineas_mm / 1e-3)
fraccion_abierto = 0.5
grating_x = (((K % p) < (fraccion_abierto * p))).astype(float)
abertura = grating_x.astype(complex) 

#Calculamos la matriz de fase cuadrática
fase_cuadratica_entrada = np.exp(1j * (np.pi*4*N_f / (M**2)) * ((K-M/2)**2 + ((P-M/2)**2)))

#Multiplicación punto a punto
campo_en_apertura = abertura * fase_cuadratica_entrada  #Este es U'[n_0,m_0,z]

padded_array = np.zeros((N, N), dtype=complex)
min_index = (N-M)//2
padded_array[min_index : min_index + M, min_index : min_index + M] = campo_en_apertura

#Ahora, debemos realizar la transformada de Fourier de 2 dimensiones
difraccion_fft = np.fft.fft2(padded_array)
centrar_fft = np.fft.fftshift(difraccion_fft) #Esta es U"[n,m,z]

x = np.linspace(-N/2, (N/2) - 1, N) * dx_entrada
y = np.linspace(-N/2, (N/2) - 1, N) * dx_entrada
X, Y = np.meshgrid(x, y)


fase_cuadratica_salida = np.exp(1j * (np.pi / (4 * Q**2 * N_f)) * ((X-N/2)**2 + (Y-N/2)**2))
campo_difraccion = centrar_fft * fase_cuadratica_salida #Este es U[n,m,z]

intensidad = abs(centrar_fft)**2
max_intensidad = np.max(intensidad)
if max_intensidad > 0:
    intensidad_log = np.log1p(intensidad / max_intensidad * 100)
    intensidad_norm = intensidad_log / np.max(intensidad_log)
else:
    intensidad_norm = intensidad

#Aplicamos escala logarítmica (para visualizar detalles en zonas de baja intensidad)
intensidad_log = np.log10(intensidad/max_intensidad + 1e-6)   #Se suma 1 a la intensidad para evitar log(0), que es -infinito

fig, ax = plt.subplots(1,2,figsize=(12,6))

extent = [-L/2 * 1e3, L/2 * 1e3, -L/2 * 1e3, L/2 * 1e3]
im0 = ax[0].imshow(np.abs(padded_array), cmap='gray', extent=extent)
ax[0].set_title("Plano de Difracción")
ax[0].set_xlabel("x en plano de difracción (m)")
ax[0].set_ylabel("y en plano de difracción (m)")
ax[0].set_facecolor('black') 
ax[0].set_aspect('equal')


extent = [x.min(), x.max(), y.min(), y.max()]
im = ax[1].imshow(intensidad_log, extent=extent, cmap="gray")
ax[1].set_title("Patrón de Difracción")
ax[1].set_xlabel("x en plano de observación (m)")
ax[1].set_ylabel("y en plano de observación (m)")

plt.colorbar(im, ax=ax[1], label="Intensidad normalizada")
# Mostrar gráficos
plt.tight_layout()
plt.show()



