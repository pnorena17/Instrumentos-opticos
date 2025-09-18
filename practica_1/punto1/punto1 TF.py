import numpy as np
import matplotlib.pyplot as plt

#En primer lugar, definimos la abertura y longitud de onda
long_de_onda = 650*(10**(-9)) #(en metros) Usamos la longitud de onda del  rojo: 650 nm
#Por decir, usaremos de abertura un cuadrado de lado l
l = 1*(10**(-5)) #(en metros) Usamos dimesión máxima: 1 mm
z_max = (l/2)**2/(long_de_onda) #(en metros) Distancia máxima de la pantalla, para que cumpla criterio de frenel
#z = z_max - 0.050 #(En metros) La disminuimos 5 cm para evitar criticidad.
z=0.1
#PODRÍAMOS REVISAR QUE z SEA POSITIVO, POR SI ALGO

##################  MÉTODO POR TRANSFORMADA DE FRESNEL    ######################################################33

# Ahora, crearemos las variables que necesitamos
N_f = (l/2)**2/(long_de_onda*z) #Numero de Fresnel, deber ser mayor a 1 para que cumpla la aproximación

M = 64 #Criterio aliasing: M > 4N_f

assert M > 4*N_f, "No cumple ele criterio de Aliasing"
    

Q = 5 #Debe ser mayor a q y depende de el orden de interpolacion
N = int(Q*M) #Numero de muestras totales ? (CREO QUE TIENE QUE SER UNA POTENCIA DE 2)

#Campo de entrada
dx_entrada = l/M #(en metros) Dividimos la dimensión máxima de la abertura entre el numero de muestras
L = N * dx_entrada

#Creamos la matriz MxM centrada en (M/2,M/2)
abertura = np.ones((M,M), dtype=complex)    #Esta es U[n_0,m_0,0]

k = (np.arange(M) - M/2) * dx_entrada
p = (np.arange(M) - M/2) * dx_entrada
K, P = np.meshgrid(k, p)

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

dx_salida = long_de_onda*z/(dx_entrada*N)

x = np.linspace(-N/2, (N/2) - 1, N) * dx_salida
y = np.linspace(-N/2, (N/2) - 1, N) * dx_salida
X, Y = np.meshgrid(x, y)

fase_cuadratica_salida = np.exp(1j * (np.pi / (4 * Q**2 * N_f)) * ((X-N/2)**2 + (Y-N/2)**2))/(1j*long_de_onda*z)
campo_difraccion = centrar_fft * fase_cuadratica_salida #Este es U[n,m,z]

intensidad = abs(campo_difraccion)**2
max_intensidad = np.max(intensidad)
if max_intensidad > 0:
    intensidad_log = np.log1p(intensidad / max_intensidad * 10)
    intensidad_norm = intensidad_log / np.max(intensidad_log)
else:
    intensidad_norm = intensidad

#Aplicamos escala logarítmica (para visualizar detalles en zonas de baja intensidad)
intensidad_log = np.log10(intensidad/max_intensidad + 1e-6)   #Se suma 1 a la intensidad para evitar log(0), que es -infinito

fig, ax = plt.subplots(1,2,figsize=(12,6))

extent = [-l/2 * 1e3, l/2 * 1e3, -l/2 * 1e3, l/2 * 1e3]
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

center_index = N // 2
perfil_intensidad = intensidad[center_index, :]

# Crear las coordenadas físicas para el eje x
# Las coordenadas x deben estar centradas en 0.
dx_salida = (long_de_onda * z) / (N * dx_entrada)
x_coords = np.arange(N) * dx_salida - (N * dx_salida) / 2

# Gráfico del perfil de intensidad en escala lineal
plt.figure(figsize=(10, 5))
plt.plot(x_coords, perfil_intensidad)
plt.title('Perfil de Intensidad a lo largo de y=0 (Escala Lineal)')
plt.xlabel('Posición en x (m)')
plt.ylabel('Intensidad')
plt.grid(True)
plt.show()

# Gráfico del perfil de intensidad en escala logarítmica
perfil_intensidad_log = np.log10(perfil_intensidad / np.max(perfil_intensidad) + 1e-6)
plt.figure(figsize=(10, 5))
plt.plot(x_coords, perfil_intensidad_log)
plt.title('Perfil de Intensidad a lo largo de y=0 (Escala Logarítmica)')
plt.xlabel('Posición en x (m)')
plt.ylabel('log10(Intensidad)')
plt.grid(True)
plt.show()

