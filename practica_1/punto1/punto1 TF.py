import numpy as np
import matplotlib.pyplot as plt

#Definimos las variables con las que vamos a trabajar
##Variables ya establecidas 
long_de_onda = 650e-9 #(en metros) Usamos la longitud de onda del  rojo: 650 nm
N = 1080    #Resolución mínima de pixeles del detector DFM 37UX290-ML
dx = 2.9e-6 #(en metros) Pixel size del detector (2.9 um)

##Variables modificables
z = 0.1     #(en metros) Distancia entre pantalla y abertura

##Variables de la abertura
#Por ejemplo, usaremos de abertura un cuadrado de lado l
l = 1e-5    #(en metros) Usamos dimesión máxima (1 mm)
M = N//4     #Muestreo de nuestra abertura
dx_0 = l/M  #(en metros) Tamaño de pixel en nuestra abertura

#Verificaciones antes de iniciar el cálculo
z_min = M*(dx_0**2)/long_de_onda #(en metros) Distancia mínima de la pantalla para que podamos usar la Transformada de Fresnel
assert z > z_min, "No cumple el criterio de z para TF"

##################  MÉTODO POR TRANSFORMADA DE FRESNEL    ######################################################33

#Creamos el espacio físico de la abertura centrada en (M/2,M/2)
n_0 = (np.arange(M) - M/2) * dx_0
m_0 = (np.arange(M) - M/2) * dx_0
N_0, M_0 = np.meshgrid(n_0, m_0)

#Creamos la matriz MxM para la abertura
campo_entrada = np.ones((M,M), dtype=complex)                  #Esta es U[n_0,m_0,0]

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

extent = [-l/2 * 1e3, l/2 * 1e3, -l/2 * 1e3, l/2 * 1e3]
im0 = ax[0].imshow(np.abs(matriz_con_relleno), cmap='gray', extent=extent)
ax[0].set_title("Plano de Difracción")
ax[0].set_xlabel("x en plano de difracción (m)")
ax[0].set_ylabel("y en plano de difracción (m)")
ax[0].set_facecolor('black') 
ax[0].set_aspect('equal')

extent = [m.min(), m.max(), n.min(), n.max()]
im = ax[1].imshow(intensidad_norm, extent=extent, cmap="gray")
ax[1].set_title("Patrón de Difracción")
ax[1].set_xlabel("x en plano de observación (m)")
ax[1].set_ylabel("y en plano de observación (m)")

plt.colorbar(im, ax=ax[1], label="Intensidad normalizada")
# Mostrar gráficos
plt.tight_layout()
plt.show()

center_index = N // 2
perfil_intensidad = intensidad[center_index, :]

# Gráfico del perfil de intensidad en escala lineal
plt.figure(figsize=(10, 5))
plt.plot(m, perfil_intensidad)
plt.title('Perfil de Intensidad a lo largo de y=0 (Escala Lineal)')
plt.xlabel('Posición en x (m)')
plt.ylabel('Intensidad')
plt.grid(True)
plt.show()

# Gráfico del perfil de intensidad en escala logarítmica
perfil_intensidad_log = np.log10(perfil_intensidad / np.max(perfil_intensidad) + 1e-6)
plt.figure(figsize=(10, 5))
plt.plot(m, perfil_intensidad_log)
plt.title('Perfil de Intensidad a lo largo de y=0 (Escala Logarítmica)')
plt.xlabel('Posición en x (m)')
plt.ylabel('log10(Intensidad)')
plt.grid(True)
plt.show()

