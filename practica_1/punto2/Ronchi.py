import numpy as np
import matplotlib.pyplot as plt

#Definimos las variables con las que vamos a trabajar
##Variables ya establecidas 
long_de_onda = 633e-9 #(en metros) Usamos la longitud de onda del  rojo: 633 nm
N = 1200    #Resolución mínima de pixeles del detector DFM 37UX290-ML
dx = 3e-6 #(en metros) Pixel size del detector (2.9 um)

##Variables modificables
z = 0.0316     #(en metros) Distancia entre pantalla y abertura

##Variables de la abertura
#Por ejemplo, usaremos de abertura un cuadrado de lado l
l = 1e-3    #(en metros) Usamos dimesión máxima (1 mm)
dx_0 = long_de_onda*z/(N*dx)  #(en metros) Tamaño de pixel en nuestra abertura
M = 2*int(l/dx_0)   #Muestreo de nuestra abertura
L = N*dx_0

print(M)

#Verificaciones antes de iniciar el cálculo
z_min = M*(dx_0**2)/long_de_onda #(en metros) Distancia mínima de la pantalla para que podamos usar la Transformada de Fresnel
#assert z > z_min, "No cumple el criterio de z para TF"

##################  MÉTODO POR TRANSFORMADA DE FRESNEL    ######################################################

#Creamos el espacio físico de la rejilla infinita centrada en (N/2, N/2)
n_0 = (np.arange(N) - N/2) * dx_0
m_0 = (np.arange(N) - N/2) * dx_0
N_0, M_0 = np.meshgrid(n_0, m_0)

#Creamos la matriz MxM para la abertura
iluminacion = np.ones((N,N), dtype=complex)                  #Esta es U[n_0,m_0,0]

#Creamos la Rejilla Ronchi
lineas_mm = 10
periodo = 1/(lineas_mm / 1e-3)
fraccion_abierto = 0.5
rejilla_x = (((N_0 % periodo) < (fraccion_abierto * periodo))).astype(float)
transmitancia = rejilla_x.astype(complex) 

#Modificamos la onda de entrada por la transmitancia encontrada
campo_entrada = iluminacion * transmitancia

#Calculamos la matriz de fase cuadrática
k = 2*np.pi/long_de_onda
fase_cuadratica_entrada = np.exp(1j * (k / (2*z)) * ((N_0)**2 + ((M_0)**2)))

#Multiplicación campo de entrada por la fase de entrada
campo_en_apertura = campo_entrada * fase_cuadratica_entrada    #Este es U'[n_0,m_0,0]

#Ahora, debemos realizar la transformada de Fourier de 2 dimensiones
difraccion_fft = np.fft.fft2(campo_en_apertura)
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
im0 = ax[0].imshow(np.abs(campo_en_apertura), cmap='gray', extent=extent)
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
