import numpy as np
import matplotlib.pyplot as plt

#En primer lugar, definimos la abertura y longitud de onda
long_de_onda = 650*(10**(-9)) #(en metros) Usamos la longitud de onda del  rojo: 650 nm
#Por decir, usaremos de abertura un cuadrado de lado l
l = 1*(10**(-3)) #(en metros) Usamos dimesión máxima: 10 cm
z = 1

##################  MÉTODO POR FUNCION DE TRANSFERENCIA EXACTA    ######################################################33

#Creamos las variables
M = 256

if M%2 == 0:
    N = 8*M
else:
    N = 8*M +1 
    
abertura = np.ones((M,M), dtype=complex)
padded_array = np.zeros((N, N), dtype=complex)
min_index = (N-M)//2
padded_array[min_index : min_index + M, min_index : min_index + M] = abertura

dx = l/M # Tamaño del píxel en la apertura
L=l*N/M # Tamaño total de la cuadrícula computacional

x = (np.arange(N) - N/2) * dx
y = (np.arange(N) - N/2) * dx
X, Y = np.meshgrid(x, y)

df_x = 1/L # Espaciado en el dominio de la frecuencia
p = (np.arange(N) - N/2) * df_x
q = (np.arange(N) - N/2) * df_x
P, Q = np.meshgrid(p, q)

f_max = M/L #Criterio de Aliasing

#Transformada de Fourier
difraccion_fft = np.fft.fft2(padded_array)
centrar_fft = np.fft.fftshift(difraccion_fft)

#Manejo ondas evanescentes
argumento_raiz = 1 - (long_de_onda**2) * (P**2 + Q**2)
# Filtro para mantener solo las frecuencias que se propagan (argumento >= 0)
filtro_propagacion = (argumento_raiz >= 0)

# Inicializamos H con ceros (complejos)
FuncionTransfer = np.zeros((N, N), dtype=complex)
# Calculamos H solo para las frecuencias que se propagan
FuncionTransfer[filtro_propagacion] = np.exp(1j*2*np.pi*z/long_de_onda * np.sqrt(argumento_raiz[filtro_propagacion]))


A = centrar_fft * FuncionTransfer
shift_A = np.fft.ifftshift(A)   #shift preparado para hacer la inversa
CampoSalida = np.fft.ifft2(shift_A) #inversa de fourier


intensidad = abs(CampoSalida)**2
max_intensidad = np.max(intensidad)
if max_intensidad > 0:
    intensidad_log = np.log1p(intensidad / max_intensidad * 1000)
    intensidad_norm = intensidad_log / np.max(intensidad_log)
else:
    intensidad_norm = intensidad

#Aplicamos escala logarítmica (para visualizar detalles en zonas de baja intensidad)
intensidad_log = np.log10(intensidad/max_intensidad + 1e-6)   #Se suma 1 a la intensidad para evitar log(0), que es -infinito

fig, ax = plt.subplots(1,2,figsize=(12,6))

extent = [-L/2 * 1e3, L/2 * 1e3, -L/2 * 1e3, L/2 * 1e3]
im0 = ax[0].imshow(np.abs(padded_array), cmap='gray', extent=extent)
ax[0].set_title("Plano de la Apertura", fontsize=14)
ax[0].set_xlabel("x (mm)", fontsize=12)
ax[0].set_ylabel("y (mm)", fontsize=12)
ax[0].set_aspect('equal')

im1 = ax[1].imshow(intensidad_norm, extent=extent, cmap="gray")
ax[1].set_title("Patrón de Difracción")
ax[1].set_xlabel("x en plano de observación (mm)")
ax[1].set_ylabel("y en plano de observación (mm)")
plt.colorbar(im1, ax=ax[1], label="Intensidad normalizada")

center_index = N // 2
perfil_intensidad = intensidad[center_index, :]

fig.tight_layout()
plt.show()