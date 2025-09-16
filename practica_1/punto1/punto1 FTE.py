import numpy as np
import matplotlib.pyplot as plt

#En primer lugar, definimos la abertura y longitud de onda
long_de_onda = 650*(10**(-9)) #(en metros) Usamos la longitud de onda del  rojo: 650 nm
#Por decir, usaremos de abertura un cuadrado de lado l
l = 1*(10**(-3)) #(en metros) Usamos dimesión máxima: 1 mm
z_max = (l/2)**2/(long_de_onda) #(en metros) Distancia máxima de la pantalla, para que cumpla criterio de frenel
#z = z_max - 0.050 #(En metros) La disminuimos 5 cm para evitar criticidad.
z=0.1
#PODRÍAMOS REVISAR QUE z SEA POSITIVO, POR SI ALGO

##################  MÉTODO POR FUNCION DE TRANSFERENCIA EXACTA    ######################################################33

#Creamos las variables


N_f = (l/2)**2/(long_de_onda*z) #Numero de Fresnel, deber ser mayor a 1 para que cumpla la aproximación
M = int((4*N_f)+20)

if M%2 == 0:
    N = 4*M
else:
    N = 4*M +1 
    
abertura = np.ones((M,M), dtype=complex)
padded_array = np.zeros((N, N), dtype=complex)
min_index = (N-M)//2
padded_array[min_index : min_index + M, min_index : min_index + M] = abertura
dx = l/M
L=l*N/M
df_x = 1/L


    

f_max = M/L #Criterio de Aliasing
#Transformada de Fourier
difraccion_fft = np.fft.fft2(padded_array)
centrar_fft = np.fft.fftshift(difraccion_fft)


x = np.arange(0, N-1, N) * dx
y = np.arange(0, N-1, N) * dx
X, Y = np.meshgrid(x, y)


p = np.linspace(0, N - 1, N) * df_x
q = np.linspace(0, N - 1, N) * df_x
P, Q = np.meshgrid(p, q)
FuncionTransfer = np.exp(1j*np.pi*z/long_de_onda * np.sqrt(1- (long_de_onda/L)**2 *((P-(N/2))**2 + (Q-(N/2))**2)))


A = centrar_fft * FuncionTransfer
shift_A = np.fft.ifftshift(A)#shift preparado para hacer la inversa
CampoSalida = np.fft.ifft2(shift_A)#inversa de fourier


intensidad = abs(CampoSalida)**2
max_intensidad = np.max(intensidad)
if max_intensidad > 0:
    intensidad_log = np.log1p(intensidad / max_intensidad * 50)
    intensidad_norm = intensidad_log / np.max(intensidad_log)
else:
    intensidad_norm = intensidad

#Aplicamos escala logarítmica (para visualizar detalles en zonas de baja intensidad)
intensidad_log = np.log10(intensidad/max_intensidad + 1e-6)   #Se suma 1 a la intensidad para evitar log(0), que es -infinito

fig, ax = plt.subplots(1,2,figsize=(12,6))

cuadrado = plt.Rectangle((1 - 10 / 2, 1 - 10 / 2), 10, 10, color = 'white')
ax[0].add_patch(cuadrado)
ax[0].set_title("Plano de Difracción")
ax[0].set_xlabel("x en plano de difracción (m)")
ax[0].set_ylabel("y en plano de difracción (m)")
ax[0].set_facecolor('black') 
ax[0].set_aspect('equal')

extent = [x.min()*z, x.max()*z, y.min()*z, y.max()*z]
im = ax[1].imshow(intensidad_log, extent=extent, cmap="gray")
ax[1].set_title("Patrón de Difracción")
ax[1].set_xlabel("x en plano de observación (m)")
ax[1].set_ylabel("y en plano de observación (m)")

