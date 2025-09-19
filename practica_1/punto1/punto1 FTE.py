import numpy as np
import matplotlib.pyplot as plt

#En primer lugar, definimos la abertura y longitud de 


long_de_onda = 633*(10**(-9)) #(en metros) Usamos la longitud de onda del  rojo: 650 nm
N = 1080 #pixeles de la camara
dx = 2.9*(10**(-6)) # Espaciado en el dominio espacial 2,9um(cuadrada)
L = N*dx #Dimensiones del sensor

l = 0.04 #(diemnsiones de la apertura cuadrada) Usamos dimensión máxima: 1 cm
M = int((l/L)*N)
print(M)
df = 1/L


##################  MÉTODO POR FUNCION DE TRANSFERENCIA EXACTA    ######################################################33


#if M%2 == 0:
 #   N = 12*M
#else:
#    N = 12*M +1 
     # Espaciado en el dominio de la frecuencia (cuadrada)
     
z = 0.04
z_max = M*(dx**2)/long_de_onda
print(z_max, M)
#assert z <= z_max, "No cumple el criterio de z para TF"


    
abertura = np.ones((M,M), dtype=complex)
padded_array = np.zeros((N, N), dtype=complex)
min_index = (N-M)//2
padded_array[min_index : min_index + M, min_index : min_index + M] = abertura

# Discretizacion dominio espacial

n = (np.arange(N) - N/2)
m = (np.arange(N) - N/2)

x = n*dx
y = m*dx

X, Y = np.meshgrid(n, m)

# Discretizacion espectro
 
p = (np.arange(N) - N/2) 
q = (np.arange(N) - N/2) 

fx = p*df
fy = q*df
P, Q = np.meshgrid(p, q)


k = 2*np.pi/long_de_onda #numero de onda

#Calculamos la matriz de fase cuadrática

#Multiplicación campo de entrada por la fase de entrada
campo_en_apertura = padded_array     

f_max = M*df #Criterio de Aliasing

#Transformada de Fourier
difraccion_fft = np.fft.fft2(campo_en_apertura)* dx**2
centrar_fft = dx**2 * np.fft.fftshift(difraccion_fft)

#Manejo ondas evanescentes
argumento_raiz = 1 - (((long_de_onda*df)**2) * (P**2 + Q**2))
print(1/long_de_onda)
print(M*df)
# Filtro para mantener solo las frecuencias que se propagan (argumento >= 0)
filtro_propagacion = (argumento_raiz >= 0)

# Inicializamos H con ceros (complejos)
FuncionTransfer = np.zeros((N, N), dtype=complex)
# Calculamos H solo para las frecuencias que se propagan
FuncionTransfer[filtro_propagacion] = np.exp(1j*z*k*np.sqrt(argumento_raiz[filtro_propagacion]))


A = centrar_fft * FuncionTransfer
shift_A = np.fft.ifftshift(A)   #shift preparado para hacer la inversa
CampoSalida = df**2 * np.fft.ifft2(shift_A) #inversa de fourier


intensidad = abs(CampoSalida)**2
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