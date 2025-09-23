import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

#Primero leemos la imagen en la ruta y la convierte en una matriz MxM
ruta=r"C:\Users\pauli\OneDrive\Documents\Universidad\Instrumentos-opticos\practica_1\punto4\Resultados\fibra0_1.jpg"

img = Image.open(ruta).convert("L") #la convertimos a blanco y negro
arr = np.array(img)/255.0 #la normalizamos [0,1]
N_size = np.shape((arr))

if N_size[0] != N_size[1]:
    N = min(N_size[0],N_size[1])
    imagen = arr[int((-N+N_size[0])/2) : int((-N+N_size[0])/2) + N, int((-N+N_size[1])/2) : int((-N+N_size[1])/2) + N]
    
else:
    N = N_size[0]
    imagen = arr

campo_detector = np.sqrt(imagen).astype(complex)

#Tamaño del pixel del detector
dx = 1.85e-6 # tamaño de pixel (1.85 um)

#### Creamos las Variables
long_onda = 633e-9 #633 nm
k = (2*np.pi)/long_onda

L = dx*N # dimensiones del sensor
df = 1/L # correspondiente en el espectro

#Valores para ajustar
z_fibra_a_detector = 0.03  # distancia de la fibra al detector (3 cm)
z_fuente_a_detector = 0.08  #7 cm

# Condiciones de buen muestreo

z_max = N*(dx**2)/long_onda
print(z_max)
#assert z_fibra_a_detector <= z_max, "No cumple el criterio de z para FTE"


######### Coordenadas Espaciales

n = np.arange(N) - N//2
m = np.arange(N) - N//2

x = n*dx
y = m*dx
X,Y = np.meshgrid(x,y)

####### Espectro

p = np.arange(N) - N//2
q = np.arange(N) - N//2

fx = p*df
fy = q*df
Fx,Fy = np.meshgrid(fx, fy)

# Creamos la fase esférica de corrección. Esta es la fase que debemos "quitar" de la imagen capturada.
fase_esferica_correccion = np.exp(-1j * k * (X**2 + Y**2) / (2 * z_fuente_a_detector))

# Corregimos el campo multiplicándolo por la fase de corrección
campo_corregido = campo_detector * fase_esferica_correccion

#### Hallemos A_0 (Espectro Angular)

A_0 = np.fft.fft2(campo_corregido)
A_0sh = np.fft.fftshift(A_0)


#### Hallemos A (Propagación del Espectro Angular en el dominio espectral)

argumento_raiz = (2 * np.pi)**2 * ((1. / long_onda) ** 2 - Fx ** 2 - Fy ** 2)

#Verificamos que usemos las ondas propagantes
tmp = np.sqrt(np.abs(argumento_raiz))
kz = np.where(argumento_raiz >= 0, tmp, 1j*tmp)

A = A_0sh * (np.exp(-1j * z_fibra_a_detector * kz))
A_ishift = np.fft.ifftshift(A)

#### Hallemos el campo de salida U

U = (np.fft.ifft2(A_ishift))


#### Grafiquemos

fig, ax = plt.subplots(1,2,figsize=(10,6))

extent = [-L/2 * 1e3, L/2 * 1e3, -L/2 * 1e3, L/2 * 1e3]
im0 = ax[0].imshow((np.abs(U)**2)*100, cmap='gray', extent=extent)
ax[0].set_title("Plano de la Apertura", fontsize=14)
ax[0].set_xlabel("x (mm)", fontsize=12)
ax[0].set_ylabel("y (mm)", fontsize=12)
ax[0].set_aspect('equal')

extent = [-L/2 * 1e3, L/2 * 1e3, -L/2 * 1e3, L/2 * 1e3]
im1 = ax[1].imshow(imagen, extent=extent, cmap="gray")
ax[1].set_title("Patrón de Difracción")
ax[1].set_xlabel("x en plano de observación (mm)")
ax[1].set_ylabel("y en plano de observación (mm)")
plt.colorbar(im1, ax=ax[1], label="Intensidad normalizada")

fig.tight_layout()
plt.show()