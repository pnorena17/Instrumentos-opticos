import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

#Primero leemos la imagen en la ruta y la convierte en una matriz MxM
ruta=r"C:\Users\pauli\OneDrive\Documents\Universidad\Instrumentos-opticos\practica_1\punto4\Resultados\fibra1_2.jpg"

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

#Tamaño del pixel del detector
dx = 1.85e-6 # tamaño de pixel (1.85 um)

N = M

#Hacemos la operación para rellenar de 0 la matriz NxN por fuera de la MxM
matriz_con_relleno = np.zeros((N, N), dtype=complex)
min_indice = (N-M)//2
matriz_con_relleno[min_indice : min_indice + M, min_indice : min_indice + M] = campo_detector

#### Creamos las Variables
long_onda = 632.8e-9 #633 nm
k = (2*np.pi)/long_onda

L = dx*N # dimensiones del sensor
df = 1/L # correspondiente en el espectro

#Valores para ajustar
z_fibra_a_detector = 0.0315  # distancia de la fibra al detector (3 cm)
z_fuente_a_detector = z_fibra_a_detector + 0.028  #7 cm

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
campo_corregido = matriz_con_relleno #* fase_esferica_correccion

#### Hallemos A_0 (Espectro Angular)
A_0 = np.fft.fft2(campo_corregido)
A_0sh = np.fft.fftshift(A_0)


#### Hallemos A (Propagación del Espectro Angular en el dominio espectral)

argumento_raiz = (2 * np.pi)**2 * ((1. / long_onda)**2 - Fx** 2 - Fy** 2)

#Verificamos que usemos las ondas propagantes
tmp = np.sqrt(np.abs(argumento_raiz))
kz = np.where(argumento_raiz >= 0, tmp, 1j*tmp)

# --- RANGO DE Z ---
z1, z2 = 0.034, 0.04 # metros (ejemplo: 5 cm a 15 cm)
zs = np.linspace(z1, z2, 8)

fig, axes = plt.subplots(2, 4, figsize=(15, 6))
axes = axes.ravel()

for idx, z in enumerate(zs):

    A = A_0sh * (np.exp(-1j * z * kz))
    A_ishift = np.fft.ifftshift(A)

    #### Hallemos el campo de salida U
    U = (np.fft.ifft2(A_ishift))


    #### Grafiquemos
    intensidad_norm = np.abs(U)**2
    intensidad_norm /= np.max(intensidad_norm)

    ax = axes[idx]
    ax.imshow(intensidad_norm, cmap='gray', extent=[-L/2, L/2, -L/2, L/2])
    ax.set_title(f"z={z*100:.3f} cm", fontsize=8)
    ax.axis("off")

fig.tight_layout()
plt.show()