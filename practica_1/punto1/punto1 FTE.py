import numpy as np
import matplotlib.pyplot as plt

#### Creamos las Variables

long_onda = 633e-9 #633 nm
k = (2*np.pi)/long_onda


N = 1080 # pixeles de la camara
dx = 3.7e-6 # tamaño de pixel (3.7 um)
L = dx*N # dimensiones del sensor
df = 1/L # correspondiente en el espectro

z = 0.02 # distancia de la abertura al detector (4 cm)

# Condiciones de buen muestreo

z_max = N*(dx**2)/long_onda
print(z_max)
assert z <= z_max, "No cumple el criterio de z para FTE"


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


#### Abertura Circular

r_0 = 1e-3 # 2mm
#abertura = (X**2 + Y**2) <= r_0**2
#U_0 = abertura.astype(np.complex128)


#### Abertura Cuadrada

l = 2e-3  # Longitud de la abertura
abertura = (np.abs(X) <= l / 2) & (np.abs(Y) <= l / 2)
U_0 = abertura.astype(np.complex128)

#### Hallemos A_0 (Espectro Angular)

A_0 = np.fft.fft2(U_0)
A_0sh = np.fft.fftshift(A_0)


#### Hallemos A (Propagación del Espectro Angular en el dominio espectral)

argumento_raiz = (2 * np.pi)**2 * ((1. / long_onda) ** 2 - Fx ** 2 - Fy ** 2)

#Verificamos que usemos las ondas propagantes
tmp = np.sqrt(np.abs(argumento_raiz))
kz = np.where(argumento_raiz >= 0, tmp, 1j*tmp)

A = A_0sh * (np.exp(1j * z * kz))
A_ishift = np.fft.ifftshift(A)

#### Hallemos el campo de salida U

U = (np.fft.ifft2(A_ishift))

##### Hallemos la intensidad

intensidad = abs(U)**2
max_intensidad = np.max(intensidad)

if max_intensidad > 0:
    intensidad_log = np.log1p(intensidad / max_intensidad * 100)
    intensidad_norm = intensidad_log / np.max(intensidad_log)
else:
    intensidad_norm = intensidad

intensidad_norm= intensidad/max_intensidad

#### Grafiquemos

# Aplicamos escala logarítmica (para visualizar detalles en zonas de baja intensidad)
intensidad_log = np.log10(intensidad/max_intensidad + 1e-6)   #Se suma 1 a la intensidad para evitar log(0), que es -infinito

fig, ax = plt.subplots(1,2,figsize=(10,6))

extent = [-L/2 * 1e3, L/2 * 1e3, -L/2 * 1e3, L/2 * 1e3]
im0 = ax[0].imshow(np.abs(abertura), cmap='gray', extent=extent)
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