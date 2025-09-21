import numpy as np
import matplotlib.pyplot as plt

#### Creamos las Variables

long_onda = 633e-9 
k = (2*np.pi)/long_onda


N = 800 # pixeles de la camara
L = 10 # dimensiones del sensor
dx = L/N # tamaño de pixel
df = 1/L # correspondiente en el espectro

z = 50000 # distancia de la abertura al detector

# Condiciones de buen muestreo

z_max = N*(dx**2)/long_onda
assert z <= z_max, "No cumple el criterio de z para TF"
print(z_max)

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

#r_0 = 1 # 2mm
#abertura = (X**2 + Y**2) <= r_0**2

#U_0 = abertura.astype(np.complex128)


#### Abertura Cuadrada

l = 2  # Longitud de la abertura
abertura = (np.abs(X) <= l / 2) & (np.abs(Y) <= l / 2)
U_0 = abertura.astype(np.complex128)

#### Hallemos A_0

A_0 = np.fft.fft2(U_0) * (dx**2)
A_0sh = np.fft.fftshift(A_0)


#### Hallemos A

argumento_raiz = 1 - ((long_onda)**2 * (Fx**2 + Fy**2))
A = A_0 * (np.exp(1j * z * k * np.sqrt(argumento_raiz)))
A_shift = np.fft.ifftshift(A)

#### Hallemos el campo de salida U

U = (np.fft.ifft2(A_shift)) * ((df)**2)
U_shift = np.fft.fftshift(U)

##### Hallemos la intensidad

intensidad = abs(U)**2
max_intensidad = np.max(intensidad)
intensidad_norm= intensidad/max_intensidad

if max_intensidad > 0:
    intensidad_log = np.log1p(intensidad / max_intensidad * 100)
    intensidad_norm = intensidad_log / np.max(intensidad_log)
else:
    intensidad_norm = intensidad

#### Grafiquemos

# Aplicamos escala logarítmica (para visualizar detalles en zonas de baja intensidad)
intensidad_log = np.log10(intensidad/max_intensidad + 1e-6)   #Se suma 1 a la intensidad para evitar log(0), que es -infinito

fig, ax = plt.subplots(1,2,figsize=(12,6))

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