import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# Definicion de funciones a usar

def generar_estrella_siemens(physical_size, num_pixels, N_spokes):
   
    # Crear la rejilla de coordenadas 
    # Creamos vectores de coordenadas en METROS, centrados en 0.
    lin = np.linspace(-physical_size / 2, physical_size / 2, num_pixels)
    x, y = np.meshgrid(lin, lin)

    # Convertir a Coordenadas Polares (Ángulo)
    # np.arctan2(y, x) nos da el ángulo de cada píxel (de -pi a +pi)
    # Este cálculo es independiente de las unidades (metros o µm).
    angle = np.arctan2(y, x)

    # Generar el Patrón de la Estrella
    pattern = np.sin(N_spokes * angle) > 0

    # Convertir de Booleano (True/False) a float (1.0/0.0)
    S_objeto = pattern.astype(float)
        
    return S_objeto

def crear_pupila_circular(M, L_pupila_calc, R_pupila):
    # Crear la rejilla de coordenadas de la pupila
    coords_pupila = np.linspace(-L_pupila_calc/2, L_pupila_calc/2, M)
    X_p, Y_p = np.meshgrid(coords_pupila, coords_pupila)
    
    # Calcular la distancia radial de cada píxel al centro
    R_p = np.sqrt(X_p**2 + Y_p**2)
    
    # Crear la apertura: 1.0 adentro del radio, 0.0 afuera
    P_pupila = np.zeros((M, M), dtype=complex)
    P_pupila[R_p < R_pupila] = 1.0 + 0.0j # 1.0 para transmisión total
    
    return P_pupila


def simular_convolucion(objeto, L_objeto, M, long_onda, f_MO, f_TL, R_pupila):

    # Calcular la escala del plano de la pupila (igual que antes)
    # Esto es crucial para que R_pupila (en mm) coincida con los píxeles
    dx_objeto = L_objeto / M
    L_pupila_calc = (long_onda * f_MO) / dx_objeto

    # Crear la pupila (¡usando la nueva función simple!)
    P_pupila_TF = crear_pupila_circular(M, L_pupila_calc, R_pupila)
    
    # Simular la propagación y el filtrado (igual que antes)
    # Objeto -> FFT -> Pupila
    S_fft = np.fft.fft2(np.fft.ifftshift(objeto))
    
    # Pupila -> Multiplicación -> Cámara
    E_cam_fft = S_fft * np.fft.fftshift(P_pupila_TF)
    
    # Cámara -> IFFT -> Imagen final
    campo_cam_convolucion = np.fft.fftshift(np.fft.ifft2(E_cam_fft))

    # Calcular magnificación y escala de la imagen final (igual que antes)
    Mag_total = f_TL / f_MO
    L_cam = L_objeto * Mag_total
    
    return campo_cam_convolucion, L_cam, P_pupila_TF, L_pupila_calc



# Definimos los Parámetros
long_onda = 533e-9      # Longitud de onda (533 nm)
f_MO = 10e-3             # Focal Objetivo (9 mm para un 20x)
f_TL = 200e-3           # Focal Lente de Tubo (200 mm)
NA = 0.5                # Apertura Numérica del objetivo (0.5 para 20x)

# Magnificación total del sistema
Mag_total = f_TL / f_MO

# Parámetros de la Cámara (Basler)
dx_real_camara = 2.74e-6 # pixel size de 2.74 µm que es el aumento del MO
num_pixels = 2848        # Resolución de la cámara
L_sensor = num_pixels*dx_real_camara

# Generamos la imagen del test
L_objeto = L_sensor / Mag_total  # Dimensión total que representará tu imagen en um
N_lines = 128                     # Número de PARES de líneas (blanco/negro).
objeto = generar_estrella_siemens(L_objeto, num_pixels, N_lines)

# Radio físico de la pupila
R_pupila = NA * f_MO
print(f"Radio de la Pupila P(x,y): {R_pupila * 1000:.2f} mm")


campo, L_magnificada, pupila_usada, L_pupila = simular_convolucion(objeto, L_objeto, num_pixels, long_onda, f_MO, f_TL, R_pupila)

# Simulamos
intensidad = np.abs(campo)**2 #Intensidad campo claro
intensidad = intensidad/np.max(intensidad)

plt.figure(figsize=(12, 6)) # Hacemos la figura más ancha
plt.suptitle("Test de resolución", fontsize=16)

# Figura 1: Test de resolucion
ext_obj = [-L_objeto/2 * 1e6, L_objeto/2 * 1e6, -L_objeto/2 * 1e6, L_objeto/2 * 1e6]
plt.subplot(1, 2, 1)
plt.imshow(np.abs(objeto), cmap='gray', extent=ext_obj)
plt.title('Objeto Original S(ξ,η)')
plt.xlabel('ξ (μm)')
plt.ylabel('η (μm)')

# Figura 2: Imagen de vista desde el microscopio
ext_cam = [-L_magnificada/2 * 1e6, L_magnificada/2 * 1e6, -L_magnificada/2 * 1e6, L_magnificada/2 * 1e6]
plt.subplot(1, 2, 2)
plt.imshow(intensidad, cmap='gray', extent=ext_cam)
plt.title('Imagen en el MO')
plt.xlabel('u (μm)')
plt.ylabel('v (μm)')


plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Figura de las Pupilas
plt.figure(figsize=(12, 5))
ext_pup = [-L_pupila/2 * 1000, L_pupila/2 * 1000, -L_pupila/2 * 1000, L_pupila/2 * 1000]
zoom_lim = R_pupila * 1.2 * 1000 

plt.subplot(1, 2, 1)
plt.imshow(np.abs(pupila_usada), cmap='gray', extent=ext_pup)
plt.title('TF P(x,y) - Campo Claro')
plt.xlabel('x (mm)')
plt.ylabel('y (mm)')
plt.xlim(-zoom_lim, zoom_lim)
plt.ylim(-zoom_lim, zoom_lim)

plt.tight_layout()
plt.show()
