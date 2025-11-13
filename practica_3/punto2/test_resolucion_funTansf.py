import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import scipy.ndimage as ndi

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


def calcular_r_blur(intensidad_img, L_magnificada, plot_diagnostico=True):
    """
    Calcula r_blur usando un filtro Laplaciano (detector de bordes).
    
    Estima el radio del círculo borroso (r_blur) en una imagen de
    estrella Siemens mediante la detección de bordes.
    
    Args:
        intensidad_img (np.ndarray): Matriz 2D de la imagen final (magnificada).
        L_magnificada (float): Tamaño físico total de la imagen (en metros).
        plot_diagnostico (bool): Si es True, muestra gráficas de depuración.
        
    Returns:
        float: El radio r_blur medido en la imagen magnificada (en metros).
    """
    num_pixels = intensidad_img.shape[0]
    
    # 1. Aplicar filtro Laplaciano para detectar bordes.
    #    Las áreas planas (sin resolver) tienden a 0.
    bordes_laplace = ndi.laplace(intensidad_img)
    intensidad_bordes = np.abs(bordes_laplace)
    
    # 2. Tomar un perfil 1D (corte horizontal por el centro)
    centro_y = num_pixels // 2
    perfil_bordes = intensidad_bordes[centro_y, :]
    
    # 3. Analizar solo la mitad derecha del perfil (radio >= 0)
    centro_x = num_pixels // 2
    perfil_derecha = perfil_bordes[centro_x:]
    
    # 4. Normalizar el perfil de bordes
    perfil_norm = perfil_derecha / np.max(perfil_derecha)
    
    # 5. Suavizar el perfil 1D para eliminar ruido de alta frecuencia.
    #    'sigma=5' es la desviación estándar del filtro Gaussiano.
    perfil_suavizado = ndi.gaussian_filter1d(perfil_norm, sigma=5)

    # 6. Definir el umbral para considerar un "borde"
    umbral_borde = 0.10  # 10% del borde máximo
    
    try:
        # 7. Encontrar el primer píxel (índice) donde el perfil 
        #    suavizado supera el umbral.
        indices_borde = np.where(perfil_suavizado > umbral_borde)[0]
        
        # 7b. Omitir artefactos de ruido cercanos al centro (índice 0)
        offset_central = 10 # Píxeles a ignorar desde el centro
        indices_validos = indices_borde[indices_borde > offset_central] 
        
        r_blur_pixels = indices_validos[0]
        
    except IndexError:
        print("Error de filtro de borde: No se encontró ningún borde válido.")
        r_blur_pixels = 0
        
    # 8. Convertir la medición de píxeles a metros
    pixel_size_magnificado = L_magnificada / num_pixels
    r_blur_mag_m = r_blur_pixels * pixel_size_magnificado
    
    # --- Sección de Gráficas de Diagnóstico ---
    if plot_diagnostico:
        plt.figure(figsize=(12, 6))
        
        # Gráfico 1: Imagen 2D de los bordes detectados
        plt.subplot(1, 2, 1)
        plt.imshow(intensidad_bordes, cmap='hot')
        plt.title('Imagen Filtrada (Laplaciano)')
        plt.xlabel('Píxeles')
        plt.ylabel('Píxeles')
        
        # Ejemplo: Descomentar para aplicar un zoom manual al centro
        # plt.xlim(centro_x - 500, centro_x + 500)
        # plt.ylim(centro_y - 500, centro_y + 500)

        # Gráfico 2: Perfil 1D (Original vs. Suavizado)
        plt.subplot(1, 2, 2)
        radios_pix = np.arange(len(perfil_norm))
        radios_um = radios_pix * pixel_size_magnificado * 1e6
        
        plt.plot(radios_um, perfil_norm, label='Perfil Original (Ruidoso)', alpha=0.3)
        plt.plot(radios_um, perfil_suavizado, label='Perfil Suavizado', color='blue', linewidth=2)
        plt.axhline(y=umbral_borde, color='r', linestyle='--', label=f'Umbral ({umbral_borde*100:.0f}%)')
        plt.axvline(x=r_blur_mag_m * 1e6, color='g', linestyle='-', label=f'r_blur = {r_blur_mag_m * 1e6:.2f} µm')
        
        plt.title('Perfil de Bordes (Desde el centro)')
        plt.xlabel('Radio (µm - magnificado)')
        plt.ylabel('Intensidad de Borde (Normalizada)')
        plt.legend()
        plt.xlim(0, radios_um[-1] / 2) # Limitar eje X para mejor visualización
        plt.grid(True)
        plt.show()
    
    return r_blur_mag_m


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

# Calcular r_blur automáticamente
# (El resultado está en metros, en la imagen magnificada)
r_blur_mag_m = calcular_r_blur(intensidad, L_magnificada)
r_blur_mag_um = r_blur_mag_m * 1e6 # Convertir a micras para leerlo

print(f"Radio borroso medido en la imagen magnificado r_blur_mag = {r_blur_mag_um:.2f} µm")

# Convertir r_blur a unidades sin magnificación
r_blur_real_m = r_blur_mag_m / Mag_total
r_blur_real_um = r_blur_real_m * 1e6

print(f"Radio borroso en el objeto real r_blur = {r_blur_real_um:.2f} µm")

# Calcular la resolución mínima
# (Usando la fórmula de la estrella de Siemens)
d_min_medido = (2 * np.pi * r_blur_real_m) / N_lines

print(f"Resolución mínima d_min_exp = {d_min_medido * 1e9:.0f} nm")

# Calcular la resolución TEÓRICA (Abbe)
d_min_teorico = long_onda / NA

print(f"Resolución mínima de Abbe d_min_teo = {d_min_teorico * 1e9:.0f} nm")

# Porcentaje de error
error = abs(d_min_medido-d_min_teorico)*100/d_min_teorico

print(f"El error de resolución obtenido es de: {error:.2f} %")

##Graficamos
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
plt.figure(figsize=(6, 5))
ext_pup = [-L_pupila/2 * 1000, L_pupila/2 * 1000, -L_pupila/2 * 1000, L_pupila/2 * 1000]
zoom_lim = R_pupila * 1.2 * 1000 
plt.imshow(np.abs(pupila_usada), cmap='gray', extent=ext_pup)
plt.title('Función pupila P(x,y)')
plt.xlabel('x (mm)')
plt.ylabel('y (mm)')
plt.xlim(-zoom_lim, zoom_lim)
plt.ylim(-zoom_lim, zoom_lim)

plt.tight_layout()
plt.show()
