import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max

# Definicion de funciones a usar

def espejo():
    """Retorna la matriz ABCD para un espejo plano."""
    return np.array([[1, 0], [0, 1]])

def traslacion(distancia):
    """Retorna la matriz ABCD para propagación en espacio libre."""
    return np.array([[1, distancia], [0, 1]])

def lente_delgada(foco):
    """Retorna la matriz ABCD para una lente delgada."""
    return np.array([[1, 0], [-1/foco, 1]])

def propagar_difracción(campo_entrada, lado_entrada, long_onda, matriz_abcd):
    """
    Propaga un campo usando la integral de Collins (método FFT).
    Maneja el caso B=0 (sistema de imagen) por separado.
    """
    A, B, C, D = matriz_abcd.ravel()
    M_puntos = campo_entrada.shape[0]
    
    # Caso Sistema de Imagen (B=0)
    if abs(B) < 1e-9:
        print("Advertencia: B=0. Sistema de Imagen (4f) detectado.")
        magnificacion = A
        L_salida = abs(magnificacion) * lado_entrada
        print(f"Propagación geométrica. Magnificación: {magnificacion:.2f}x")
        
        if magnificacion < 0:
            campo_salida = np.flip(campo_entrada) # Inverimos la imagen imagen
        else:
            campo_salida = campo_entrada
        return campo_salida, L_salida

    # Propagación General (B != 0)
    k = 2 * np.pi / long_onda
    
    # 1. Coordenadas de entrada
    dx_entrada = lado_entrada / M_puntos
    coords_entrada = np.linspace(-lado_entrada/2, lado_entrada/2, M_puntos)
    p, q = np.meshgrid(coords_entrada, coords_entrada)

    # 2. Fase cuadrática A
    fase_cuadratica_A = np.exp(1j * k * A / (2 * B) * (p**2 + q**2))
    campo_intermedio = campo_entrada * fase_cuadratica_A

    # 3. Transformada de Fourier
    campo_fourier = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(campo_intermedio)))

    # 4. Coordenadas de salida
    L_salida = long_onda * abs(B) / dx_entrada
    coords_salida = np.linspace(-L_salida/2, L_salida/2, M_puntos)
    X, Y = np.meshgrid(coords_salida, coords_salida)

    # 5. Fase cuadrática D y escala
    fase_cuadratica_D = np.exp(1j * k * D / (2 * B) * (X**2 + Y**2))
    factor_escala = (dx_entrada**2) / (1j * long_onda * B)
    
    campo_salida = factor_escala * fase_cuadratica_D * campo_fourier
    
    return campo_salida, L_salida



# Funciones para la simulación

def subir_imagen(ruta):
    """Carga una imagen desde 'ruta', la normaliza, la hace cuadrada y la devuelve como 'complex'."""
    try:
        img = Image.open(ruta).convert("L") #la convertimos a blanco y negro
        arr = np.array(img)/255.0 #la normalizamos [0,1]
    except FileNotFoundError:
        print(f"ERROR: No se encontró la imagen en {ruta}. Usando un objeto de prueba.")
        M_temp = 1024
        coords_temp = np.linspace(-0.5, 0.5, M_temp)
        XX, YY = np.meshgrid(coords_temp, coords_temp)
        arr = (XX**2 + YY**2) < 0.1
        arr = arr.astype(float)

    M_size = np.shape(arr)
    if M_size[0] != M_size[1]:
        M = max(M_size[0],M_size[1])
        objeto = np.zeros((M,M), dtype=complex) # Usar 'complex'
        start_row, start_col = int((M-M_size[0])/2), int((M-M_size[1])/2)
        objeto[start_row : start_row + M_size[0], start_col : start_col + M_size[1]] = arr
    else:
        M = M_size[0]
        objeto = arr.astype(complex) # Usar 'complex'
    
    print(f"Tamaño de la malla (M): {M}x{M}")
    return objeto, M

# Funcion para el camino 1 (cam1)
def camino_1(objeto, L_objeto, M, long_onda, f, L1_M1, L2_M1, 
             radio_dc_bloquear, sigma_muesca):
    """
    Simula el Camino 1 (4f) aplicando un FILTRO DE MUESCA GAUSSIANO (Punto 3).
    - radio_dc_bloquear: Radio (metros) del centro a ignorar para buscar picos.
    - sigma_muesca: Ancho/difuminado (metros) de las muescas Gaussianas.
    """
    print("\n--- Calculando Camino 1 (Punto 3: Filtro de Muesca Gaussiano) ---")

    # 1. Propagación S -> M1
    print("Paso 1: Propagando S -> M1 (Plano de Fourier)...")
    M_S_M1 = traslacion(f) @ lente_delgada(f) @ traslacion(f)
    campo_M1, L_M1 = propagar_difracción(objeto, L_objeto, long_onda, M_S_M1)

    # 2. Aplicamos la transmitancia t(x,y)
    print("Paso 2: Creando filtro t(x,y) (Apertura M1 + Muescas Gaussianas)...")
    coords_M1 = np.linspace(-L_M1/2, L_M1/2, M)
    X_M1, Y_M1 = np.meshgrid(coords_M1, coords_M1)

    # Filtro 1: Apertura finita del espejo M1
    apertura_M1 = np.zeros((M, M))
    apertura_M1[ (np.abs(X_M1) < (L1_M1 / 2)) & (np.abs(Y_M1) < (L2_M1 / 2)) ] = 1

    # Filtro de muesca Gaussiano para filtrar las frecuencias
    print("Paso 2a: Detectando picos de ruido...")
    
    # A. Encontrar coordenadas de picos de ruido (en píxeles)
    intensidad_M1 = np.abs(campo_M1)**2
    intensidad_M1_sin_dc = intensidad_M1.copy()
    
    # Calcular radio FÍSICO desde el centro
    R_M1 = np.sqrt(X_M1**2 + Y_M1**2)
    
    # Poner a cero la intensidad dentro del radio de bloqueo
    intensidad_M1_sin_dc[R_M1 < radio_dc_bloquear] = 0
    print(f"Bloqueando {radio_dc_bloquear*1000:.2f} mm centrales para detectar ruido.")
    
    # Encontrar las coordenadas de picos más brillantes (no-DC)
    coordenadas_picos = peak_local_max(
        intensidad_M1_sin_dc, 
        min_distance=10, 
        threshold_rel=0.1, # Ignoramos el pico central ya que corresponde a las frecuencias bajas
        num_peaks=10  # Maxima cantidad de picos
    )
    
    print(f"Se detectaron {len(coordenadas_picos)} picos de ruido.")
    
    # B. Crear la máscara (filtro_anti_ruido)
    # Empezar con un filtro que deja pasar todo
    filtro_anti_ruido = np.ones((M, M))
    
    # Creemos el filtro de muesca Gaussiano para cada pico
    for (py, px) in coordenadas_picos:
        
        # Obtenemos la coordenada física del pico (en metros)
        x_pico_coord = X_M1[py, px]
        y_pico_coord = Y_M1[py, px]
        
        # Reportar la frecuencia espacial
        fx = x_pico_coord / (long_onda * f)
        fy = y_pico_coord / (long_onda * f)
        print(f"Pico de ruido detectado en (fx, fy) = ({fx:6.2f}, {fy:6.2f}) 1/m")
        
        # Calculamos  la muesca Gaussiana para el pico
        # G(x,y) = 1.0 - exp( -dist^2 / (2*sigma^2) )
        R_cuadrado_pico = (X_M1 - x_pico_coord)**2 + (Y_M1 - y_pico_coord)**2
        muesca_gaussiana = 1.0 - np.exp(-R_cuadrado_pico / (2 * sigma_muesca**2))
        
        # Multiplicar el filtro principal por esta muesca
        filtro_anti_ruido = filtro_anti_ruido * muesca_gaussiana
            
    print(f"Picos bloqueados con muescas Gaussianas de sigma = {sigma_muesca*1000:.2f} mm.")

    # El filtro total va a ser la multiplicación de los dos filtros (la interseccion)
    t_total = apertura_M1 * filtro_anti_ruido
    campo_M1_filtrado = campo_M1 * t_total
    
    # 3. Propagación M1 -> Cam1
    print("Paso 3: Propagando M1 -> Cam1 (Plano Imagen)...")
    M_M1_Cam1 = traslacion(f) @ lente_delgada(f) @ traslacion(f)
    campo_cam1, L_cam1 = propagar_difracción(campo_M1_filtrado, L_M1, long_onda, M_M1_Cam1)
    
    return campo_cam1, L_cam1, t_total, L_M1

def camino_2(objeto, L_objeto, long_onda, f, d):
    """Simula el Camino 2 (Transformada de Fourier)."""
    print("\n--- Calculando Camino 2 (Ideal) ---")
    matriz_cam_2 = traslacion(f)@lente_delgada(f)@traslacion(f/2)@espejo()@traslacion(d-(f/2))
    campo_cam2, L_cam2 = propagar_difracción(objeto, L_objeto, long_onda, matriz_cam_2)
    return campo_cam2, L_cam2


def ver_resultados(intensidad_cam1, L_cam1, intensidad_cam2, L_cam2,
                          t_total, L_M1, L1_M1, L2_M1):
    """Crea y muestra las dos figuras de resultados: el filtro y las cámaras."""
    print("\nMostrando resultados...")

    # Figura 1: El Filtro
    plt.figure(figsize=(7, 6))
    plt.imshow(t_total, cmap='gray', extent=[-L_M1/2*100, L_M1/2*100, -L_M1/2*100, L_M1/2*100])
    plt.title('Filtro t(x,y) Aplicado en Plano M1 (Muescas Gaussianas)')
    plt.xlabel('x (cm)')
    plt.ylabel('y (cm)')
    plt.colorbar(label='Transmitancia (0 a 1)')
    # Zoom
    plt.xlim(-L1_M1/2 * 100 * 1.2, L1_M1/2 * 100 * 1.2)
    plt.ylim(-L2_M1/2 * 100 * 1.2, L2_M1/2 * 100 * 1.2)

    # Figura 2: Las Cámaras 1y 2
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    # Flipeamos la intensidad para conservar la posicion original de la imagen (para que no salga invertida)
    plt.imshow(np.flip(intensidad_cam1), cmap='gray', extent=[-L_cam1/2*100, L_cam1/2*100, -L_cam1/2*100, L_cam1/2*100])
    plt.title('Intensidad en Cámara 1 (Filtrada con Muescas)')
    plt.xlabel('u (cm)')
    plt.ylabel('v (cm)')

    plt.subplot(1, 2, 2)
    plt.imshow(np.log(1 + intensidad_cam2), cmap='gray', extent=[-L_cam2/2*100, L_cam2/2*100, -L_cam2/2*100, L_cam2/2*100])
    plt.title('Intensidad en Cámara 2 (Ideal TF, Escala Log)')
    plt.xlabel("x' (cm)")
    plt.ylabel("y' (cm)")

    plt.tight_layout()
    plt.show()



# Definimos los Parámetros
f = 0.500 # 500 mm
d = 2*f    # d>f
long_onda = 633e-9 # 633 nm
L_objeto = 1e-2 # Imágen cuadrada de lado = 1 cm

# Aperturas
L1_M1 = 0.0104 # 10.4 mm
L2_M1 = 0.0058 # 5.8 mm

# 1. Radio (metros) del centro a ignorar al buscar picos. (en mm)
radio_dc_a_bloquear = 0.0003 

# 2. Sigma (metros) de las muescas para controlar el difuminado (en mm)
sigma_muesca = 0.0002

# Ruta de la imagen

#ruta_imagen = r"C:\Users\david\OneDrive\Desktop\practica2\img1.bmp"
ruta_imagen =r"C:\Users\david\OneDrive\Desktop\Noise images\Noise (1).png"

# Cargamos el objeto
objeto, M = subir_imagen(ruta_imagen)

# Simulamos el Camino 1
campo_cam1, L_cam1, t_total, L_M1 = camino_1(
    objeto, L_objeto, M, long_onda, f, L1_M1, L2_M1, 
    radio_dc_a_bloquear, sigma_muesca
)

# Simulamos el Camino 2
campo_cam2, L_cam2 = camino_2(
    objeto, L_objeto, long_onda, f, d
)

# Calculamos intensidades
intensidad_cam1 = np.abs(campo_cam1)**2
intensidad_cam2 = np.abs(campo_cam2)**2

# Mostramos todo
ver_resultados(
    intensidad_cam1, L_cam1, intensidad_cam2, L_cam2,
    t_total, L_M1, L1_M1, L2_M1
)