import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from skimage.feature import peak_local_max

import scipy.signal



#1. FUNCIONES DE ÓPTICA Y PROPAGACIÓN


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
            campo_salida = np.flip(campo_entrada) # Invertir imagen
        else:
            campo_salida = campo_entrada
        return campo_salida, L_salida

    # Propagación General (B != 0)
    k = 2 * np.pi / long_onda
    dx_entrada = lado_entrada / M_puntos
    coords_entrada = np.linspace(-lado_entrada/2, lado_entrada/2, M_puntos)
    p, q = np.meshgrid(coords_entrada, coords_entrada)
    fase_cuadratica_A = np.exp(1j * k * A / (2 * B) * (p**2 + q**2))
    campo_intermedio = campo_entrada * fase_cuadratica_A
    campo_fourier = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(campo_intermedio)))
    L_salida = long_onda * abs(B) / dx_entrada
    coords_salida = np.linspace(-L_salida/2, L_salida/2, M_puntos)
    X, Y = np.meshgrid(coords_salida, coords_salida)
    fase_cuadratica_D = np.exp(1j * k * D / (2 * B) * (X**2 + Y**2))
    factor_escala = (dx_entrada**2) / (1j * long_onda * B)
    campo_salida = factor_escala * fase_cuadratica_D * campo_fourier
    return campo_salida, L_salida



#2. FUNCIONES DE SIMULACIÓN Y CARGA DE DATOS


def cargar_imagen_simple(ruta):
    """Carga una imagen simple y la devuelve como un array numpy."""
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
    return arr

def preparar_imagenes_comparacion(ruta_simulacion, ruta_experimental):
    """
    Carga la imagen de simulación y la experimental.
    Las empadrona (padding) al mismo tamaño máximo 'M'.
    """
    arr_sim = cargar_imagen_simple(ruta_simulacion)
    arr_exp = cargar_imagen_simple(ruta_experimental)

    # Encontrar las dimensiones máximas
    M_sim_y, M_sim_x = arr_sim.shape
    M_exp_y, M_exp_x = arr_exp.shape
    M = max(M_sim_y, M_sim_x, M_exp_y, M_exp_x) # Dimensión cuadrada máxima

    # Empadronar la imagen de simulación (devuelta como complex)
    objeto_simulacion = np.zeros((M,M), dtype=complex)
    start_row, start_col = int((M-M_sim_y)/2), int((M-M_sim_x)/2)
    objeto_simulacion[start_row : start_row + M_sim_y, start_col : start_col + M_sim_x] = arr_sim

    # Empadronar la imagen experimental (devuelta como float)
    img_experimental = np.zeros((M,M))
    start_row, start_col = int((M-M_exp_y)/2), int((M-M_exp_x)/2)
    img_experimental[start_row : start_row + M_exp_y, start_col : start_col + M_exp_x] = arr_exp

    print(f"Imágenes cargadas y empadronadas a {M}x{M}")
    return objeto_simulacion, img_experimental, M


def simular_camino_1_filtrado(objeto, L_objeto, M, long_onda, f, L1_M1, L2_M1,
                             radio_dc_bloquear, sigma_muesca):
    """Simula el Camino 1 (4f) aplicando un FILTRO DE MUESCA GAUSSIANO (Punto 3)."""
    print("\n--- Calculando Camino 1 (Punto 3: Filtro de Muesca Gaussiano) ---")

    # 1. Propagación S -> M1
    print("Paso 1: Propagando S -> M1 (Plano de Fourier)...")
    M_S_M1 = traslacion(f) @ lente_delgada(f) @ traslacion(f)
    campo_M1, L_M1 = propagar_difracción(objeto, L_objeto, long_onda, M_S_M1)

    # 2. Aplicar la transmitancia t(x,y)
    print("Paso 2: Creando filtro t(x,y) (Apertura M1 + Muescas Gaussianas)...")
    coords_M1 = np.linspace(-L_M1/2, L_M1/2, M)
    X_M1, Y_M1 = np.meshgrid(coords_M1, coords_M1)
    apertura_M1 = np.zeros((M, M))
    apertura_M1[ (np.abs(X_M1) < (L1_M1 / 2)) & (np.abs(Y_M1) < (L2_M1 / 2)) ] = 1

    # --- LÓGICA DE FILTRO DE MUESCA GAUSSIANO ---
    print("Paso 2a: Detectando picos de ruido...")
    intensidad_M1 = np.abs(campo_M1)**2
    intensidad_M1_sin_dc = intensidad_M1.copy()
    R_M1 = np.sqrt(X_M1**2 + Y_M1**2)
    intensidad_M1_sin_dc[R_M1 < radio_dc_bloquear] = 0
    print(f"Bloqueando {radio_dc_bloquear*1000:.2f} mm centrales para detectar ruido.")

    coordenadas_picos = peak_local_max(
        intensidad_M1_sin_dc,
        min_distance=10,
        threshold_rel=0.1,
        num_peaks=6 # Buscamos 6 picos
    )
    print(f"Se detectaron {len(coordenadas_picos)} picos de ruido.")

    filtro_anti_ruido = np.ones((M, M))

    for (py, px) in coordenadas_picos:
        x_pico_coord = X_M1[py, px]
        y_pico_coord = Y_M1[py, px]
        fx = x_pico_coord / (long_onda * f)
        fy = y_pico_coord / (long_onda * f)
        print(f"  Pico de ruido detectado en (fx, fy) = ({fx:6.2f}, {fy:6.2f}) 1/m")
        R_cuadrado_pico = (X_M1 - x_pico_coord)**2 + (Y_M1 - y_pico_coord)**2
        muesca_gaussiana = 1.0 - np.exp(-R_cuadrado_pico / (2 * sigma_muesca**2))
        filtro_anti_ruido = filtro_anti_ruido * muesca_gaussiana

    print(f"Picos bloqueados con muescas Gaussianas de sigma = {sigma_muesca*1000:.2f} mm.")
   

    t_total = apertura_M1 * filtro_anti_ruido
    campo_M1_filtrado = campo_M1 * t_total

    # 3. Propagación M1 -> Cam1
    print("Paso 3: Propagando M1 -> Cam1 (Plano Imagen)...")
    M_M1_Cam1 = traslacion(f) @ lente_delgada(f) @ traslacion(f)
    campo_cam1, L_cam1 = propagar_difracción(campo_M1_filtrado, L_M1, long_onda, M_M1_Cam1)

    return campo_cam1, L_cam1, t_total, L_M1

def simular_camino_2(objeto, L_objeto, long_onda, f, d):
    """Simula el Camino 2 (Transformada de Fourier)."""
    print("\n--- Calculando Camino 2 (Ideal) ---")
    matriz_cam_2 = traslacion(f)@lente_delgada(f)@traslacion(f/2)@espejo()@traslacion(d-(f/2))
    campo_cam2, L_cam2 = propagar_difracción(objeto, L_objeto, long_onda, matriz_cam_2)
    return campo_cam2, L_cam2



# 3. FUNCIONES DE VISUALIZACIÓN Y ANÁLISIS


def ver_resultados_filtrado(intensidad_cam1, L_cam1, intensidad_cam2, L_cam2,
                            t_total, L_M1, L1_M1, L2_M1):
    """Crea y muestra las dos figuras del resultado del FILTRADO."""
    print("\nMostrando resultados del FILTRADO...")

    # --- Figura 1: El Filtro ---
    plt.figure(figsize=(7, 6))
    plt.imshow(t_total, cmap='gray', extent=[-L_M1/2*100, L_M1/2*100, -L_M1/2*100, L_M1/2*100])
    plt.title(r'Filtro t(x,y) Aplicado en M1')
    plt.xlabel('x (cm)')
    plt.ylabel('y (cm)')
    plt.colorbar(label='Transmitancia (0 a 1)')
    plt.xlim(-L1_M1/2 * 100 * 1.2, L1_M1/2 * 100 * 1.2)
    plt.ylim(-L2_M1/2 * 100 * 1.2, L2_M1/2 * 100 * 1.2)

    # Figura 2: Las Cámaras
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(np.flip(intensidad_cam1), cmap='gray', extent=[-L_cam1/2*100, L_cam1/2*100, -L_cam1/2*100, L_cam1/2*100])
    plt.title(r'Intensidad en Cámara 1 (Filtrada con Muescas)')
    plt.xlabel('u (cm)')
    plt.ylabel('v (cm)')

    plt.subplot(1, 2, 2)
    plt.imshow(np.log(1 + intensidad_cam2), cmap='gray', extent=[-L_cam2/2*100, L_cam2/2*100, -L_cam2/2*100, L_cam2/2*100])
    plt.title(r'Intensidad en Cámara 2 (Ideal TF, Escala Log)')
    plt.xlabel(r"x' (cm)")
    plt.ylabel(r"y' (cm)")
    plt.tight_layout()

# Funcion para la correlacion
def analizar_similitud_con_correlacion(img_1_intensidad, img_2_intensidad):
    """
    Calcula la correlación numérica entre dos imágenes (img_1 y img_2)
    para cuantificar qué tan parecidas son.
    """
    print("\n--- Iniciando Análisis de Similitud (Correlación Numérica) ---")

    # 1. Calcular la correlación cruzada
    print("Calculando correlación cruzada (Img_1 ★ Img_2)...")
    correlacion = scipy.signal.correlate(
        img_1_intensidad,
        img_2_intensidad,
        mode='same',
        method='fft'
    )

    # 2. Calcular el "Valor" (Coeficiente de Correlación Normalizado)
    print("Calculando autocorrelaciones (Img_1 ★ Img_1) y (Img_2 ★ Img_2)...")
    autocorr_1 = scipy.signal.correlate(
        img_1_intensidad,
        img_1_intensidad,
        mode='same',
        method='fft'
    )
    autocorr_2 = scipy.signal.correlate(
        img_2_intensidad,
        img_2_intensidad,
        mode='same',
        method='fft'
    )

    # Encontrar el pico de la correlación cruzada
    (py_corr, px_corr) = np.unravel_index(np.argmax(correlacion), correlacion.shape)
    valor_pico_corr = correlacion[py_corr, px_corr]

    # Normalizar
    denominador = np.sqrt(np.max(autocorr_1) * np.max(autocorr_2))
    if denominador == 0:
        valor_normalizado = 0
    else:
        valor_normalizado = valor_pico_corr / denominador

    print("--- ¡RESULTADO DEL ANÁLISIS! ---")
    print(f"Valor Pico de Correlación (Img_1 ★ Img_2): {valor_pico_corr:.2e}")
    print(f"Coeficiente de Correlación Normalizado: {valor_normalizado:.4f}")

    # 3. Extraer las "Curvas" (Perfiles 1D)
    perfil_horizontal = correlacion[py_corr, :]
    perfil_vertical = correlacion[:, px_corr]
    eje_x = np.arange(correlacion.shape[1])
    eje_y = np.arange(correlacion.shape[0])

    # 4. Graficar el análisis de correlación
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Análisis Numérico de Similitud (Experimental vs. Simulación Corregida)', fontsize=16)

    # Gráfica 1: Mapa de Correlación 2D
    im = axs[0].imshow(correlacion, cmap='hot')
    axs[0].set_title(r'Mapa de Correlación (Exp ★ Sim)')
    axs[0].plot(px_corr, py_corr, 'g+', markersize=10, label=f'Pico ({valor_normalizado:.3f})')
    axs[0].set_xlabel('Desplazamiento X (píxeles)')
    axs[0].set_ylabel('Desplazamiento Y (píxeles)')
    axs[0].legend()
    fig.colorbar(im, ax=axs[0])

    # Gráfica 2: Curvas 1D
    axs[1].plot(eje_x, perfil_horizontal, 'r-', label='Perfil Horizontal')
    axs[1].plot(eje_y, perfil_vertical, 'b--', label='Perfil Vertical')
    axs[1].set_title(r'Curvas de Correlación (Perfiles del Pico)')
    axs[1].set_xlabel('Posición (píxeles)')
    axs[1].set_ylabel('Intensidad de Correlación (un. arb.)')
    axs[1].grid(True)
    axs[1].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.9])


def ver_comparacion_visual(img_simulada_corregida, img_experimental):
    """
    Muestra una comparación visual lado a lado de la imagen
    simulada (ya filtrada y des-invertida) y la experimental.
    Ambas imágenes ya deben estar empadronadas al mismo tamaño.
    """
    print("\nMostrando comparación visual (Simulación vs. Experimental)...")

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle('Simulación vs. Experimental', fontsize=16)

    # Gráfica 1: Resultado Simulado (ya des-invertido)
    im1 = axs[0].imshow(img_simulada_corregida, cmap='gray')
    axs[0].set_title(r'Resultado Simulado')
    axs[0].set_xlabel('u (píxeles)')
    axs[0].set_ylabel('v (píxeles)')
    # fig.colorbar(im1, ax=axs[0], fraction=0.046, pad=0.04) # Opcional: barra de color

    # Gráfica 2: Resultado Experimental
    im2 = axs[1].imshow(img_experimental, cmap='gray')
    axs[1].set_title(r'Resultado Experimental (Laboratorio)')
    axs[1].set_xlabel('x (píxeles)')
    axs[1].set_ylabel('y (píxeles)')
    # fig.colorbar(im2, ax=axs[1], fraction=0.046, pad=0.04) # Opcional: barra de color

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])



# 4. SCRIPT PRINCIPAL (Ejecución directa) ---


#Definimos los Parámetros
f = 0.500 # 500 mm
d = 2*f    # d>f
long_onda = 633e-9 # 633 nm
L_objeto = 1e-2 # Imágen cuadrada de lado = 1 cm

# Aperturas
L1_M1 = 0.0104 # 10.4 mm
L2_M1 = 0.0058 # 5.8 mm

# --- Parámetros de Filtrado (¡Juega con estos!) ---
radio_dc_a_bloquear = 0.0005 # 0.5 mm
sigma_muesca = 0.0002 # 0.2 mm


# Imagenes para la correlacion
# 1. Imagen de entrada
ruta_imagen_simulacion = r"C:\Users\david\OneDrive\Desktop\practica2\img0.bmp"

# 2. Imagen experimental
ruta_imagen_experimental = r"C:\Users\david\OneDrive\Desktop\practica2\img2.bmp"


# Ejecutar Simulación y Preparación

# Cargar y empadronar AMBAS imágenes al mismo tamaño (M)
objeto_simulacion, img_experimental_padded, M = preparar_imagenes_comparacion(
    ruta_imagen_simulacion,
    ruta_imagen_experimental
)

# Simular Camino 1 (Filtrado) - USAMOS LA IMAGEN DE SIMULACIÓN
campo_cam1, L_cam1, t_total, L_M1 = simular_camino_1_filtrado(
    objeto_simulacion, L_objeto, M, long_onda, f, L1_M1, L2_M1,
    radio_dc_a_bloquear, sigma_muesca
)

# (Opcional) Simular Camino 2
campo_cam2, L_cam2 = simular_camino_2(
    objeto_simulacion, L_objeto, long_onda, f, d
)

# Calcular intensidades de la SIMULACIÓN
intensidad_cam1_simulada = np.abs(campo_cam1)**2




max_intensidad_cam1 = np.max(intensidad_cam1_simulada)     #Encontramos la intensidad máxima para normalizar

#Buscamos una intensidad que se vea bien el patrón
if max_intensidad_cam1 > 0:
    intensidad_log_cam1 = np.log1p(intensidad_cam1_simulada / max_intensidad_cam1 * 85)
    intensidad_norm_cam1 = intensidad_log_cam1 / np.max(intensidad_log_cam1)
else:
    intensidad_norm_cam1 = intensidad_cam1_simulada



intensidad_cam2_simulada = np.abs(campo_cam2)**2




# Análisis

# 1. Mostrar los resultados del FILTRADO (las imágenes de la simulación)
ver_resultados_filtrado(
    intensidad_norm_cam1, L_cam1, intensidad_cam2_simulada, L_cam2,
    t_total, L_M1, L1_M1, L2_M1
)

# 2. Mostramos el análisis de SIMILITUD (Simulación vs. Experimental)
print("\n\n--- COMPARANDO RESULTADO SIMULADO vs. IMAGEN EXPERIMENTAL ---")

# Img_1 = Experimental, Img_2 = Simulada corregida
analizar_similitud_con_correlacion(
    img_experimental_padded,               # Imagen 1 (Experimental)
    np.flip(intensidad_norm_cam1)      # Imagen 2 (Simulada y des-invertida)
)

# 3. Mostrar la comparación visual lado a lado ---
ver_comparacion_visual(
    np.flip(intensidad_norm_cam1),     # Imagen 1 (Simulada y des-invertida)
    img_experimental_padded                # Imagen 2 (Experimental)
)

# Mostrar todas las figuras
plt.show()