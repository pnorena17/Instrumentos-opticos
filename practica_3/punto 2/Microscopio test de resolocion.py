import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


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



def pupila(M, L_pupila_calc, R_pupila, R_bloqueo=0, R_anillo_int=0, R_anillo_ext=0):
    
    coords_pupila = np.linspace(-L_pupila_calc/2, L_pupila_calc/2, M)
    X_p, Y_p = np.meshgrid(coords_pupila, coords_pupila)
    R_p = np.sqrt(X_p**2 + Y_p**2)
    
    P_apertura = np.zeros((M, M), dtype=complex)
    P_apertura[R_p < R_pupila] = 1.0
    
    P_pupila = P_apertura
       
    return P_pupila

# Funcion para el microscopio

def simular_microscopio_4f(objeto, L_objeto, M, long_onda, f_MO, f_TL, R_pupila):
    
    # Propagación S a Pupila (TF 1) 
    print("1. Propagando Objeto -> Plano Pupila (TF)")
    M_S_Pupila = traslacion(f_MO) @ lente_delgada(f_MO) @ traslacion(f_MO)
    campo_pupila, L_pupila_calc = propagar_difracción(objeto, L_objeto, long_onda, M_S_Pupila)
    
    # Creamos el Filtro de Pupila P(x,y) 
    print("2. Creando pupila P(x,y) de tipo...")
    P_pupila = pupila(M, L_pupila_calc, R_pupila)
    
    # Aplicamos Filtro 
    print("3. Aplicando filtro en el plano pupila.")
    campo_pupila_filtrado = campo_pupila * P_pupila
    
    # Propagación de Pupila a Cámara (TF 2)
    print("4. Propagando Plano Pupila -> Cámara (TF)")
    M_Pupila_Cam = traslacion(f_TL) @ lente_delgada(f_TL) @ traslacion(f_TL)
    campo_cam, L_cam = propagar_difracción(campo_pupila_filtrado, L_pupila_calc, long_onda, M_Pupila_Cam)
    
    return campo_cam, L_cam, P_pupila, L_pupila_calc
 
    
 
# Definimos los Parámetros
    
long_onda = 550e-9      # Longitud de onda (550 nm)
f_MO = 9e-3             # Focal Objetivo (9 mm para un 20x)
f_TL = 180e-3           # Focal Lente de Tubo (180 mm)
NA = 0.4                # Apertura Numérica del objetivo (0.4 para 20x)

# Parámetros de la Cámara (Basler)
dx_real_camara = 2.74e-6 # pixel size de 3.75 µm

# Ruta de la Muestra 
ruta_imagen = r"C:\Users\david\OneDrive\Desktop\imagenes descargadas pez simulacion\Siemens_star.png" 


# Cargamos la imagen del test
objeto, M = subir_imagen(ruta_imagen)

# Tamaño del objeto
L_objeto = (dx_real_camara * M * f_MO) / f_TL

# Magnificación total del sistema
Mag_total = f_TL / f_MO

# Radio físico de la pupila
R_pupila = NA * f_MO
print(f"      Radio de la Pupila P(x,y): {R_pupila * 1000:.2f} mm (calculado con NA)")

# Límite de resolución teórico de Abbe
d_min_abbe = long_onda / NA
lp_mm_abbe = 1 / (d_min_abbe * 1000) # (1 / d_en_mm)
print(f"  Límite de Abbe (d = λ/NA): {d_min_abbe * 1e6:.3f} µm (Teórico)")
print(f"  Equivalente en lp/mm: {lp_mm_abbe:.1f} lp/mm (Teórico)")


# Simulamos

campo_cc, L_cc, pupila_cc, L_pupila = simular_microscopio_4f(
    objeto, L_objeto, M, long_onda, f_MO, f_TL, R_pupila
)


int_cc = np.abs(campo_cc)**2 #Intensidad


print("\nMostrando resultados...")

plt.figure(figsize=(12, 6))
plt.suptitle("Simulación de Test de Resolución (Tarea 2b)", fontsize=16)

# Figura 1: Muestra Original
ext_obj = [-L_objeto/2 * 1e6, L_objeto/2 * 1e6, -L_objeto/2 * 1e6, L_objeto/2 * 1e6]
plt.subplot(1, 2, 1)
plt.imshow(np.abs(objeto), cmap='gray', extent=ext_obj)
plt.title('Objeto Original S(ξ,η) (Amplitud)')
plt.xlabel('ξ (μm)')
plt.ylabel('η (μm)')

# Figura 2: Imagen Simulada
ext_cam = [-L_cc/2 * 1e6, L_cc/2 * 1e6, -L_cc/2 * 1e6, L_cc/2 * 1e6]
plt.subplot(1, 2, 2)
plt.imshow(int_cc, cmap='gray', extent=ext_cam)
plt.title('Imagen Simulada en Cámara')
plt.xlabel('u (μm)')
plt.ylabel('v (μm)')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Figura de la Pupila
plt.figure(figsize=(6, 5))
ext_pup = [-L_pupila/2 * 1000, L_pupila/2 * 1000, -L_pupila/2 * 1000, L_pupila/2 * 1000]
plt.imshow(np.abs(pupila_cc), cmap='gray', extent=ext_pup)
plt.title('Pupila P(x,y)')
plt.xlabel('x (mm)')
plt.ylabel('y (mm)')
# Zoom al radio de la pupila
plt.xlim(-R_pupila * 1.2 * 1000, R_pupila * 1.2 * 1000)
plt.ylim(-R_pupila * 1.2 * 1000, R_pupila * 1.2 * 1000)
plt.colorbar()
plt.show()