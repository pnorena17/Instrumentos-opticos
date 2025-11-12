import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# Definicion de funciones a usar

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


def pupila(M, L_pupila_calc, tipo_filtro, R_pupila, R_bloqueo=0):

    coords_pupila = np.linspace(-L_pupila_calc/2, L_pupila_calc/2, M)
    X_p, Y_p = np.meshgrid(coords_pupila, coords_pupila)
    R_p = np.sqrt(X_p**2 + Y_p**2)
    
    P_apertura = np.zeros((M, M), dtype=complex)
    P_apertura[R_p < R_pupila] = 1.0
    
    # Tipos de filtro (campo claro y oscuro)
    if tipo_filtro == "campo claro":
        P_pupila = P_apertura
        
    elif tipo_filtro == "campo oscuro":
        P_bloqueo = np.ones((M,M), dtype=complex)
        P_bloqueo[R_p < R_bloqueo] = 0.0 # Creamos el circulo negro
        
        P_pupila = P_apertura * P_bloqueo
    
    return P_pupila


def simular_con_convolucion(objeto, L_objeto, M, long_onda, f_MO, f_TL, 
                             tipo_filtro, R_pupila, R_bloqueo=0):

    # Necesitamos el tamaño del plano pupila (L_pupila_calc) para
    # crear la pupila P(x,y) a la escala correcta
    dx_objeto = L_objeto / M
    L_pupila_calc = (long_onda * f_MO) / dx_objeto

    # Creamos la FT (que es la Pupila P(x,y)) 
    P_pupila_TF = pupila(M, L_pupila_calc, tipo_filtro, R_pupila, R_bloqueo)
    
    # Calculamos la Convolucion

    # (ifftshift centra el objeto antes de la FFT)
    S_fft = np.fft.fft2(np.fft.ifftshift(objeto))
    
    # (fftshift centra la OTF/Pupila para que coincida con S_fft)
    E_cam_fft = S_fft * np.fft.fftshift(P_pupila_TF)
    
    # (fftshift deshace el ifftshift inicial)
    campo_cam_convolucion = np.fft.fftshift(np.fft.ifft2(E_cam_fft))

    # La magnificación del sistema
    Mag_total = f_TL / f_MO
    L_cam = L_objeto * Mag_total
    
    return campo_cam_convolucion, L_cam, P_pupila_TF, L_pupila_calc



# Definimos los Parámetros
long_onda = 550e-9      # Longitud de onda (550 nm)
f_MO = 9e-3             # Focal Objetivo (9 mm para un 20x)
f_TL = 180e-3           # Focal Lente de Tubo (180 mm)
NA = 0.4                # Apertura Numérica del objetivo (0.4 para 20x)

# Parámetros de la Cámara (Basler)
dx_real_camara = 2.74e-6/20 # pixel size de 2.74 µm/20 que es el aumento del MO

# Ruta de la Muestra 
ruta_imagen = r"C:\Users\pauli\OneDrive\Documents\Universidad\Instrumentos-opticos\practica_3\imagenes\Siemens_star.png" 

# Cargamos la imagen del test
objeto, M = subir_imagen(ruta_imagen)

# Tamaño del objeto
L_objeto = (dx_real_camara * M * f_MO) / f_TL

# Magnificación total del sistema
Mag_total = f_TL / f_MO

# Radio físico de la pupila
R_pupila = NA * f_MO
print(f"      Radio de la Pupila P(x,y): {R_pupila * 1000:.2f} mm (calculado con NA)")


# Parametros de los filtros
R_bloqueo_co = R_pupila * 0.3


# Campo claro
campo_cc, L_cc, pupila_cc, L_pupila = simular_con_convolucion(
    objeto, L_objeto, M, long_onda, f_MO, f_TL,
    tipo_filtro='campo claro', 
    R_pupila=R_pupila
)

# Campo oscuro
campo_co, L_co, pupila_co, _ = simular_con_convolucion(
    objeto, L_objeto, M, long_onda, f_MO, f_TL, 
    tipo_filtro='campo oscuro', 
    R_pupila=R_pupila,
    R_bloqueo=R_bloqueo_co
)

# Simulamos
int_cc = np.abs(campo_cc)**2 #Intensidad campo claro
int_co = np.abs(campo_co)**2 #Intensidad campo oscuro

print("\nMostrando resultados...")

plt.figure(figsize=(18, 6)) # Hacemos la figura más ancha
plt.suptitle("Comparación: Campo Claro vs. Campo Oscuro (Método Convolución)", fontsize=16)

# Figura 1: Test de resolucion
ext_obj = [-L_objeto/2 * 1e6, L_objeto/2 * 1e6, -L_objeto/2 * 1e6, L_objeto/2 * 1e6]
plt.subplot(1, 3, 1)
plt.imshow(np.abs(objeto), cmap='gray', extent=ext_obj)
plt.title('Objeto Original S(ξ,η)')
plt.xlabel('ξ (μm)')
plt.ylabel('η (μm)')

# Figura 2: Imagen de Campo Claro
ext_cam = [-L_cc/2 * 1e6, L_cc/2 * 1e6, -L_cc/2 * 1e6, L_cc/2 * 1e6]
plt.subplot(1, 3, 2)
plt.imshow(int_cc, cmap='gray', extent=ext_cam)
plt.title('Imagen Simulada (Campo Claro)')
plt.xlabel('u (μm)')
plt.ylabel('v (μm)')

# Figura 3: Imagen de Campo Oscuro
plt.subplot(1, 3, 3)
plt.imshow(np.log1p(int_co), cmap='gray', extent=ext_cam)
plt.title('Imagen Simulada (Campo Oscuro, log)')
plt.xlabel('u (μm)')
plt.ylabel('v (μm)')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Figura de las Pupilas
plt.figure(figsize=(12, 5))
ext_pup = [-L_pupila/2 * 1000, L_pupila/2 * 1000, -L_pupila/2 * 1000, L_pupila/2 * 1000]
zoom_lim = R_pupila * 1.2 * 1000 

plt.subplot(1, 2, 1)
# --- LÍNEA CORREGIDA (sin np.fft.fftshift) ---
plt.imshow(np.abs(pupila_cc), cmap='gray', extent=ext_pup)
plt.title('TF P(x,y) - Campo Claro')
plt.xlabel('x (mm)')
plt.ylabel('y (mm)')
plt.xlim(-zoom_lim, zoom_lim)
plt.ylim(-zoom_lim, zoom_lim)

plt.subplot(1, 2, 2)
# --- LÍNEA CORREGIDA (sin np.fft.fftshift) ---
plt.imshow(np.abs(pupila_co), cmap='gray', extent=ext_pup)
plt.title('TF P(x,y) - Campo Oscuro')
plt.xlabel('x (mm)')
plt.ylabel('y (mm)')
plt.xlim(-zoom_lim, zoom_lim)
plt.ylim(-zoom_lim, zoom_lim)

plt.tight_layout()
plt.show()