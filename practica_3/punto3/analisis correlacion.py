import numpy as np
import matplotlib.pyplot as plt
import io 
from scipy import ndimage
from scipy.stats import pearsonr # <-- ¡NUEVA IMPORTACIÓN!
from skimage.transform import resize # <-- ¡NUEVA IMPORTACIÓN!


# Definicion de funciones a usar

def cargar_muestra_compleja(ruta_archivo):
    with open(ruta_archivo, 'r') as f:
        contenido_texto = f.read()
        
    contenido_texto_j = contenido_texto.replace('i', 'j')
    f_virtual = io.StringIO(contenido_texto_j)
    
    objeto = np.loadtxt(f_virtual,
        dtype=np.complex128,
        delimiter=',')
    
    f_virtual.close()

    M = objeto.shape[0]
    print(f"Tamaño de la malla (M): {M}x{M}")
    return objeto, M


def pupila(M, L_pupila_calc, tipo_filtro, R_pupila, R_bloqueo=0, R_punto_fase=0, atenuacion=0.0, fase_stop=0.0):
    
    # Creamos la rejilla de coordenadas de la pupila
    coords_pupila = np.linspace(-L_pupila_calc/2, L_pupila_calc/2, M)
    X_p, Y_p = np.meshgrid(coords_pupila, coords_pupila)
    R_p = np.sqrt(X_p**2 + Y_p**2)
    
    P_apertura = np.zeros((M, M), dtype=complex)
    P_apertura[R_p < R_pupila] = 1.0+0.0j 
    
    if tipo_filtro == "campo_claro":
        P_pupila = P_apertura

    elif tipo_filtro == "campo_oscuro_variable":
        P_bloqueo_mask = np.ones((M,M), dtype=complex)
        valor_stop = atenuacion * np.exp(1j * fase_stop)
        P_bloqueo_mask[R_p < R_bloqueo] = valor_stop 
        P_pupila = P_apertura * P_bloqueo_mask
        
    elif tipo_filtro == "contraste_fase":
        P_pupila = P_apertura.copy() 
        mascara_punto_dc = (R_p < R_punto_fase)
        P_pupila[mascara_punto_dc] = 1j 
        
    return P_pupila

def simular_con_convolucion(objeto, L_objeto, M, long_onda, f_MO, f_TL, 
                            tipo_filtro, R_pupila, R_bloqueo=0, 
                            R_punto_fase=0, atenuacion=0.0, fase_stop=0.0):
    
    dx_objeto = L_objeto / M
    L_pupila_calc = (long_onda * f_MO) / dx_objeto
    
    P_pupila_OTF = pupila(M, L_pupila_calc, tipo_filtro, R_pupila, 
                          R_bloqueo, R_punto_fase, atenuacion, fase_stop)
    
    S_fft = np.fft.fft2(np.fft.ifftshift(objeto))
    
    E_cam_fft = S_fft * np.fft.fftshift(P_pupila_OTF)
    
    campo_cam_convolucion = np.fft.fftshift(np.fft.ifft2(E_cam_fft))
    
    Mag_total = f_TL / f_MO
    L_cam = L_objeto * Mag_total
    
    return campo_cam_convolucion, L_cam, P_pupila_OTF, L_pupila_calc



# Definimos los Parámetros
long_onda = 533e-9 # Ajuste para que se vea bien la fase original
f_MO = 9e-3
f_TL = 200e-3
NA = 0.5
dx_real_camara = 2.74e-6

ruta_muestra = r"C:\Users\david\OneDrive\Desktop\MuestrasBio\MuestraBio_E06.csv" 
objeto, M = cargar_muestra_compleja(ruta_muestra)

L_objeto = 390e-6 # 390 µm 
print(f"Lado Físico de Muestra (L_objeto): {L_objeto * 1e6:.2f} µm")

Mag_total = f_TL / f_MO
R_pupila = NA * f_MO

# --- Parámetros Óptimos Fijos para el filtro ---
atenuacion_co = 0.7       
R_bloqueo_co_var = R_pupila * 0.01 # Usamos un radio del 1%
fase_stop_co = np.pi / 2 # Desfase óptimo de 90°


# Simulamos la "mejor imagen"
print("\n--- SIMULANDO LA IMAGEN ÓPTIMA (90° desfase, 1% radio) ---")

campo_optimo, L_cam_optimo, pupila_optima, L_pupila_optima = simular_con_convolucion(
    objeto, L_objeto, M, long_onda, f_MO, f_TL, 
    tipo_filtro='campo_oscuro_variable', 
    R_pupila=R_pupila,
    R_bloqueo=R_bloqueo_co_var, 
    atenuacion=atenuacion_co,   
    fase_stop=fase_stop_co       
)

int_optima = np.abs(campo_optimo)**2

# Normalizamos la intensidad
max_intensidad_optima = np.max(int_optima)
if max_intensidad_optima > 0:
    intensidad_log_optima = np.log1p(int_optima / max_intensidad_optima * 10)
    intensidad_norm_optima = intensidad_log_optima / np.max(intensidad_log_optima)
else:
    intensidad_norm_optima = int_optima
    
# --- Preparar la Fase Original para Comparación ---
fase_original_recortada = np.angle(objeto)

# Voy a normalizar la fase original a [0, 1] para la comparación visual y cuantitativa.
fase_original_norm = (fase_original_recortada - np.min(fase_original_recortada)) / \
                     (np.max(fase_original_recortada) - np.min(fase_original_recortada))


# --- CÁLCULO DE MÉTRICAS CUANTITATIVAS para la IMAGEN ÓPTIMA ---
metrica_contraste_optima = np.std(intensidad_norm_optima)
metrica_detalle_optima = ndimage.laplace(intensidad_norm_optima).var()

# --- CÁLCULO DE MÉTRICA DE SIMILITUD (CORRELACIÓN) ---
# Variable para almacenar la matriz de fase final (redimensionada o no)
fase_para_correlacion = fase_original_norm

if fase_original_norm.shape != intensidad_norm_optima.shape:
    print("\nADVERTENCIA: Las dimensiones de la fase original y la imagen simulada no coinciden.")
    print(f"Redimensionando fase {fase_original_norm.shape} -> {intensidad_norm_optima.shape}")
    
    fase_para_correlacion = resize(fase_original_norm, intensidad_norm_optima.shape, anti_aliasing=True)
    correlacion, _ = pearsonr(fase_para_correlacion.flatten(), intensidad_norm_optima.flatten())
else:
    correlacion, _ = pearsonr(fase_original_norm.flatten(), intensidad_norm_optima.flatten())


print("\n--- ANÁLISIS CUANTITATIVO DE LA IMAGEN ÓPTIMA ---")
print(f"Contraste (std): {metrica_contraste_optima:.4f}")
print(f"Detalle (Lap-Var): {metrica_detalle_optima:e}")
print(f"Correlación de Pearson con Fase Original: {correlacion:.4f}")


# Figura 1: Comparación de Fase Original vs. Imagen Óptima 
plt.figure(figsize=(14, 7)) 

# Subtítulo principal de la figura
plt.suptitle(f"Comparación Cuantitativa: Fase Original vs. Imagen Óptima (Desfase=90°, Radio={R_bloqueo_co_var/R_pupila*100:.1f}%)", fontsize=16)

ext_obj = [-L_objeto/2 * 1e6, L_objeto/2 * 1e6, -L_objeto/2 * 1e6, L_objeto/2 * 1e6]
ext_cam = [-L_cam_optimo/2 * 1e6, L_cam_optimo/2 * 1e6, -L_cam_optimo/2 * 1e6, L_cam_optimo/2 * 1e6]


# Gráfico 1: Fase Original
ax1 = plt.subplot(1, 2, 1)
im1 = plt.imshow(fase_original_norm, cmap='gray', extent=ext_obj, vmin=0, vmax=1) # Normalizamos a 0-1
plt.title('Fase Original Normalizada de la Muestra')
plt.xlabel('ξ (μm)')
plt.ylabel('η (μm)')
plt.colorbar(im1, ax=ax1, label='Fase (Normalizada)')


# Gráfico 2: Imagen Óptima (Intensidad Normalizada)
ax2 = plt.subplot(1, 2, 2)
im2 = plt.imshow(intensidad_norm_optima, cmap='gray', extent=ext_cam, vmin=0, vmax=1)
plt.title('Imagen Óptima (Filtro de Campo Oscuro Variable)')
plt.xlabel('u (μm)')
plt.ylabel('v (μm)')
plt.colorbar(im2, ax=ax2, label='Intensidad (Normalizada)')

# Añadir las métricas
ax2.set_title(
    f'Imagen contraste de fase\n'
    #f'Contraste (std): {metrica_contraste_optima:.4f}\n'
    #f'Detalle (Lap-Var): {metrica_detalle_optima:e}\n'
    f'Correlación con Fase Original: {correlacion:.4f}'
)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


# Figura 2: Grafico de correlacion

# Preparamos los datos para el scatter plot
fase_flat = fase_para_correlacion.flatten()
intensidad_flat = intensidad_norm_optima.flatten()

# Hacemos un submuestreo aleatorio para no graficar millones de puntos
num_pixeles_plot = 5000
total_pixeles = len(fase_flat)
indices = np.random.choice(total_pixeles, num_pixeles_plot, replace=False)

fase_sample = fase_flat[indices]
intensidad_sample = intensidad_flat[indices]

# Graficamos
plt.figure(figsize=(8, 6))
plt.scatter(fase_sample, intensidad_sample, alpha=0.1, s=5)
plt.title(f'Correlación Pixel a Pixel (Muestra de {num_pixeles_plot} puntos)\nCoeficiente de Pearson (r) = {correlacion:.4f}', fontsize=14)
plt.xlabel('Fase Original (Normalizada)', fontsize=12)
plt.ylabel('Intensidad Final (Normalizada)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)


plt.tight_layout()
plt.show()