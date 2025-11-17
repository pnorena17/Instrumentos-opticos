import numpy as np
import matplotlib.pyplot as plt
import io 


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
    # Calcular la distancia radial de cada píxel al centro
    R_p = np.sqrt(X_p**2 + Y_p**2)
    
    # Creamos la apertura: 1.0 adentro del radio, 0.0 afuera
    P_apertura = np.zeros((M, M), dtype=complex)
    P_apertura[R_p < R_pupila] = 1.0+0.0j # 1.0 para transmisión total
    
    if tipo_filtro == "campo_claro":
        # Esta es la "Pupila Circular" que pediste ver
        P_pupila = P_apertura


    elif tipo_filtro == "campo_oscuro_variable":
        
        # Filtro de campo oscuro con atenuación y fase variables
        P_bloqueo_mask = np.ones((M,M), dtype=complex)
        
        # Calculamos el valor complejo del stop central
        valor_stop = atenuacion * np.exp(1j * fase_stop)
        
        # Aplicamos el stop usando R_bloqueo
        P_bloqueo_mask[R_p < R_bloqueo] = valor_stop 
        P_pupila = P_apertura * P_bloqueo_mask
        
    elif tipo_filtro == "contraste_fase":
        # Filtro Zernike-punto (idealizado) para comparar
        P_pupila = P_apertura.copy() 
        # Aplicamos el stop usando R_punto_fase
        mascara_punto_dc = (R_p < R_punto_fase)
        P_pupila[mascara_punto_dc] = 1j 
        
    return P_pupila

def simular_con_convolucion(objeto, L_objeto, M, long_onda, f_MO, f_TL, 
                            tipo_filtro, R_pupila, R_bloqueo=0, 
                            R_punto_fase=0, atenuacion=0.0, fase_stop=0.0):
    
    # Calculamos la escala del plano de la pupila
    dx_objeto = L_objeto / M
    L_pupila_calc = (long_onda * f_MO) / dx_objeto
    
    # Creamos la pupila
    P_pupila_OTF = pupila(M, L_pupila_calc, tipo_filtro, R_pupila, 
                          R_bloqueo, R_punto_fase, atenuacion, fase_stop)
    
    # Simulamos la propagación y el filtrado
    S_fft = np.fft.fft2(np.fft.ifftshift(objeto))
    
    # Pupila -> Multiplicacion -> Camara
    E_cam_fft = S_fft * np.fft.fftshift(P_pupila_OTF)
    
    # Camara -> IFFT -> Imagen final
    campo_cam_convolucion = np.fft.fftshift(np.fft.ifft2(E_cam_fft))
    
    # Calculamos magnificacion y escala de la imagen final
    Mag_total = f_TL / f_MO
    L_cam = L_objeto * Mag_total
    
    return campo_cam_convolucion, L_cam, P_pupila_OTF, L_pupila_calc



# Definimos los Parámetros
long_onda = 550e-9
f_MO = 9e-3
f_TL = 180e-3
NA = 0.4
dx_real_camara = 2.74e-6

ruta_muestra = r"C:\Users\user\Desktop\MuestrasBio\MuestraBio_E03.csv" 
objeto, M = cargar_muestra_compleja(ruta_muestra)

L_objeto = 390e-6 # 390 µm 
print(f"Lado Físico de Muestra (L_objeto): {L_objeto * 1e6:.2f} µm")

# Magnificación total del sistema
Mag_total = f_TL / f_MO

# Radio físico de la pupila
R_pupila = NA * f_MO

# Parámetros para los filtros
R_punto_fase_cf = R_pupila * 0.05       # Radio del stop (5% de la pupila)
R_bloqueo_co_var = R_punto_fase_cf      # El radio es el mismo

atenuacion_co = 0.8      # Atenuación (transparencia) del stop (ej. 30%)
fase_stop_co = np.pi / 2 # Desfase (grosor) del stop (90 grados)


# Simulamos

# Campo Claro
campo_cc, L_cc, pupila_cc, L_pupila = simular_con_convolucion(
    objeto, L_objeto, M, long_onda, f_MO, f_TL, 
    tipo_filtro='campo_claro', 
    R_pupila=R_pupila
)

# "Campo Oscuro Variable"
campo_co_var, L_co, pupila_co_var, _ = simular_con_convolucion(
    objeto, L_objeto, M, long_onda, f_MO, f_TL, 
    tipo_filtro='campo_oscuro_variable', 
    R_pupila=R_pupila,
    R_bloqueo=R_bloqueo_co_var, 
    atenuacion=atenuacion_co,
    fase_stop=fase_stop_co
)

# Contraste de Fase (Zernike)
campo_cf, L_cf, pupila_cf, _ = simular_con_convolucion(
    objeto, L_objeto, M, long_onda, f_MO, f_TL, 
    tipo_filtro='contraste_fase', 
    R_pupila=R_pupila,
    R_punto_fase=R_punto_fase_cf
)

# Simulamos las intensidades
int_co_var = np.abs(campo_co_var)**2
int_cf = np.abs(campo_cf)**2

# Normalizamos las intensidades para mejor visualización
max_intensidad_co_var = np.max(int_co_var)
max_intensidad_cf = np.max(int_cf)

#Buscamos una intensidad para que se vea bien 
if max_intensidad_co_var > 0:
    intensidad_log_co = np.log1p(int_co_var / max_intensidad_co_var * 10)
    intensidad_norm_co = intensidad_log_co / np.max(intensidad_log_co)
else:
    intensidad_norm_co = int_co_var
    
    
if max_intensidad_cf > 0:
    intensidad_log_cf = np.log1p(int_cf / max_intensidad_cf * 10)
    intensidad_norm_cf = intensidad_log_cf / np.max(intensidad_log_cf)
else:
    intensidad_norm_cf = int_cf

print("\nMostrando resultados...")


# Figura 1: Imágenes Simuladas
plt.figure(figsize=(12, 6)) 
plt.suptitle("Comparación de Métodos de Fase (Intensidad)", fontsize=16)
ext_cam = [-L_co/2 * 1e6, L_co/2 * 1e6, -L_co/2 * 1e6, L_co/2 * 1e6]

plt.subplot(1, 2, 1)
plt.imshow(intensidad_norm_co, cmap='gray', extent=ext_cam, vmin=0, vmax=1) 
plt.title('Imagen (Contraste de Fase)')
plt.xlabel('u (μm)')
plt.ylabel('v (μm)')

plt.subplot(1, 2, 2) 
plt.imshow(intensidad_norm_cf, cmap='gray', extent=ext_cam, vmin=0, vmax=1)
plt.title('Imagen (Zernike)')
plt.xlabel('u (μm)')
plt.ylabel('v (μm)')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show() 


# Figura 2: Pupilas

plt.figure(figsize=(12, 5))
plt.suptitle("Comparación de Amplitud de Pupilas de Fase", fontsize=16)
ext_pup = [-L_pupila/2 * 1000, L_pupila/2 * 1000, -L_pupila/2 * 1000, L_pupila/2 * 1000]
zoom_lim = R_pupila * 1.2 * 1000

plt.subplot(1, 2, 1)

# Mostramos la Amplitud. El stop central se verá en escala de grises.
plt.imshow(np.abs(pupila_co_var), cmap='gray', extent=ext_pup, vmin=0, vmax=1)
plt.title('Pupila (Amplitud)')
plt.xlabel('x (mm)')
plt.ylabel('y (mm)')
plt.xlim(-zoom_lim, zoom_lim)
plt.ylim(-zoom_lim, zoom_lim)
plt.colorbar(label='Atenuación (Amplitud)')

plt.subplot(1, 2, 2)

plt.imshow(np.angle(pupila_cf), cmap='twilight', extent=ext_pup, vmin=-np.pi, vmax=np.pi)
plt.title('Pupila Zernike (Fase)')
plt.xlabel('x (mm)')
plt.ylabel('y (mm)')
plt.xlim(-zoom_lim, zoom_lim)
plt.ylim(-zoom_lim, zoom_lim)

plt.tight_layout()
plt.show()