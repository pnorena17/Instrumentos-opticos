import numpy as np
import matplotlib.pyplot as plt
import io 
from scipy import ndimage # <-- ¡NUEVA IMPORTACIÓN!


# Definicion de funciones a usar

def cargar_muestra_compleja(ruta_archivo):
    # Esta función está perfecta, no se toca
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
long_onda = 533e-9
f_MO = 9e-3
f_TL = 200e-3
NA = 0.5
dx_real_camara = 2.74e-6

ruta_muestra = r"C:\Users\david\OneDrive\Desktop\MuestrasBio\MuestraBio_E06.csv" 
objeto, M = cargar_muestra_compleja(ruta_muestra)

L_objeto = 390e-6 # 390 µm 
print(f"Lado Físico de Muestra (L_objeto): {L_objeto * 1e6:.2f} µm")

# Magnificación total del sistema
Mag_total = f_TL / f_MO

# Radio físico de la pupila
R_pupila = NA * f_MO

# Parámetros para el barrido 
atenuacion_co = 0.7       # Atenuación (transparencia) del stop 
# (Usamos un radio del 5%)
R_bloqueo_co_var = R_pupila * 0.05 



# Definimos los desfases que queremos probar (en radianes)

barrido_fases = np.linspace(np.pi/3, 2*np.pi/3, 3)

# Listas para guardar los resultados
resultados_intensidad = []
resultados_pupilas = []
resultados_contraste = [] # Lista para métrica de contraste


print("\n--- INICIANDO BARRIDO DE DESFASES ---")

for fase_actual in barrido_fases:
    print(f"Simulando para fase_stop = {fase_actual/np.pi:.2f} * pi ({np.degrees(fase_actual):.0f} deg)")
    
    # Ejecutamos la simulación con el desfase actual
    campo_co_var, L_co, pupila_co_var, L_pupila = simular_con_convolucion(
        objeto, L_objeto, M, long_onda, f_MO, f_TL, 
        tipo_filtro='campo_oscuro_variable', 
        R_pupila=R_pupila,
        R_bloqueo=R_bloqueo_co_var, # Radio
        atenuacion=atenuacion_co,   # Atenuación 
        fase_stop=fase_actual       # Variable del barrido
    )
    
    # Simulamos la intensidad
    int_co_var = np.abs(campo_co_var)**2
    
    # Normalizamos la intensidad 
    max_intensidad_co_var = np.max(int_co_var)
    if max_intensidad_co_var > 0:
        intensidad_log_co = np.log1p(int_co_var / max_intensidad_co_var * 10)
        intensidad_norm_co = intensidad_log_co / np.max(intensidad_log_co)
    else:
        intensidad_norm_co = int_co_var
    

    # Usamos la imagen normalizada 'intensidad_norm_co' que es la que se grafica
    
    # Métrica de Contraste
    metrica_contraste = np.std(intensidad_norm_co)
    

    # Guardamos los resultados
    resultados_intensidad.append(intensidad_norm_co)
    resultados_pupilas.append(pupila_co_var)
    resultados_contraste.append(metrica_contraste)


print("--- BARRIDO COMPLETO ---")


# Simulamos el Zernike Ideal para comparar
R_punto_fase_cf = R_pupila * 0.05 # Usamos el 5% como referencia
campo_cf, L_cf, pupila_cf, _ = simular_con_convolucion(
    objeto, L_objeto, M, long_onda, f_MO, f_TL, 
    tipo_filtro='contraste_fase', 
    R_pupila=R_pupila,
    R_punto_fase=R_punto_fase_cf
)
# (El resto de esta simulación es solo para la Figura 3, la dejamos igual)
int_cf = np.abs(campo_cf)**2
max_intensidad_cf = np.max(int_cf)
if max_intensidad_cf > 0:
    intensidad_log_cf = np.log1p(int_cf / max_intensidad_cf * 10)
    intensidad_norm_cf = intensidad_log_cf / np.max(intensidad_log_cf)
else:
    intensidad_norm_cf = int_cf

print("\nMostrando resultados...")


# Figura 1: Original vs Zernike Ideal
plt.figure(figsize=(12, 6)) 
plt.suptitle("Comparación de Referencia", fontsize=16)
ext_obj = [-L_objeto/2 * 1e6, L_objeto/2 * 1e6, -L_objeto/2 * 1e6, L_objeto/2 * 1e6]
ext_cam = [-L_co/2 * 1e6, L_co/2 * 1e6, -L_co/2 * 1e6, L_co/2 * 1e6]

# Graficamos la Fase Original (la "verdad")
plt.subplot(1, 2, 1)
plt.imshow(np.angle(objeto), cmap='twilight', extent=ext_obj) 

plt.title('Fase Original de la Muestra')
plt.xlabel('ξ (μm)')
plt.ylabel('η (μm)')
plt.colorbar(label='Fase (rad)')

# Graficamos el Zernike Ideal (el "blanco")
plt.subplot(1, 2, 2) 
# Mostramos el mejor resultado del barrido (el de 90°)
idx_90_grados = barrido_fases.tolist().index(np.pi/2)
plt.imshow(resultados_intensidad[idx_90_grados], cmap='gray', extent=ext_cam, vmin=0, vmax=1)
plt.title('Imagen (Mejor resultado del barrido a 90°)')
plt.xlabel('u (μm)')
plt.ylabel('v (μm)')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show() 


# Figura 2: Barrido de fases
N_fases = len(barrido_fases)
# Ajusta el tamaño de la figura según cuántos items tengas
plt.figure(figsize=(N_fases * 4, 6)) # Un poco más alta para el texto
plt.suptitle(f"Barrido de Desfase del Stop (Radio = {R_bloqueo_co_var/R_pupila*100:.1f}%, Atenuación = {atenuacion_co*100:.0f}%)", fontsize=16)

for i in range(N_fases):
    ax = plt.subplot(1, N_fases, i + 1)
    
    # Extraemos todos los datos de este paso
    intensidad_actual = resultados_intensidad[i]
    fase_actual = barrido_fases[i]
    contraste_actual = resultados_contraste[i]

    
    # Graficamos la imagen de intensidad
    plt.imshow(intensidad_actual, cmap='gray', extent=ext_cam, vmin=0, vmax=1)
    
    # Titulos
    titulo = (
        f"Desfase = {np.degrees(fase_actual):.0f}°\n"
        f"Contraste (std): {contraste_actual:.4f}\n"
        
    )
    plt.title(titulo)
    
    plt.xlabel('u (μm)')
    if i == 0:
        plt.ylabel('v (μm)')
    else:
        ax.set_yticklabels([]) 

plt.tight_layout(rect=[0, 0.03, 1, 0.9])
plt.show()

