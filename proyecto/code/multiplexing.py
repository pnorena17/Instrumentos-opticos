import numpy as np
import math
import matplotlib.pyplot as plt
from encript_image import encriptar_drpe, desencriptar_drpe

def dividir_en_bloques_iguales(imagen, filas, cols):
    """Divide la imagen en bloques para procesarlos por separado."""
    alto, ancho = imagen.shape
    h_bloque = math.ceil(alto / filas)
    w_bloque = math.ceil(ancho / cols)
    
    bloques = []
    datos_bloques = [] 
    
    for i in range(filas):
        for j in range(cols):
            y_inicio = i * h_bloque
            x_inicio = j * w_bloque
            
            # Recorte con manejo de límites
            recorte = imagen[y_inicio : min(y_inicio + h_bloque, alto), 
                             x_inicio : min(x_inicio + w_bloque, ancho)]
            
            # Padding para uniformidad de tamaño
            h_actual, w_actual = recorte.shape
            pad_h = h_bloque - h_actual
            pad_w = w_bloque - w_actual
            
            if pad_h > 0 or pad_w > 0:
                recorte = np.pad(recorte, ((0, pad_h), (0, pad_w)), mode='constant')
            
            bloques.append(recorte)
            
            datos_bloques.append({
                'bloque_original': recorte,
                'idx': len(bloques)-1,
                'coords': (i, j, h_actual, w_actual) 
            })
            
    return bloques, datos_bloques, (h_bloque, w_bloque)

def multiplexar_imagen_en_partes(imagen_grande, filas_grid=2, cols_grid=2, 
                                 radio_pupila=None, dx=None, long_onda=None, foco=None, logico_qr=0):
    """
    Encripta por partes usando parámetros físicos.
    """
    bloques, datos_bloques, (h_b, w_b) = dividir_en_bloques_iguales(imagen_grande, filas_grid, cols_grid)
    
    campo_total = np.zeros((h_b, w_b), dtype=complex)
    banco_llaves = []
    
    print(f"Multiplexando: Dividiendo imagen en {filas_grid}x{cols_grid} bloques de {h_b}x{w_b} px.")
    
    for item in datos_bloques:
        bloque = item['bloque_original']
        
        # --- AQUÍ SÍ SE USAN LOS PARÁMETROS FÍSICOS ---
        img_encriptada, k1, k2, pupila = encriptar_drpe(
            bloque, 
            radio_pupila=radio_pupila,
            dx=dx, 
            long_onda=long_onda, 
            foco=foco,
            matriz_mascara=logico_qr
        )
        
        campo_total += img_encriptada
        
        banco_llaves.append({
            'idx': item['idx'],
            'k1': k1,
            'k2': k2,
            'coords': item['coords'],
            'original': bloque
        })
        
    return campo_total, banco_llaves, (imagen_grande.shape)


def desencriptar_y_reconstruir(paquete_optico, banco_llaves, dim_original, dx=6.5e-6, ver_paso_a_paso=False):
    """
    Recupera la imagen sumada.
    AHORA RECIBE 'dx' para poder calcular las escalas en mm.
    """
    alto_tot, ancho_tot = dim_original
    imagen_reconstruida = np.zeros((alto_tot, ancho_tot), dtype=int)
    
    print(f"--- Recuperando y Ensamblando ({len(banco_llaves)} partes) ---")

    if ver_paso_a_paso:
        plt.figure(figsize=(16, 5))
        n_cols = len(banco_llaves) + 1
        
        # --- CÁLCULO DE ESCALAS FÍSICAS ---
        h_enc, w_enc = paquete_optico.shape
        L_x = w_enc * dx * 1000 # ancho en mm
        L_y = h_enc * dx * 1000 # alto en mm
        extent_enc = [-L_x/2, L_x/2, -L_y/2, L_y/2]
        
        ax = plt.subplot(1, n_cols, 1)
        im = ax.imshow(np.abs(paquete_optico), cmap='gray', extent=extent_enc)
        ax.set_title("Paquete Óptico\n(Multiplexado)")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Amplitud')
    
    for i, item in enumerate(banco_llaves):
        k1 = item['k1']
        k2 = item['k2']
        r, c, h_real, w_real = item['coords']
        
        # Desencriptamos (Nota: desencriptar no suele requerir parámetros físicos extra 
        # a menos que simules propagación inversa, pero si tu función desencriptar_drpe 
        # los pide, agrégalos aquí también).
        campo_recuperado = desencriptar_drpe(paquete_optico, k1, k2)
        
        img_ruidosa = np.abs(campo_recuperado)

        # Normalización y Limpieza
        vmin, vmax = img_ruidosa.min(), img_ruidosa.max()
        img_norm = (img_ruidosa - vmin) / (vmax - vmin) if vmax > vmin else img_ruidosa
            
        p1, p99 = np.percentile(img_ruidosa, (1, 99))
        img_clipped = np.clip(img_ruidosa, p1, p99)
        
        if p99 > p1:
            img_norm = (img_clipped - p1) / (p99 - p1)
        else:
            img_norm = img_clipped

        # Umbralización
        bloque_limpio = np.where(img_norm > 0.6, 1, 0)
        
        if ver_paso_a_paso:
            # Escala para sub-bloques
            h_bl, w_bl = bloque_limpio.shape
            L_x_bl = w_bl * dx * 1000
            L_y_bl = h_bl * dx * 1000
            extent_bl = [-L_x_bl/2, L_x_bl/2, -L_y_bl/2, L_y_bl/2]

            ax = plt.subplot(1, n_cols, i + 2)
            im_part = ax.imshow(bloque_limpio, cmap='gray', extent=extent_bl)
            ax.set_title(f"Parte {i+1}\n(Pos: {r},{c})")
            ax.set_xlabel("x (mm)")
            # Barra de color discreta/binaria
            plt.colorbar(im_part, ax=ax, fraction=0.046, pad=0.04)
        
        # Reconstrucción del mosaico
        h_bloque, w_bloque = img_ruidosa.shape
        y_pos = r * h_bloque
        x_pos = c * w_bloque
        
        bloque_util = bloque_limpio[0:h_real, 0:w_real]
        imagen_reconstruida[y_pos : y_pos + h_real, x_pos : x_pos + w_real] = bloque_util
        
    if ver_paso_a_paso:
        plt.tight_layout()
        plt.show()
        
    return imagen_reconstruida