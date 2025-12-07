import numpy as np
import math
import cv2
import matplotlib.pyplot as plt # Necesario para graficar
from encript_image import encriptar_drpe, desencriptar_drpe


def dividir_en_bloques_iguales(imagen, filas, cols):
    # Dividimos la imagen en "bloques" pequeños
    alto, ancho = imagen.shape
    h_bloque = math.ceil(alto / filas)
    w_bloque = math.ceil(ancho / cols)
    
    bloques = []
    
    # Datos para reconstrucción
    datos_bloques = [] 
    
    for i in range(filas):
        for j in range(cols):
            y_inicio = i * h_bloque
            x_inicio = j * w_bloque
            
            # Recorte (con manejo de bordes)
            recorte = imagen[y_inicio : min(y_inicio + h_bloque, alto), 
                             x_inicio : min(x_inicio + w_bloque, ancho)]
            
            # Padding si el bloque quedó más pequeño (bordes derechos/inferiores)
            # Para sumar campos ópticos, todos deben medir lo mismo
            h_actual, w_actual = recorte.shape
            
            pad_h = h_bloque - h_actual
            pad_w = w_bloque - w_actual
            
            if pad_h > 0 or pad_w > 0:
                recorte = np.pad(recorte, ((0, pad_h), (0, pad_w)), mode='constant')
            
            bloques.append(recorte)
            
            datos_bloques.append({
                'bloque_original': recorte, # Guardamos el original para comparar visualmente
                'idx': len(bloques)-1,
                'coords': (i, j, h_actual, w_actual) 
            })
            
    return bloques, datos_bloques, (h_bloque, w_bloque)

def multiplexar_imagen_en_partes(imagen_grande, filas_grid=2, cols_grid=2, 
                                 radio_pupila=None, dx=None, long_onda=None, foco=None, logico_qr = 0):
    """
    Toma una imagen (QR grande), la divide y encripta cada parte sumando los campos.
    Ahora soporta parámetros FÍSICOS.
    """
    # Tomamos el QR y lo dividimos para encriptarlo y "sumarlo" (multiplexing)
    
    # Dividimos
    bloques, datos_bloques, (h_b, w_b) = dividir_en_bloques_iguales(imagen_grande, filas_grid, cols_grid)
    
    campo_total = np.zeros((h_b, w_b), dtype=complex)
    banco_llaves = []
    
    print(f"Multiplexando: Dividiendo imagen en {filas_grid}x{cols_grid} bloques de {h_b}x{w_b} px.")
    
    for item in datos_bloques:
        bloque = item['bloque_original']
        
        # Encriptamos cada bloque individualmente
        
        img_encriptada, k1, k2, pupila = encriptar_drpe(
            bloque, 
            radio_pupila=radio_pupila,
            dx=dx, 
            long_onda=long_onda, 
            foco=foco,
            matriz_mascara = logico_qr
        )
        
        # Superposición de campos
        campo_total += img_encriptada
        
        # Guardamos llaves
        banco_llaves.append({
            'idx': item['idx'],
            'k1': k1,
            'k2': k2,
            'coords': item['coords'],
            'original': bloque # Guardamos referencia para comparar
        })
        
    return  campo_total, banco_llaves, (imagen_grande.shape)


def desencriptar_y_reconstruir(paquete_optico, banco_llaves, dim_original, ver_paso_a_paso=False):
    
    # Tomamos cada paquete superpuesto y extraemos una imagen con su respectiva llave
    alto_tot, ancho_tot = dim_original
    
    # Reconstruimos
    imagen_reconstruida = np.zeros((alto_tot, ancho_tot), dtype=int)
    
    print(f"--- Recuperando y Ensamblando ({len(banco_llaves)} partes) ---")
    
    for i, item in enumerate(banco_llaves):
        k1 = item['k1']
        k2 = item['k2']
        r, c, h_real, w_real = item['coords']
        bloque_original = item.get('original', None) # Recuperamos el original para comparar
        
        # Desencriptamos
        # Usamos el paquete completo, pero con la llave de ese bloque.
        campo_recuperado = desencriptar_drpe(paquete_optico, k1, k2)
        

        img_ruidosa = np.abs(campo_recuperado)

        # 4. Limpieza simple para reconstrucción (Normalizar 0-1)
        vmin, vmax = img_ruidosa.min(), img_ruidosa.max()
        if vmax > vmin:
            img_norm = (img_ruidosa - vmin) / (vmax - vmin)
        else:
            img_norm = img_ruidosa
            
        # Umbral simple
        img_ruidosa = np.where(img_norm > np.mean(img_norm), 1, 0).astype(np.uint8)

        #img_suave = cv2.medianBlur(img_ruidosa, 3)
        
        p1, p99 = np.percentile(img_ruidosa, (1, 99))
        img_clipped = np.clip(img_ruidosa, p1, p99)

        # 3. Normalización (0 a 255)
        # Usamos los percentiles recortados como límites
        if p99 > p1:
            img_norm = (img_clipped - p1) / (p99 - p1)
        else:
            img_norm = img_clipped

        # Umbral simple
        bloque_limpio = np.where(img_norm > 0.9, 1, 0)
        
        # Graficamos
        if ver_paso_a_paso:
            plt.figure(figsize=(12, 4))
            
            # A: Original (Lo que queremos obtener)
            plt.subplot(1, 3, 1)
            plt.title(f"Bloque {i+1} Original (Ground Truth)")
            plt.imshow(bloque_original, cmap='gray')
            plt.axis('off')
            
            # B: El Sistema Encriptado (Lo que viaja por la fibra/aire)
            # Mostramos magnitud del campo complejo total
            plt.subplot(1, 3, 2)
            plt.title("Sistema Encriptado (Ruido Total)")
            plt.imshow(np.abs(paquete_optico), cmap='gray') # 'inferno' resalta bien el ruido
            plt.axis('off')
            
            # C: Lo recuperado
            plt.subplot(1, 3, 3)
            plt.title(f"Bloque {i+1} Recuperado")
            plt.imshow(bloque_limpio, cmap='gray')
            plt.axis('off')
            
            plt.suptitle(f"Proceso de Desencriptación - Bloque {i+1}", fontsize=14)
            plt.tight_layout()
            plt.show()
        
        # 5. Pegar en el lienzo final
        # Necesitamos saber el tamaño de los bloques usados en la encriptación
        h_bloque, w_bloque = img_ruidosa.shape
        
        y_pos = r * h_bloque
        x_pos = c * w_bloque
        
        # Recortamos el padding extra que agregamos antes (si hubo)
        bloque_util = bloque_limpio[0:h_real, 0:w_real]
        
        imagen_reconstruida[y_pos : y_pos + h_real, x_pos : x_pos + w_real] = bloque_util
        
    return imagen_reconstruida