import numpy as np
import math
from encript_image import encriptar_drpe, desencriptar_drpe

def dividir_en_bloques_iguales(imagen, filas, cols):
    """
    Divide una imagen grande en una lista de bloques pequeños.
    IMPORTANTE: Si la división no es exacta, rellena con ceros (padding)
    para que TODOS los bloques tengan el mismo tamaño (shape).
    """
    alto, ancho = imagen.shape
    
    # Tamaño objetivo de cada bloque
    # Usamos ceil para asegurar que cubra toda la imagen
    h_bloque = math.ceil(alto / filas)
    w_bloque = math.ceil(ancho / cols)
    
    bloques = []
    coords = [] # Guardamos donde iba cada bloque para rearmar
    
    for i in range(filas):
        for j in range(cols):
            y_inicio = i * h_bloque
            x_inicio = j * w_bloque
            
            # Recorte base
            # Si nos salimos de la imagen, numpy devuelve lo que haya hasta el borde
            recorte = imagen[y_inicio : y_inicio + h_bloque, x_inicio : x_inicio + w_bloque]
            
            # --- PADDING DE SEGURIDAD ---
            # Para sumar matrices, todas deben medir EXACTAMENTE lo mismo.
            # Si el recorte del borde es más pequeño, lo rellenamos con negro.
            h_actual, w_actual = recorte.shape
            
            if h_actual < h_bloque or w_actual < w_bloque:
                bloque_pad = np.zeros((h_bloque, w_bloque), dtype=imagen.dtype)
                bloque_pad[:h_actual, :w_actual] = recorte
                bloque_final = bloque_pad
            else:
                bloque_final = recorte
                
            bloques.append(bloque_final)
            coords.append((y_inicio, x_inicio, h_actual, w_actual)) # Guardamos tamaño real útil
            
    return bloques, coords, (alto, ancho)

def multiplexar_imagen_en_partes(imagen_qr, filas_grid=3, cols_grid=3, radio_pupila=None):
    """
    1. Parte la imagen del QR en filas*cols pedazos.
    2. Encripta cada pedazo.
    3. Suma todo en un paquete pequeño.
    """
    
    # 1. Dividir
    lista_partes, coordenadas, dim_original = dividir_en_bloques_iguales(imagen_qr, filas_grid, cols_grid)
    
    # 2. Preparar el paquete (del tamaño de UN BLOQUE, no de la imagen entera)
    h_block, w_block = lista_partes[0].shape
    paquete_optico = np.zeros((h_block, w_block), dtype=complex)
    banco_llaves = []
    
    print(f"--- Multiplexando imagen en {len(lista_partes)} partes ({filas_grid}x{cols_grid}) ---")
    print(f"Tamaño original: {imagen_qr.shape} -> Tamaño paquete: {paquete_optico.shape}")

    # 3. Encriptar y Sumar
    for i, bloque in enumerate(lista_partes):
        
        # DRPE normal sobre el bloque
        campo_encriptado, k1, k2, _ = encriptar_drpe(bloque, radio_pupila=radio_pupila)
        
        # Suma coherente
        paquete_optico += campo_encriptado
        
        # Guardamos llaves y coordenadas para saber dónde poner este pedazo luego
        banco_llaves.append({
            'idx': i,
            'k1': k1,
            'k2': k2,
            'coords': coordenadas[i] # (y, x, h_real, w_real)
        })
        print(f"   > Parte {i+1}/{len(lista_partes)} encriptada.")
        
    # Retornamos dimensiones originales para poder crear el lienzo al volver
    return paquete_optico, banco_llaves, dim_original

def recuperar_y_ensamblar_imagen(paquete_optico, banco_llaves, dim_original):
    """
    1. Recorre las llaves.
    2. Desencripta cada pedazo (limpiando ruido).
    3. Pega el pedazo en el lugar correcto del lienzo original.
    """
    alto_tot, ancho_tot = dim_original
    
    # Lienzo reconstruido
    imagen_reconstruida = np.zeros((alto_tot, ancho_tot), dtype=int)
    
    print(f"--- Recuperando y Ensamblando ({len(banco_llaves)} partes) ---")
    
    for item in banco_llaves:
        idx = item['idx']
        k1 = item['k1']
        k2 = item['k2']
        y, x, h_real, w_real = item['coords']
        
        # 1. Desencriptar
        campo = desencriptar_drpe(paquete_optico, k1, k2)
        
        # 2. Limpieza Básica (Absoluto + Normalizar + Binarizar)
        img_ruidosa = np.abs(campo)
        
        # Normalizar 0-1
        vmin, vmax = img_ruidosa.min(), img_ruidosa.max()
        if vmax > vmin:
            img_norm = (img_ruidosa - vmin) / (vmax - vmin)
        else:
            img_norm = img_ruidosa
            
        # Binarizar (Umbral 0.5 suele funcionar bien para QR)
        # Nota: Aquí asumimos que el ruido de fondo es menor que la señal.
        # Con 9 partes (3x3), el ruido será considerable, tal vez necesites ajustar el umbral.
        bloque_limpio = np.where(img_norm > 0.5, 1, 0)
        
        # 3. Pegar en el lienzo
        # Ojo: El bloque recuperado puede tener padding extra si estaba en el borde.
        # Usamos h_real y w_real para recortar solo la parte útil.
        imagen_reconstruida[y : y + h_real, x : x + w_real] = bloque_limpio[:h_real, :w_real]
        
        # print(f"   > Parte {idx} recuperada.")
        
    return imagen_reconstruida