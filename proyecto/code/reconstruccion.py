from pyzbar.pyzbar import decode, ZBarSymbol
from PIL import Image
import cv2
import numpy as np

def leer_qr_individual(matriz_qr):
    """
    Intenta leer un SOLO QR con pre-procesamiento agresivo.
    """
    # 1. Convertir a uint8 (0-255)
    # Si viene de óptica, puede ser float o complex. Tomamos magnitud y normalizamos.
    if isinstance(matriz_qr, np.ndarray):
        img_data = np.abs(matriz_qr) # Asegurar reales positivos
        
        # Normalizar a 0-255
        if img_data.max() > 0:
            img_data = (img_data / img_data.max()) * 255
        
        img_uint8 = img_data.astype(np.uint8)
        
        # --- PRE-PROCESAMIENTO CLAVE ---
        
        # A. UMBRALIZADO (Binarización)
        # Esto elimina el gris/ruido y deja solo BLANCO o NEGRO puro.
        _, img_bin = cv2.threshold(img_uint8, 127, 255, cv2.THRESH_BINARY)
        
        # B. ZOOM (Upscaling)
        # Los lectores fallan con QRs pequeños. Lo agrandamos 2x o 3x.
        h, w = img_bin.shape
        scale = 3
        img_grande = cv2.resize(img_bin, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
        
        # C. BORDE BLANCO (Quiet Zone)
        # ZBar NECESITA un marco blanco alrededor.
        # Si tu QR es negro sobre blanco (estándar), agregamos borde blanco (255).
        # Si tu QR es blanco sobre negro (negativo), ZBar a veces falla.
        # Vamos a probar leerlo tal cual, y si falla, lo invertimos.
        
        img_con_borde = cv2.copyMakeBorder(
            img_grande, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255
        )
        
        # Convertir a PIL para ZBar
        img_pil = Image.fromarray(img_con_borde)
        
    else:
        return None

    # 2. Decodificar
    # Usamos symbols=[ZBarSymbol.QRCODE] para que NO busque PDF417 y quite las advertencias
    decoded_objects = decode(img_pil, symbols=[ZBarSymbol.QRCODE])
    
    # INTENTO 2: Si falla, Invertimos el color (Negativo)
    if not decoded_objects:
        img_inv = 255 - img_con_borde
        img_pil_inv = Image.fromarray(img_inv)
        decoded_objects = decode(img_pil_inv, symbols=[ZBarSymbol.QRCODE])
    
    if not decoded_objects:
        return None
    
    # Procesar datos
    obj = decoded_objects[0]
    
    try:
        texto = obj.data.decode('utf-8')
        partes = texto.split(':')
        
        if len(partes) < 6: return None
        
        idx = int(partes[0])
        h = int(partes[1])
        w = int(partes[2])
        filas_tot = int(partes[3])
        cols_tot = int(partes[4])
        raw_str = partes[5]
        
        array_plano = np.array(list(map(int, raw_str)))
        
        if len(array_plano) == h * w:
            bloque = array_plano.reshape((h, w))
            return {
                'idx': idx, 
                'bloque': bloque, 
                'h': h, 
                'w': w,
                'filas_tot': filas_tot,
                'cols_tot': cols_tot
            }
    except Exception as e:
        print(f"Error parseando QR: {e}")
        
    return None

def reconstruir_mosaico(lista_qrs_limpios):
    """
    Recibe una LISTA de matrices QR (ya desencriptadas y limpias).
    Lee cada una, extrae su posición y rearma la imagen final.
    """
    datos_bloques = []
    config_grid = None
    
    print(f"Intentando leer {len(lista_qrs_limpios)} QRs...")

    # 1. Leer cada QR de la lista
    for i, qr in enumerate(lista_qrs_limpios):
        datos = leer_qr_individual(qr)
        if datos:
            datos_bloques.append(datos)
            if config_grid is None:
                config_grid = (datos['filas_tot'], datos['cols_tot'])
        else:
            print(f" - QR #{i} no se pudo leer (daño excesivo o ruido).")

    if not datos_bloques:
        print("No se pudo recuperar ningún bloque válido.")
        return None

    # 2. Ordenar por índice para asegurar el pegado correcto
    datos_bloques.sort(key=lambda x: x['idx'])
    
    # 3. Preparar lienzo final
    # Usamos la configuración del primer bloque leído para saber el tamaño total
    tf, tc = config_grid
    
    # Necesitamos saber el tamaño de los bloques para crear el lienzo.
    # Asumimos que todos son casi iguales, usamos el máximo encontrado.
    max_h = max(b['h'] for b in datos_bloques)
    max_w = max(b['w'] for b in datos_bloques)
    
    # Lienzo vacío
    alto_fin = tf * max_h
    ancho_fin = tc * max_w
    imagen_final = np.zeros((alto_fin, ancho_fin), dtype=int)
    
    print(f"Reconstruyendo imagen de {alto_fin}x{ancho_fin} px con {len(datos_bloques)} bloques...")

    # 4. Pegar bloques en su sitio
    for b in datos_bloques:
        idx = b['idx']
        h, w = b['h'], b['w']
        bloque = b['bloque']
        
        # Calcular fila y columna basada en el índice lineal
        r = idx // tc
        c = idx % tc
        
        # Coordenadas pixel (usando el tamaño real del bloque o el maximo? 
        # Usamos el tamaño del bloque para pegarlo, pero la posición basada en max_h/max_w 
        # para mantener la grilla alineada si hay pequeñas variaciones)
        y = r * max_h 
        x = c * max_w
        
        # Pegar (con cuidado de no salirnos si el bloque es raro)
        # Recortamos si se pasa, o rellenamos
        y_end = min(y + h, alto_fin)
        x_end = min(x + w, ancho_fin)
        
        # Ajustar bloque si se recorta
        h_eff = y_end - y
        w_eff = x_end - x
        
        imagen_final[y:y_end, x:x_end] = bloque[:h_eff, :w_eff]

    return imagen_final