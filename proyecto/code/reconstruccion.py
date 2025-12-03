from pyzbar.pyzbar import decode
from PIL import Image
import numpy as np

def leer_qr_individual(matriz_qr):
    """
    Intenta leer un SOLO QR (matriz numpy binaria o gris).
    Retorna el diccionario de datos o None si falla.
    """
    # 1. Preparar imagen para pyzbar (0-255 uint8)
    # Asumimos que la entrada es 0 (negro/blanco) y 1 (blanco/negro)
    # Pyzbar lee mejor códigos oscuros sobre fondo claro.
    # Si tu matriz tiene 1=Luz/Blanco y 0=Negro -> Invertimos: (1-x)*255
    if isinstance(matriz_qr, np.ndarray):
        # Normalizar si no es binaria pura (por si viene de DRPE con ruido)
        if matriz_qr.max() <= 1.0:
            img_uint8 = ((1 - matriz_qr) * 255).astype(np.uint8)
        else:
            img_uint8 = (255 - matriz_qr).astype(np.uint8)
            
        img = Image.fromarray(img_uint8)
    else:
        return None

    # 2. Decodificar
    decoded_objects = decode(img)
    
    if not decoded_objects:
        return None
    
    # Tomamos el primero (debería haber solo uno por matriz)
    obj = decoded_objects[0]
    
    try:
        texto = obj.data.decode('utf-8')
        # Parsear "IDX:H:W:TF:TC:DATOS"
        partes = texto.split(':')
        
        if len(partes) < 6: return None
        
        idx = int(partes[0])
        h = int(partes[1])
        w = int(partes[2])
        filas_tot = int(partes[3])
        cols_tot = int(partes[4])
        raw_str = partes[5]
        
        # Reconstruir bloque de imagen
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
        print(f"Error parseando QR {idx if 'idx' in locals() else '?'}: {e}")
        
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