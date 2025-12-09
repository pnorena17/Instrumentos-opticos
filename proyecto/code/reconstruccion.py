from pyzbar.pyzbar import decode, ZBarSymbol
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter

def leer_qr_individual(matriz_qr, ver_debug=False):
    """
    Versión MEJORADA: Usa recorte de percentiles y Otsu para adaptarse a cualquier nivel de ruido.
    """
    if matriz_qr is None: return None

    # 1. Obtener Magnitud
    if isinstance(matriz_qr, np.ndarray):
        img_data = np.abs(matriz_qr)
    else:
        return None

    # 2. Recorte de Picos (Percentile Clipping)
    # Ignoramos el 1% más oscuro y el 1% más brillante (ruido/hot pixels)
    # Esto soluciona que una imagen se vea negra por culpa de un solo pixel brillante.
    p1, p99 = np.percentile(img_data, (1, 99))
    img_clipped = np.clip(img_data, p1, p99)

    # 3. Normalización (0 a 255)
    # Usamos los percentiles recortados como límites
    if p99 > p1:
        img_norm = (img_clipped - p1) / (p99 - p1)
    else:
        img_norm = img_clipped
        
    img_uint8 = (img_norm * 255).astype(np.uint8)

    # 4. Filtro de Mediana (Anti-Speckle)
    # Usamos OpenCV (es más rápido que scipy y ya tienes cv2 importado)
    img_suave = cv2.medianBlur(img_uint8, 3)

    # 5. Binarización Automática (Otsu)
    # OpenCV calcula el umbral ideal estadísticamente.
    # Usamos THRESH_BINARY_INV asumiendo que el fondo suele ser oscuro en óptica.
    # Si tu QR es negro sobre blanco, quita el _INV.
    thresh_val, img_bin = cv2.threshold(img_suave, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


    
    # --- VISUALIZACIÓN DE DEBUG ---
    if ver_debug:
        plt.figure(figsize=(10, 3))
        plt.subplot(1, 3, 1); plt.title("Original"); plt.imshow(img_clipped, cmap='gray')
        plt.subplot(1, 3, 2); plt.title("Filtro de mediana"); plt.imshow(img_suave, cmap='gray')
        plt.subplot(1, 3, 3); plt.title(f"QR para decodificación"); plt.imshow(img_bin, cmap='gray')
        plt.tight_layout(); plt.show()
    # ------------------------------

# --- PASO NUEVO: REDIMENSIONAMIENTO INTELIGENTE ---
    h, w = img_bin.shape
    
    # ZBar funciona mejor con imágenes entre 400 y 800 píxeles de lado.
    # Si tu imagen es gigante (ej. 2048), la reducimos para facilitar la lectura.
    TARGET_SIZE = 600
    
    if h > TARGET_SIZE or w > TARGET_SIZE:
        # Calculamos el factor de reducción para mantener la proporción
        factor = TARGET_SIZE / max(h, w)
        new_h = int(h * factor)
        new_w = int(w * factor)
        
        # IMPORTANTE: Usar INTER_AREA para reducir. 
        # Esto promedia los píxeles y ELIMINA RUIDO automáticamente.
        img_lista_para_leer = cv2.resize(img_bin, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        # Si es muy pequeña (ej. < 200), la agrandamos (lo que ya tenías)
        scale = 3
        img_lista_para_leer = cv2.resize(img_bin, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)

    # 5. Borde Blanco (Quiet Zone)
    img_final = cv2.copyMakeBorder(
        img_lista_para_leer, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255
    )

    img_final = median_filter(img_final, size=3)
    img_final = median_filter(img_final, size=3)

    if ver_debug:
        plt.figure()
        plt.imshow(img_final, cmap='gray')
        plt.title("QR procesado")
        plt.show()

    # 7. Intentar Leer (Directo e Invertido)
    # Intento A: Tal cual salió de Otsu
    decoded_objects = decode(Image.fromarray(img_final), symbols=[ZBarSymbol.QRCODE])
    
    # Intento B: Invertido (Por si Otsu eligió el fondo como figura)
    if not decoded_objects:
        img_inv = 255 - img_final
        decoded_objects = decode(Image.fromarray(img_inv), symbols=[ZBarSymbol.QRCODE])
    
    if not decoded_objects:
        return None
    
    # --- PARSEO DE DATOS (Igual que antes) ---
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
            return {'idx': idx, 'bloque': bloque, 'h': h, 'w': w, 'filas_tot': filas_tot, 'cols_tot': cols_tot}
    except Exception:
        pass
        
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