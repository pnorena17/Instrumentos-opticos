import segno
import numpy as np
import math

def dividir_imagen(matriz, filas, cols):
    alto, ancho = matriz.shape
    h_bloque = math.ceil(alto / filas)
    w_bloque = math.ceil(ancho / cols)
    bloques = []
    
    for i in range(filas):
        for j in range(cols):
            y1 = i * h_bloque
            y2 = min((i + 1) * h_bloque, alto)
            x1 = j * w_bloque
            x2 = min((j + 1) * w_bloque, ancho)
            bloque = matriz[y1:y2, x1:x2]
            bloques.append(bloque)
    return bloques, h_bloque, w_bloque

def generar_mosaico_raw(imagen_binaria, filas=6, cols=8, escala=2):
    """
    Genera una matriz de QRs donde cada píxel es un carácter '0' o '1'.
    """
    bloques, _, _ = dividir_imagen(imagen_binaria, filas, cols)
    qr_matrices = []
    
    print(f"Generando {len(bloques)} códigos QR modo Numérico...")

    for i, bloque in enumerate(bloques):
        # 1. Aplanar matriz
        flat = bloque.flatten().astype(int)
        
        # 2. Convertir a string puro "010101..."
        # Esto activa el modo Numérico de QR (muy denso)
        payload_str = "".join(map(str, flat))
        
        h, w = bloque.shape
        
        # 3. Header: "IDX:H:W:TF:TC:DATOS"
        # Usamos ':' como separador
        contenido = f"{i}:{h}:{w}:{filas}:{cols}:{payload_str}"
        
        try:
            # error='L' permite más datos. Version=None deja que segno elija la mejor (probablemente 20-40)
            qr = segno.make(contenido, error='L') 
            
            # Convertir a matriz numpy con borde
            # border=4 es más seguro para lectura automática en alta densidad
            iterador = qr.matrix_iter(border=4) 
            matriz_qr = np.array([list(row) for row in iterador], dtype=int)
            
            # Escalar si es necesario
            if escala > 1:
                matriz_qr = np.repeat(np.repeat(matriz_qr, escala, axis=0), escala, axis=1)
                
            qr_matrices.append(matriz_qr)
            
        except Exception as e:
            print(f"Error en bloque {i}. Demasiados datos. Aumenta el número de filas/cols.")
            print(f"Longitud intentada: {len(contenido)}")
            return None

    # Unir QRs en un solo panel gigante
    # Asumimos que todos los QRs tienen tamaños similares, tomamos el max
    if not qr_matrices: return None
    
    max_h = max(q.shape[0] for q in qr_matrices)
    max_w = max(q.shape[1] for q in qr_matrices)
    
    panel_h = filas * max_h
    panel_w = cols * max_w
    panel = np.zeros((panel_h, panel_w), dtype=int) # Fondo negro (o blanco según lógica)
    
    # Rellenar con blanco (0) o negro (1) según tu preferencia visual final
    # Aquí asumimos 0=blanco para el lienzo base si usas imshow 'gray'
    panel.fill(0) 

    for k, qr in enumerate(qr_matrices):
        r = k // cols
        c = k % cols
        y = r * max_h
        x = c * max_w
        h_qr, w_qr = qr.shape
        panel[y:y+h_qr, x:x+w_qr] = qr

    return panel