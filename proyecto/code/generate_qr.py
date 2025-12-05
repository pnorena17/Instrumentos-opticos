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
            x1 = int(x1) # Aseguramos enteros
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)
            
            bloque = matriz[y1:y2, x1:x2]
            bloques.append(bloque)
    return bloques, h_bloque, w_bloque

def generar_lista_qrs(imagen_binaria, filas=6, cols=8, escala=10):
    """
    Genera una LISTA de matrices QR individuales.
    Cada QR contiene en su header el índice 'i' para saber su coordenada.
    """
    bloques, _, _ = dividir_imagen(imagen_binaria, filas, cols)
    qr_matrices = []
    
    print(f"Generando {len(bloques)} códigos QR individuales...")

    for i, bloque in enumerate(bloques):
        # 1. Aplanar matriz
        flat = bloque.flatten().astype(int)
        
        # 2. Convertir a string "010101..."
        payload_str = "".join(map(str, flat))
        
        h, w = bloque.shape
        
        # 3. Header: "IDX:H:W:TF:TC:DATOS"
        # IDX (i) es clave: con él calculas la fila (i // cols) y columna (i % cols)
        contenido = f"{i}:{h}:{w}:{filas}:{cols}:{payload_str}"
        
        try:
            # RECOMENDACIÓN: Usar error='H' para óptica (soporta 30% daño)
            # Usar 'L' (7%) es arriesgado con el ruido speckle, pero lo dejo a tu elección.
            qr = segno.make(contenido, error='H') 
            
            # Convertir a matriz numpy con borde
            iterador = qr.matrix_iter(border=4) 
            matriz_qr = np.array([list(row) for row in iterador], dtype=int)
            
            # Escalar
            if escala > 1:
                matriz_qr = np.repeat(np.repeat(matriz_qr, escala, axis=0), escala, axis=1)
                
            qr_matrices.append(matriz_qr)
            
        except Exception as e:
            print(f"Error en bloque {i}: {e}")
            return None

    # Retornamos la lista directa.
    return qr_matrices